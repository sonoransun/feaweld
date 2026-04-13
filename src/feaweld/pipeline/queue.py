"""SQLite-backed persistent job queue for analysis cases.

Provides :class:`AnalysisJobQueue`, a durable FIFO/priority queue that
stores :class:`~feaweld.pipeline.workflow.AnalysisCase` jobs in an
SQLite database.  Workers pull pending jobs via :meth:`worker_loop` and
write :class:`~feaweld.pipeline.workflow.WorkflowResult` back.

Design notes
------------
* Job IDs are UUID-4 strings.
* The case is stored as JSON (from Pydantic ``model_dump``).
* Results are stored as ``pickle`` blobs — they contain numpy arrays
  that do not round-trip through JSON.
* The database uses WAL mode for safe concurrent reads/writes from
  the same process (multiple threads) or from a single writer +
  multiple readers across processes.
"""

from __future__ import annotations

import enum
import json
import pickle
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feaweld.core.logging import get_logger

if TYPE_CHECKING:
    from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult

logger = get_logger(__name__)

_SCHEMA_VERSION = 1

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'pending',
    priority   INTEGER NOT NULL DEFAULT 0,
    case_json  TEXT NOT NULL,
    result     BLOB,
    error_msg  TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_priority
    ON jobs (status, priority DESC, created_at ASC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class JobStatus(str, enum.Enum):
    """Lifecycle states for a queued analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Queue implementation
# ---------------------------------------------------------------------------


class AnalysisJobQueue:
    """Persistent job queue backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created on first use.
    """

    def __init__(self, db_path: Path | str = "feaweld_jobs.db") -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        # Eagerly initialize the schema on the calling thread.
        self._ensure_schema(self._get_conn())

    # -- connection management ----------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """Create tables if they don't exist."""
        conn.executescript(_CREATE_TABLE_SQL)
        # Record schema version.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(_SCHEMA_VERSION)),
        )
        conn.commit()

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- job submission -----------------------------------------------------

    def submit(
        self,
        case: AnalysisCase,
        priority: int = 0,
    ) -> str:
        """Submit an analysis case for execution.

        Parameters
        ----------
        case:
            The analysis case to run.
        priority:
            Higher values are dequeued first.

        Returns
        -------
        str
            A UUID-4 job identifier.
        """
        job_id = str(uuid.uuid4())
        case_json = json.dumps(case.model_dump(mode="json"), default=str)
        now = time.time()

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, case_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, JobStatus.PENDING.value, priority, case_json, now),
        )
        conn.commit()
        logger.info("Job submitted: %s (priority=%d)", job_id, priority)
        return job_id

    # -- status queries -----------------------------------------------------

    def get_status(self, job_id: str) -> JobStatus:
        """Return the current status of a job.

        Raises
        ------
        KeyError
            If *job_id* is not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job ID: {job_id}")
        return JobStatus(row["status"])

    def get_result(self, job_id: str) -> WorkflowResult | None:
        """Return the :class:`WorkflowResult` for a completed job.

        Returns ``None`` if the job has not finished or the result blob
        is missing.

        Raises
        ------
        KeyError
            If *job_id* is not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT status, result FROM jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job ID: {job_id}")
        if row["result"] is None:
            return None
        try:
            return pickle.loads(row["result"])
        except Exception as exc:
            logger.warning(
                "Failed to unpickle result for job %s: %s", job_id, exc,
            )
            return None

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List queued jobs.

        Parameters
        ----------
        status:
            If provided, filter to jobs in this state.
        limit:
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            Each dict has keys: ``job_id``, ``status``, ``priority``,
            ``created_at``, ``started_at``, ``finished_at``,
            ``error_msg``.
        """
        conn = self._get_conn()
        if status is not None:
            rows = conn.execute(
                "SELECT job_id, status, priority, created_at, started_at, "
                "finished_at, error_msg "
                "FROM jobs WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT job_id, status, priority, created_at, started_at, "
                "finished_at, error_msg "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.  Returns ``True`` if the job was pending
        and is now removed, ``False`` otherwise."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM jobs WHERE job_id = ? AND status = ?",
            (job_id, JobStatus.PENDING.value),
        )
        conn.commit()
        cancelled = cur.rowcount > 0
        if cancelled:
            logger.info("Job cancelled: %s", job_id)
        return cancelled

    def purge(self, status: JobStatus | None = None) -> int:
        """Delete jobs.  If *status* is given, only jobs in that state
        are removed.  Returns the number of rows deleted."""
        conn = self._get_conn()
        if status is not None:
            cur = conn.execute(
                "DELETE FROM jobs WHERE status = ?", (status.value,),
            )
        else:
            cur = conn.execute("DELETE FROM jobs")
        conn.commit()
        return cur.rowcount

    # -- worker loop --------------------------------------------------------

    def worker_loop(
        self,
        max_concurrent: int = 1,
        poll_interval: float = 1.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Pull and execute jobs.  **Blocking.**

        Parameters
        ----------
        max_concurrent:
            Number of jobs to run simultaneously (currently sequential;
            concurrency is reserved for a future thread-pool extension).
        poll_interval:
            Seconds to wait between polling when the queue is empty.
        stop_event:
            If provided, the loop exits when this event is set.

        The loop runs forever until *stop_event* is set or the process
        is interrupted.
        """
        logger.info(
            "Worker loop started (max_concurrent=%d, poll=%.1fs)",
            max_concurrent, poll_interval,
        )

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("Worker loop: stop event received")
                break

            job = self._claim_next()
            if job is None:
                time.sleep(poll_interval)
                continue

            job_id: str = job["job_id"]
            logger.info("Worker picked up job %s", job_id)

            try:
                from feaweld.pipeline.workflow import AnalysisCase, run_analysis

                case = AnalysisCase(**json.loads(job["case_json"]))
                result = run_analysis(case)
                self._mark_completed(job_id, result)
            except Exception as exc:
                logger.error("Job %s failed: %s", job_id, exc)
                self._mark_failed(job_id, str(exc))

    # -- internal -----------------------------------------------------------

    def _claim_next(self) -> dict[str, Any] | None:
        """Atomically claim the highest-priority pending job.

        Returns the row as a dict, or ``None`` if the queue is empty.
        """
        conn = self._get_conn()
        now = time.time()
        # Single atomic UPDATE + SELECT via RETURNING (SQLite 3.35+).
        try:
            row = conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? "
                "WHERE job_id = ("
                "  SELECT job_id FROM jobs WHERE status = ? "
                "  ORDER BY priority DESC, created_at ASC LIMIT 1"
                ") RETURNING *",
                (JobStatus.RUNNING.value, now, JobStatus.PENDING.value),
            ).fetchone()
            conn.commit()
        except sqlite3.OperationalError:
            # Fallback for SQLite < 3.35 (no RETURNING).
            row = conn.execute(
                "SELECT job_id, case_json FROM jobs WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC LIMIT 1",
                (JobStatus.PENDING.value,),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = ? "
                    "WHERE job_id = ?",
                    (JobStatus.RUNNING.value, now, row["job_id"]),
                )
                conn.commit()

        if row is None:
            return None
        return dict(row)

    def _mark_completed(self, job_id: str, result: WorkflowResult) -> None:
        """Record a successful result."""
        conn = self._get_conn()
        try:
            blob = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning(
                "Could not pickle result for job %s: %s", job_id, exc,
            )
            blob = None
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, finished_at = ? "
            "WHERE job_id = ?",
            (JobStatus.COMPLETED.value, blob, time.time(), job_id),
        )
        conn.commit()
        logger.info("Job %s completed", job_id)

    def _mark_failed(self, job_id: str, error_msg: str) -> None:
        """Record a failed job."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ?, error_msg = ?, finished_at = ? "
            "WHERE job_id = ?",
            (JobStatus.FAILED.value, error_msg, time.time(), job_id),
        )
        conn.commit()
        logger.info("Job %s marked as failed", job_id)
