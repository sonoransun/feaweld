"""Checkpoint / restart support for long-running analysis pipelines.

Persists :class:`~feaweld.pipeline.dag.PipelineContext` state to disk
after each completed stage so that a crashed or interrupted run can
resume from the last successful stage.

Storage format
--------------
A checkpoint directory contains:

* ``meta.json`` — completed stage names, case config hash, and a
  mapping of context keys to their serialization type.
* ``<key>.npz`` — for values that are numpy arrays.
* ``<key>.pkl`` — for everything else (pickled).

Checksums
---------
``meta.json`` stores the SHA-256 of the case configuration so the
loader can warn when a checkpoint was created from a different case.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from feaweld.core.logging import get_logger
from feaweld.pipeline.dag import PipelineContext

logger = get_logger(__name__)

# Sentinel used when a context value is ``None`` — we still want to
# record that the key exists.
_NONE_SENTINEL = "__feaweld_none__"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    context: PipelineContext,
    path: Path | str,
    completed_stages: list[str],
    *,
    case_config: dict[str, Any] | None = None,
) -> None:
    """Serialize pipeline state after each stage.

    Parameters
    ----------
    context:
        The live :class:`PipelineContext`.
    path:
        Directory to write the checkpoint into.  Created if it does not
        exist; **existing contents are overwritten**.
    completed_stages:
        Ordered list of stage names that have finished successfully.
    case_config:
        Optional dict representation of the :class:`AnalysisCase` for
        integrity checking on reload.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    key_types: dict[str, str] = {}  # key -> "npz" | "pkl" | "none"

    for key in context.keys():
        value = context[key]

        if value is None:
            key_types[key] = "none"
            continue

        safe_name = _safe_filename(key)

        if isinstance(value, np.ndarray):
            try:
                np.savez_compressed(path / f"{safe_name}.npz", data=value)
                key_types[key] = "npz"
                continue
            except Exception as exc:
                logger.debug(
                    "numpy save failed for key %r, falling back to pickle: %s",
                    key, exc,
                )

        # Fallback: pickle.
        try:
            with open(path / f"{safe_name}.pkl", "wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            key_types[key] = "pkl"
        except Exception as exc:
            logger.warning(
                "Could not serialize context key %r: %s", key, exc,
            )

    config_hash = _hash_config(case_config) if case_config else ""

    meta = {
        "completed_stages": completed_stages,
        "config_hash": config_hash,
        "key_types": key_types,
        "errors": context.errors,
    }
    meta_path = path / "meta.json"
    tmp_meta = path / "meta.json.tmp"
    try:
        with open(tmp_meta, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        tmp_meta.replace(meta_path)
    except Exception as exc:
        logger.error("Failed to write checkpoint meta: %s", exc)
        # Clean up partial temp file.
        tmp_meta.unlink(missing_ok=True)
        raise

    logger.info(
        "Checkpoint saved to %s (%d keys, %d completed stages)",
        path, len(key_types), len(completed_stages),
    )


def load_checkpoint(
    path: Path | str,
    *,
    case_config: dict[str, Any] | None = None,
) -> tuple[PipelineContext, list[str]]:
    """Load a checkpoint directory.

    Parameters
    ----------
    path:
        Directory previously written by :func:`save_checkpoint`.
    case_config:
        If provided, its hash is compared against the stored hash.  A
        mismatch logs a warning but does **not** raise.

    Returns
    -------
    (context, completed_stages)
        A freshly-constructed :class:`PipelineContext` populated with the
        deserialized data, and the list of completed stage names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or ``meta.json`` is missing.
    CheckpointCorruptedError
        If ``meta.json`` is unparseable.
    """
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {path} (meta.json missing)"
        )

    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise CheckpointCorruptedError(
            f"Checkpoint meta.json is corrupted: {exc}"
        ) from exc

    completed_stages: list[str] = meta.get("completed_stages", [])
    key_types: dict[str, str] = meta.get("key_types", {})
    stored_hash = meta.get("config_hash", "")

    # Integrity check.
    if case_config and stored_hash:
        current_hash = _hash_config(case_config)
        if current_hash != stored_hash:
            logger.warning(
                "Checkpoint config hash mismatch: checkpoint was created "
                "from a different AnalysisCase.  Proceeding anyway, but "
                "results may be inconsistent."
            )

    context = PipelineContext()

    for key, kind in key_types.items():
        safe_name = _safe_filename(key)

        if kind == "none":
            context[key] = None
            continue

        if kind == "npz":
            npz_path = path / f"{safe_name}.npz"
            if not npz_path.exists():
                logger.warning(
                    "Checkpoint missing file for key %r: %s", key, npz_path,
                )
                continue
            try:
                with np.load(npz_path) as data:
                    context[key] = data["data"]
            except Exception as exc:
                logger.warning(
                    "Failed to load numpy checkpoint for key %r: %s",
                    key, exc,
                )
            continue

        if kind == "pkl":
            pkl_path = path / f"{safe_name}.pkl"
            if not pkl_path.exists():
                logger.warning(
                    "Checkpoint missing file for key %r: %s", key, pkl_path,
                )
                continue
            try:
                with open(pkl_path, "rb") as fh:
                    context[key] = pickle.load(fh)
            except Exception as exc:
                logger.warning(
                    "Failed to load pickle checkpoint for key %r: %s",
                    key, exc,
                )
            continue

        logger.warning("Unknown checkpoint kind %r for key %r", kind, key)

    # Restore errors.
    for err in meta.get("errors", []):
        context.add_error(err)

    logger.info(
        "Checkpoint loaded from %s (%d keys, %d completed stages)",
        path, len(key_types), len(completed_stages),
    )
    return context, completed_stages


def clear_checkpoint(path: Path | str) -> None:
    """Remove a checkpoint directory if it exists."""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
        logger.info("Checkpoint cleared: %s", path)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CheckpointCorruptedError(Exception):
    """Raised when a checkpoint cannot be loaded due to corruption."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_filename(key: str) -> str:
    """Convert an arbitrary context key to a filesystem-safe name."""
    return key.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _hash_config(config: dict[str, Any] | None) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serializable config
    dict."""
    if config is None:
        return ""
    try:
        canonical = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
    except Exception:
        return ""
