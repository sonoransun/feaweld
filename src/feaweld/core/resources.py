"""Linux resource monitoring and subprocess limits.

Provides utilities for tracking memory usage via /proc, enforcing
resource limits on child processes (CalculiX), and detecting memory
pressure so the pipeline can switch to memory-mapped arrays.
"""

from __future__ import annotations

import os
import resource
import subprocess
from pathlib import Path

from feaweld.core.logging import get_logger

logger = get_logger(__name__)


def get_memory_usage() -> dict[str, int]:
    """Return current process memory usage in KB.

    Reads ``/proc/self/status`` on Linux.  Returns an empty dict on
    platforms where that file is unavailable.

    Keys returned (when available):
        VmRSS, VmPeak, VmSize, VmSwap
    """
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {}

    result: dict[str, int] = {}
    try:
        text = status_path.read_text()
        for line in text.splitlines():
            for key in ("VmRSS", "VmPeak", "VmSize", "VmSwap"):
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        result[key] = int(parts[1])  # kB
    except (OSError, ValueError):
        pass
    return result


def check_memory_pressure(threshold_mb: float = 2048) -> bool:
    """Return *True* if resident memory exceeds *threshold_mb*.

    Falls back to ``resource.getrusage`` on non-Linux platforms.
    """
    usage = get_memory_usage()
    if "VmRSS" in usage:
        rss_mb = usage["VmRSS"] / 1024.0
    else:
        # Fallback: maxrss from getrusage (in KB on Linux, bytes on macOS)
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = maxrss / 1024.0

    if rss_mb > threshold_mb:
        logger.warning(
            "Memory pressure detected: %.0f MB RSS exceeds %.0f MB threshold",
            rss_mb,
            threshold_mb,
        )
        return True
    return False


def run_subprocess_with_limits(
    cmd: list[str],
    *,
    memory_mb: float | None = None,
    cpu_seconds: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with optional resource limits.

    Uses ``resource.setrlimit`` in a ``preexec_fn`` to enforce limits
    before the child exec's.  This is Linux-specific; on other platforms
    the limits are silently skipped.

    Args:
        cmd: Command and arguments.
        memory_mb: Virtual memory cap in MB (RLIMIT_AS).
        cpu_seconds: CPU time cap in seconds (RLIMIT_CPU).
        cwd: Working directory for the child.
        env: Environment variables (merged with os.environ).
        timeout: Wall-clock timeout in seconds.

    Returns:
        ``subprocess.CompletedProcess`` with captured stdout/stderr.

    Raises:
        subprocess.CalledProcessError: If the child exits non-zero.
        subprocess.TimeoutExpired: If *timeout* is exceeded.
    """

    def _set_limits() -> None:
        if memory_mb is not None:
            limit_bytes = int(memory_mb * 1024 * 1024)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, OSError):
                pass
        if cpu_seconds is not None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass

    merged_env = {**os.environ, **(env or {})}
    logger.debug("Running subprocess: %s (mem=%s MB, cpu=%s s)", cmd, memory_mb, cpu_seconds)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=merged_env,
        timeout=timeout,
        preexec_fn=_set_limits,
        check=True,
    )
