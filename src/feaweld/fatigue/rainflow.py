"""Four-point rainflow cycle counting per ASTM E1049.

Implements the standard four-point algorithm to extract closed
hysteresis cycles from an arbitrary stress-time signal.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _extract_peaks_valleys(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reduce a signal to its peaks and valleys (turning points).

    Consecutive equal values are collapsed.  The first and last points
    are always retained.
    """
    if len(signal) < 3:
        return np.array(signal, dtype=np.float64)

    pts: list[float] = [float(signal[0])]
    for i in range(1, len(signal) - 1):
        prev, cur, nxt = float(signal[i - 1]), float(signal[i]), float(signal[i + 1])
        # Keep turning points (local max or local min)
        if (cur - prev) * (nxt - cur) < 0:
            pts.append(cur)
        # Also keep if flat then changes direction
        elif cur != prev and cur == nxt:
            continue  # skip intermediate flat points
    pts.append(float(signal[-1]))

    # Remove consecutive duplicates
    filtered: list[float] = [pts[0]]
    for v in pts[1:]:
        if v != filtered[-1]:
            filtered.append(v)

    return np.array(filtered, dtype=np.float64)


def rainflow_count(
    signal: NDArray[np.float64],
) -> list[tuple[float, float, float]]:
    """Four-point rainflow cycle counting (ASTM E1049).

    Parameters
    ----------
    signal : NDArray
        1-D array of stress (or strain) values vs. time.

    Returns
    -------
    list[tuple[float, float, float]]
        Each tuple is ``(stress_range, mean_stress, count)`` where count
        is 0.5 for half-cycles and 1.0 for full cycles.
    """
    # Step 1: reduce to peaks and valleys
    pts = _extract_peaks_valleys(signal)
    if len(pts) < 2:
        return []

    # Working copy as a Python list for efficient pop / insert
    history: list[float] = pts.tolist()
    cycles: list[tuple[float, float, float]] = []

    # Step 2: four-point rainflow extraction
    i = 0
    while len(history) >= 4:
        # Consider points S1, S2, S3, S4
        S1 = history[i]
        S2 = history[i + 1]
        S3 = history[i + 2]
        S4 = history[i + 3]

        r_inner = abs(S3 - S2)
        r_left = abs(S2 - S1)
        r_right = abs(S4 - S3)

        if r_inner <= r_left and r_inner <= r_right:
            # Extract cycle (S2, S3)
            stress_range = r_inner
            mean_stress = (S2 + S3) / 2.0
            cycles.append((stress_range, mean_stress, 1.0))
            # Remove S2 and S3 from history
            del history[i + 1]
            del history[i + 1]  # was i+2, now i+1 after first deletion
            # Reset scan
            i = max(0, i - 2)
        else:
            i += 1
            if i + 3 >= len(history):
                break

    # Step 3: handle residue -- count remaining ranges as half-cycles
    for j in range(len(history) - 1):
        stress_range = abs(history[j + 1] - history[j])
        mean_stress = (history[j + 1] + history[j]) / 2.0
        if stress_range > 0:
            cycles.append((stress_range, mean_stress, 0.5))

    return cycles


# ---------------------------------------------------------------------------
# Multi-axial rainflow (Track G)
# ---------------------------------------------------------------------------

def _signed_max_principal(stress_voigt: NDArray[np.float64]) -> float:
    """Return the signed maximum principal stress for one Voigt tensor.

    Voigt convention here matches :class:`feaweld.core.types.StressField`:
    ``[σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]``. The returned scalar is the
    principal stress with the largest absolute value, carrying its own
    sign (tension positive).
    """
    s = stress_voigt
    tensor = np.array(
        [
            [s[0], s[3], s[5]],
            [s[3], s[1], s[4]],
            [s[5], s[4], s[2]],
        ],
        dtype=np.float64,
    )
    eigvals = np.linalg.eigvalsh(tensor)
    # Pick the principal stress with largest magnitude (signed).
    idx = int(np.argmax(np.abs(eigvals)))
    return float(eigvals[idx])


def _projected_normal_stress(
    stress_voigt: NDArray[np.float64], normal: NDArray[np.float64]
) -> float:
    """Cauchy projection ``σ_n = n · σ · n`` for a single Voigt tensor."""
    s = stress_voigt
    tensor = np.array(
        [
            [s[0], s[3], s[5]],
            [s[3], s[1], s[4]],
            [s[5], s[4], s[2]],
        ],
        dtype=np.float64,
    )
    return float(normal @ tensor @ normal)


def rainflow_multiaxial(
    stress_history: NDArray[np.float64],
    method: Literal["projection", "principal"] = "principal",
    plane_normal: NDArray[np.float64] | None = None,
) -> list[tuple[float, float, NDArray[np.float64]]]:
    """Multi-axial rainflow cycle counting via scalar projection.

    Parameters
    ----------
    stress_history:
        ``(n_t, 6)`` Voigt stress history
        ``[σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]`` per timestep.
    method:
        ``"principal"`` uses the signed max-magnitude principal stress as
        the scalar counting variable. ``"projection"`` projects each
        tensor onto ``plane_normal`` using the Cauchy relation
        ``σ_n = n · σ · n``.
    plane_normal:
        Unit vector normal to the critical plane; required when
        ``method="projection"``.

    Returns
    -------
    list of tuples
        Each tuple is ``(range_scalar, mean_scalar, mean_tensor_voigt)``
        where ``mean_tensor_voigt`` is the component-wise mean of the two
        Voigt tensors that bound the extracted cycle in the scalar
        history.

    Notes
    -----
    Scalar cycle counting is delegated to :func:`rainflow_count` so the
    ASTM E1049 four-point algorithm is not reimplemented here.
    """
    stress_history = np.asarray(stress_history, dtype=np.float64)
    if stress_history.ndim != 2 or stress_history.shape[1] != 6:
        raise ValueError(
            f"stress_history must have shape (n_t, 6), got {stress_history.shape}"
        )

    n_t = stress_history.shape[0]

    if method == "principal":
        scalars = np.array(
            [_signed_max_principal(stress_history[i]) for i in range(n_t)],
            dtype=np.float64,
        )
    elif method == "projection":
        if plane_normal is None:
            raise ValueError(
                "plane_normal is required when method='projection'"
            )
        n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-15:
            raise ValueError("plane_normal must be non-zero")
        n = n / norm
        scalars = np.array(
            [_projected_normal_stress(stress_history[i], n) for i in range(n_t)],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"method must be 'principal' or 'projection', got {method!r}"
        )

    scalar_cycles = rainflow_count(scalars)

    # Map each scalar cycle back to the nearest bounding timesteps in the
    # original multi-axial history so the caller can recover the mean
    # tensor. For the MVP we match on the two history values that
    # produced the cycle's peak and valley via nearest-value search.
    results: list[tuple[float, float, NDArray[np.float64]]] = []
    for stress_range, mean_stress, _count in scalar_cycles:
        peak = mean_stress + 0.5 * stress_range
        valley = mean_stress - 0.5 * stress_range
        i_peak = int(np.argmin(np.abs(scalars - peak)))
        i_valley = int(np.argmin(np.abs(scalars - valley)))
        mean_tensor = 0.5 * (
            stress_history[i_peak] + stress_history[i_valley]
        )
        results.append((stress_range, mean_stress, mean_tensor))

    return results
