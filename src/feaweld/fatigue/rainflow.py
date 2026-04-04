"""Four-point rainflow cycle counting per ASTM E1049.

Implements the standard four-point algorithm to extract closed
hysteresis cycles from an arbitrary stress-time signal.
"""

from __future__ import annotations

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
