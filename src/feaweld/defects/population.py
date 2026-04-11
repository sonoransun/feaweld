"""Stochastic sampling of weld defect populations from quality-level limits.

The :func:`sample_iso5817_population` function draws Poisson-distributed
counts of pores, undercuts, and slag inclusions for a given ISO 5817
quality level (B/C/D) and weld geometry.  Each defect's geometry is
sampled uniformly below the acceptance limit so that the *entire*
population passes :func:`validate_population`.

Sampling is pure numpy and does **not** require Gmsh.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from feaweld.core.types import Point3D
from feaweld.defects.loader import load_acceptance_criteria
from feaweld.defects.types import (
    Defect,
    PoreDefect,
    SlagInclusion,
    UndercutDefect,
)


_Level = Literal["B", "C", "D"]


def sample_iso5817_population(
    level: _Level,
    weld_length: float,
    weld_width: float,
    plate_thickness: float = 10.0,
    seed: int = 0,
    pore_rate_per_mm: float = 0.05,
    undercut_rate_per_mm: float = 0.02,
    slag_rate_per_mm: float = 0.01,
) -> list[Defect]:
    """Draw a realistic defect population consistent with an ISO 5817 level.

    Parameters
    ----------
    level:
        ISO 5817 quality level (``"B"``, ``"C"``, or ``"D"``).
    weld_length, weld_width:
        Weld dimensions in mm.  Defect counts scale with ``weld_length``.
    plate_thickness:
        Plate thickness in mm.  Sets the maximum admissible pore diameter
        via the ``pore_max_d_over_t`` rule.
    seed:
        Seed for the internal :class:`numpy.random.Generator`.  Same
        seed + inputs yields bit-identical populations.
    pore_rate_per_mm, undercut_rate_per_mm, slag_rate_per_mm:
        Poisson intensities (defects per mm of weld).

    Returns
    -------
    list[Defect]
        Mixed list of :class:`PoreDefect`, :class:`UndercutDefect`, and
        :class:`SlagInclusion` objects.  Every entry satisfies the
        corresponding ISO 5817 acceptance limit for ``level``.
    """
    criteria = load_acceptance_criteria("ISO 5817")
    limits = criteria["levels"][level]

    rng = np.random.default_rng(seed)
    defects: list[Defect] = []

    # ------------------------------------------------------------------
    # Pores
    # ------------------------------------------------------------------
    pore_max_d = min(
        limits["pore_max_d_over_t"] * plate_thickness,
        limits["pore_absolute_max_mm"],
    )
    n_pores = int(rng.poisson(pore_rate_per_mm * weld_length))
    for _ in range(n_pores):
        diameter = 0.0
        for _attempt in range(10):
            d = float(rng.uniform(0.1, max(pore_max_d, 0.1 + 1e-9)))
            if d <= pore_max_d:
                diameter = d
                break
        else:  # pragma: no cover - unreachable with uniform-below-limit
            raise ValueError(f"Could not draw valid pore for level {level}")
        x = float(rng.uniform(0.0, weld_length))
        y = float(rng.uniform(-weld_width / 2.0, weld_width / 2.0))
        z = float(rng.uniform(-plate_thickness / 2.0, plate_thickness / 2.0))
        defects.append(PoreDefect(center=Point3D(x, y, z), diameter=diameter))

    # ------------------------------------------------------------------
    # Undercut runs
    # ------------------------------------------------------------------
    uc_max = float(limits["undercut_max_mm"])
    n_undercuts = int(rng.poisson(undercut_rate_per_mm * weld_length))
    for _ in range(n_undercuts):
        depth = float(rng.uniform(0.0, uc_max))
        x0 = float(rng.uniform(0.0, max(weld_length * 0.9, 1e-9)))
        length = float(rng.uniform(1.0, max(weld_length * 0.1, 1.0 + 1e-9)))
        defects.append(
            UndercutDefect(
                start=Point3D(x0, weld_width / 2.0, 0.0),
                end=Point3D(x0 + length, weld_width / 2.0, 0.0),
                depth=depth,
                profile="V",
            )
        )

    # ------------------------------------------------------------------
    # Slag inclusions
    # ------------------------------------------------------------------
    slag_len_max = float(limits["slag_length_max_mm"])
    n_slag = int(rng.poisson(slag_rate_per_mm * weld_length))
    for _ in range(n_slag):
        semi_len = float(rng.uniform(0.1, max(slag_len_max / 2.0, 0.1 + 1e-9)))
        semi_width = float(rng.uniform(0.1, max(semi_len * 0.5, 0.1 + 1e-9)))
        semi_depth = float(rng.uniform(0.1, max(semi_width, 0.1 + 1e-9)))
        x = float(rng.uniform(0.0, weld_length))
        y = float(rng.uniform(-weld_width / 2.0, weld_width / 2.0))
        defects.append(
            SlagInclusion(
                center=Point3D(x, y, 0.0),
                semi_axes=(semi_len, semi_width, semi_depth),
            )
        )

    return defects


def validate_population(
    defects: list[Defect],
    level: str,
    plate_thickness: float = 10.0,
) -> list[str]:
    """Return a list of violation strings (empty if the population is valid).

    Each string is a short human-readable reason such as
    ``"pore d=3.2 > limit"``.  Used both by the sampler tests (expecting
    an empty list) and by callers who want to audit an externally
    supplied defect list.
    """
    criteria = load_acceptance_criteria("ISO 5817")
    limits = criteria["levels"][level]

    violations: list[str] = []
    for d in defects:
        dtype = getattr(d, "defect_type", None)
        if dtype == "pore":
            diameter = d.diameter  # type: ignore[attr-defined]
            if diameter > limits["pore_absolute_max_mm"]:
                violations.append(
                    f"pore d={diameter:.3f} > {limits['pore_absolute_max_mm']}"
                )
            if diameter / plate_thickness > limits["pore_max_d_over_t"]:
                violations.append(
                    f"pore d/t={diameter / plate_thickness:.4f} > "
                    f"{limits['pore_max_d_over_t']}"
                )
        elif dtype == "undercut":
            depth = d.depth  # type: ignore[attr-defined]
            if depth > limits["undercut_max_mm"]:
                violations.append(
                    f"undercut d={depth:.3f} > {limits['undercut_max_mm']}"
                )
        elif dtype == "slag":
            crit = d.critical_dimension()
            if crit > limits["slag_length_max_mm"]:
                violations.append(
                    f"slag length={crit:.3f} > {limits['slag_length_max_mm']}"
                )
    return violations


__all__ = [
    "sample_iso5817_population",
    "validate_population",
]
