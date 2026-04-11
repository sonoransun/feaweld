"""BS 7910 / IIW analytical FAT downgrade helpers for weld defects.

Each helper returns a :class:`KnockdownResult` containing the downgraded
FAT class, the multiplicative knockdown factor, and a short rationale
string. The rules here are intentionally simplified for the MVP; the
full BS 7910 Annex F and IIW recommendation tables are bundled under
``src/feaweld/data/defect_fat/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from feaweld.defects.types import (
    ClusterPorosity,
    LackOfFusionDefect,
    PoreDefect,
    RootGapDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
)


@dataclass
class KnockdownResult:
    """Result of applying a defect-based FAT downgrade."""

    downgraded_fat: float
    knockdown_factor: float
    rationale: str


def porosity_fat_downgrade(
    pore_diameter: float, plate_thickness: float, base_fat: float
) -> KnockdownResult:
    """BS 7910 Annex F pore FAT downgrade (simplified MVP rule).

    - ``d / t <= 0.02`` => no downgrade.
    - Otherwise factor = ``max(0.5, 1 - 5 * (d/t - 0.02))`` (floor at 0.5).
    """
    if plate_thickness <= 0.0:
        raise ValueError("plate_thickness must be positive")
    ratio = pore_diameter / plate_thickness
    if ratio <= 0.02:
        return KnockdownResult(base_fat, 1.0, "pore below BS 7910 threshold")
    factor = max(0.5, 1.0 - 5.0 * (ratio - 0.02))
    return KnockdownResult(
        base_fat * factor,
        factor,
        f"BS 7910 pore d/t={ratio:.3f}",
    )


def lof_fat_downgrade(lof_depth: float, base_fat: float) -> KnockdownResult:
    """IIW rule: lack-of-fusion is a planar flaw -> drop to the FAT 36 floor."""
    floor = 36.0
    if base_fat <= floor:
        return KnockdownResult(base_fat, 1.0, "already at planar-flaw floor")
    factor = floor / base_fat
    return KnockdownResult(
        floor,
        factor,
        f"IIW LoF depth={lof_depth:.2f}mm -> FAT{int(floor)} floor",
    )


def undercut_fat_downgrade(
    undercut_depth: float, plate_thickness: float, base_fat: float
) -> KnockdownResult:
    """IIW undercut rule: linear reduction, floored at 0.7."""
    if plate_thickness <= 0.0:
        raise ValueError("plate_thickness must be positive")
    ratio = undercut_depth / plate_thickness
    factor = max(0.7, 1.0 - 1.5 * ratio)
    return KnockdownResult(
        base_fat * factor,
        factor,
        f"IIW undercut d/t={ratio:.3f}",
    )


def slag_fat_downgrade(
    semi_axes: tuple[float, float, float], base_fat: float
) -> KnockdownResult:
    """BS 7910: inclusions behave like pores with an equivalent diameter."""
    a, b, c = semi_axes
    d_eff = 2.0 * (a * b * c) ** (1.0 / 3.0)
    return porosity_fat_downgrade(d_eff, 20.0, base_fat)


def defect_fat_downgrade(
    defect: Any, base_fat: float, plate_thickness: float = 20.0
) -> KnockdownResult:
    """Dispatch a defect to the appropriate downgrade helper.

    Uses ``defect.defect_type`` and known concrete classes to pick a rule.
    Unknown defect types return a pass-through result (factor 1.0).
    """
    if isinstance(defect, PoreDefect):
        return porosity_fat_downgrade(defect.diameter, plate_thickness, base_fat)
    if isinstance(defect, ClusterPorosity):
        # Use mean pore size against plate thickness.
        return porosity_fat_downgrade(defect.size_mean, plate_thickness, base_fat)
    if isinstance(defect, SlagInclusion):
        return slag_fat_downgrade(defect.semi_axes, base_fat)
    if isinstance(defect, UndercutDefect):
        return undercut_fat_downgrade(defect.depth, plate_thickness, base_fat)
    if isinstance(defect, LackOfFusionDefect):
        return lof_fat_downgrade(max(defect.extent_u, defect.extent_v), base_fat)
    if isinstance(defect, SurfaceCrack):
        return lof_fat_downgrade(defect.depth, base_fat)
    if isinstance(defect, RootGapDefect):
        return porosity_fat_downgrade(defect.gap_width, defect.plate_thickness, base_fat)
    return KnockdownResult(
        base_fat,
        1.0,
        f"no downgrade rule for defect_type={getattr(defect, 'defect_type', type(defect).__name__)}",
    )
