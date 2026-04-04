"""IIW weld detail category lookup.

Provides typed access to the 80 IIW weld detail categories stored in
``iiw_weld_details.json``, including search by joint type, weld type,
and FAT class recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from feaweld.data.cache import get_cache


@dataclass
class WeldDetail:
    """One IIW weld detail category."""

    detail_number: int
    description: str
    fat_class: int
    joint_type: str
    loading: str
    weld_type: str
    requirements: str


def _to_weld_detail(entry: dict) -> WeldDetail:
    """Convert a raw dict from the JSON data to a WeldDetail."""
    return WeldDetail(
        detail_number=entry["detail_number"],
        description=entry["description"],
        fat_class=entry["fat_class"],
        joint_type=entry["joint_type"],
        loading=entry["loading"],
        weld_type=entry["weld_type"],
        requirements=entry["requirements"],
    )


def get_weld_detail(detail_number: int) -> WeldDetail:
    """Look up a single IIW weld detail by its detail number.

    Args:
        detail_number: The IIW detail number (e.g. 200, 413).

    Returns:
        WeldDetail dataclass.

    Raises:
        KeyError: If the detail number is not found.
    """
    data = get_cache().get("sn_curves/iiw_weld_details")
    for entry in data:
        if entry["detail_number"] == detail_number:
            return _to_weld_detail(entry)
    available = sorted(e["detail_number"] for e in data)
    raise KeyError(
        f"IIW weld detail not found: {detail_number}. "
        f"Available numbers range from {available[0]} to {available[-1]}."
    )


def find_weld_details(
    *,
    joint_type: str | None = None,
    weld_type: str | None = None,
    loading: str | None = None,
    min_fat: int | None = None,
    max_fat: int | None = None,
) -> list[WeldDetail]:
    """Search weld details by filter criteria.

    All filters are optional and combined with AND logic.

    Args:
        joint_type: Filter by joint type (e.g. ``"butt"``, ``"cruciform"``).
        weld_type: Filter by weld type (e.g. ``"fillet"``, ``"full_pen"``).
        loading: Filter by loading mode (e.g. ``"direct_stress"``, ``"shear"``).
        min_fat: Minimum FAT class (inclusive).
        max_fat: Maximum FAT class (inclusive).

    Returns:
        List of matching WeldDetail entries, ordered by detail number.
    """
    data = get_cache().get("sn_curves/iiw_weld_details")
    results: list[WeldDetail] = []

    for entry in data:
        if joint_type is not None and entry["joint_type"] != joint_type:
            continue
        if weld_type is not None and entry["weld_type"] != weld_type:
            continue
        if loading is not None and entry["loading"] != loading:
            continue
        if min_fat is not None and entry["fat_class"] < min_fat:
            continue
        if max_fat is not None and entry["fat_class"] > max_fat:
            continue
        results.append(_to_weld_detail(entry))

    results.sort(key=lambda d: d.detail_number)
    return results


def recommend_fat_class(
    joint_type: str,
    weld_type: str,
    loading: str = "direct_stress",
) -> WeldDetail | None:
    """Recommend the most appropriate FAT class for a given configuration.

    Returns the detail with the highest FAT class matching the given
    joint type, weld type, and loading mode.  This is a conservative
    starting point — the user should verify requirements are met.

    Args:
        joint_type: Joint type (e.g. ``"butt"``, ``"cruciform"``).
        weld_type: Weld type (e.g. ``"fillet"``, ``"full_pen"``).
        loading: Loading mode (default ``"direct_stress"``).

    Returns:
        Best-matching WeldDetail, or ``None`` if no match is found.
    """
    matches = find_weld_details(
        joint_type=joint_type,
        weld_type=weld_type,
        loading=loading,
    )
    if not matches:
        return None
    return max(matches, key=lambda d: d.fat_class)


def list_weld_details() -> list[WeldDetail]:
    """Return all IIW weld detail entries."""
    data = get_cache().get("sn_curves/iiw_weld_details")
    return [_to_weld_detail(e) for e in data]
