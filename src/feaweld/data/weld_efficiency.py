"""Weld joint efficiency factor lookup.

Provides efficiency factors from major pressure vessel and structural
codes (ASME VIII Div 1, AWS D1.1, EN 13445) based on joint type and
NDE examination level.
"""

from __future__ import annotations

from dataclasses import dataclass

from feaweld.data.cache import get_cache


@dataclass
class WeldEfficiency:
    """Weld joint efficiency for one code/joint/examination combination."""

    standard: str
    joint_type: str
    examination: str
    efficiency: float
    notes: str = ""


def _to_efficiency(entry: dict) -> WeldEfficiency:
    """Convert a raw dict from the JSON data to a WeldEfficiency."""
    return WeldEfficiency(
        standard=entry["standard"],
        joint_type=entry["joint_type"],
        examination=entry["examination"],
        efficiency=float(entry["efficiency"]),
        notes=entry.get("notes", ""),
    )


def get_weld_efficiency(
    standard: str,
    joint_type: str,
    examination: str,
) -> WeldEfficiency:
    """Look up a weld joint efficiency factor.

    Args:
        standard: Code standard (e.g. ``"ASME_VIII_Div1"``, ``"AWS_D1.1"``).
        joint_type: Joint type (e.g. ``"Type_1"``, ``"CJP"``, ``"Butt_full_pen"``).
        examination: NDE examination level (e.g. ``"Full_RT"``, ``"Spot_RT"``).

    Returns:
        WeldEfficiency dataclass.

    Raises:
        KeyError: If the combination is not found.
    """
    data = get_cache().get("weld_efficiency/tables")
    for entry in data:
        if (
            entry["standard"] == standard
            and entry["joint_type"] == joint_type
            and entry["examination"] == examination
        ):
            return _to_efficiency(entry)

    # Build a helpful error message
    available = [
        f"{e['standard']}/{e['joint_type']}/{e['examination']}" for e in data
    ]
    raise KeyError(
        f"Weld efficiency not found: {standard}/{joint_type}/{examination}. "
        f"Available combinations ({len(available)}): {available[:10]}..."
    )


def list_efficiencies(
    *,
    standard: str | None = None,
    joint_type: str | None = None,
) -> list[WeldEfficiency]:
    """List weld efficiency entries with optional filtering.

    Args:
        standard: Filter by code standard.
        joint_type: Filter by joint type.

    Returns:
        List of matching WeldEfficiency entries.
    """
    data = get_cache().get("weld_efficiency/tables")
    results: list[WeldEfficiency] = []

    for entry in data:
        if standard is not None and entry["standard"] != standard:
            continue
        if joint_type is not None and entry["joint_type"] != joint_type:
            continue
        results.append(_to_efficiency(entry))

    return results
