"""AWS filler metal lookup and base-metal matching.

Provides typed access to ~80 filler metal classifications from AWS A5.x
specifications, covering SMAW, GMAW, FCAW, and SAW processes for
carbon steel, low-alloy steel, stainless steel, and nickel alloy
consumables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from feaweld.data.cache import get_cache


@dataclass
class FillerMetal:
    """Properties of one filler metal classification."""

    classification: str
    aws_spec: str
    process: str
    tensile_mpa: float
    yield_mpa: float
    elongation_pct: float
    charpy_temp_c: float
    charpy_j: float
    base_metal_match: list[str] = field(default_factory=list)


def _to_filler(entry: dict) -> FillerMetal:
    """Convert a raw dict from the JSON data to a FillerMetal."""
    return FillerMetal(
        classification=entry["classification"],
        aws_spec=entry["aws_spec"],
        process=entry["process"],
        tensile_mpa=float(entry["tensile_mpa"]),
        yield_mpa=float(entry["yield_mpa"]),
        elongation_pct=float(entry["elongation_pct"]),
        charpy_temp_c=float(entry["charpy_temp_c"]),
        charpy_j=float(entry["charpy_j"]),
        base_metal_match=entry.get("base_metal_match", []),
    )


def get_filler_metal(classification: str) -> FillerMetal:
    """Look up a filler metal by its AWS classification.

    Args:
        classification: AWS classification (e.g. ``"E7018"``, ``"ER70S-6"``).

    Returns:
        FillerMetal dataclass.

    Raises:
        KeyError: If the classification is not found.
    """
    data = get_cache().get("filler_metals/aws_a5")
    for entry in data:
        if entry["classification"] == classification:
            return _to_filler(entry)
    available = [e["classification"] for e in data]
    raise KeyError(
        f"Filler metal not found: {classification!r}. "
        f"Available ({len(available)}): {available[:15]}..."
    )


def list_filler_metals(
    *,
    process: str | None = None,
    aws_spec: str | None = None,
    min_tensile_mpa: float | None = None,
) -> list[FillerMetal]:
    """List filler metals with optional filtering.

    Args:
        process: Filter by welding process (e.g. ``"SMAW"``, ``"GMAW"``).
        aws_spec: Filter by AWS specification (e.g. ``"A5.1"``, ``"A5.18"``).
        min_tensile_mpa: Minimum tensile strength in MPa.

    Returns:
        List of matching FillerMetal entries.
    """
    data = get_cache().get("filler_metals/aws_a5")
    results: list[FillerMetal] = []

    for entry in data:
        if process is not None and entry["process"] != process:
            continue
        if aws_spec is not None and entry["aws_spec"] != aws_spec:
            continue
        if min_tensile_mpa is not None and entry["tensile_mpa"] < min_tensile_mpa:
            continue
        results.append(_to_filler(entry))

    return results


def filler_for_base_metal(
    base_metal: str,
    *,
    process: str | None = None,
) -> list[FillerMetal]:
    """Find filler metals compatible with a given base metal.

    Args:
        base_metal: Base metal identifier (e.g. ``"A36"``, ``"316SS"``).
        process: Optionally restrict to a welding process.

    Returns:
        List of compatible FillerMetal entries, sorted by tensile strength.
    """
    data = get_cache().get("filler_metals/aws_a5")
    results: list[FillerMetal] = []

    for entry in data:
        if base_metal not in entry.get("base_metal_match", []):
            continue
        if process is not None and entry["process"] != process:
            continue
        results.append(_to_filler(entry))

    results.sort(key=lambda fm: fm.tensile_mpa)
    return results
