"""CCT diagram lookup for steel grades.

Loads continuous cooling transformation data from bundled JSON and
returns :class:`~feaweld.multiscale.meso.CCTDiagram` instances that
can predict phase compositions at arbitrary cooling rates.
"""

from __future__ import annotations

import numpy as np

from feaweld.data.cache import get_cache
from feaweld.multiscale.meso import CCTDiagram


def get_cct_diagram(grade: str) -> CCTDiagram:
    """Get a CCT diagram for a steel grade.

    Args:
        grade: Steel grade identifier (e.g. ``"A36"``, ``"4140"``, ``"X80"``).

    Returns:
        CCTDiagram populated with phase-fraction curves.

    Raises:
        KeyError: If the grade is not found.
    """
    all_grades = get_cache().get("cct/steel_grades")
    if grade not in all_grades:
        raise KeyError(
            f"CCT grade not found: {grade!r}. Available: {sorted(all_grades.keys())}"
        )

    entry = all_grades[grade]
    rates = np.array(entry["cooling_rates"], dtype=np.float64)

    # Build phase-fraction arrays.  Stainless / duplex grades may store
    # "austenite" instead of (or in addition to) "ferrite"; the CCTDiagram
    # ferrite_fraction field is reused for austenite when Ac1 == 0.
    ferrite = np.array(entry.get("ferrite", [0.0] * len(rates)), dtype=np.float64)
    pearlite = np.array(entry.get("pearlite", [0.0] * len(rates)), dtype=np.float64)
    bainite = np.array(entry.get("bainite", [0.0] * len(rates)), dtype=np.float64)
    martensite = np.array(entry.get("martensite", [0.0] * len(rates)), dtype=np.float64)

    return CCTDiagram(
        cooling_rates=rates,
        Ac1=float(entry["Ac1"]),
        Ac3=float(entry["Ac3"]),
        Ms=float(entry["Ms"]),
        ferrite_fraction=ferrite,
        pearlite_fraction=pearlite,
        bainite_fraction=bainite,
        martensite_fraction=martensite,
    )


def list_cct_grades() -> list[str]:
    """Return all available steel grade names for CCT lookup."""
    all_grades = get_cache().get("cct/steel_grades")
    return sorted(all_grades.keys())


def find_closest_cct(carbon_equivalent: float) -> str:
    """Find the steel grade whose carbon equivalent is closest to the given value.

    This is a convenience function for when the user knows the CE but
    not the specific grade designation.

    Args:
        carbon_equivalent: Target carbon equivalent (IIW formula).

    Returns:
        The grade name with the closest carbon equivalent.
    """
    all_grades = get_cache().get("cct/steel_grades")

    best_grade: str | None = None
    best_diff = float("inf")

    for grade_name, entry in all_grades.items():
        ce = entry.get("carbon_equivalent", 0.0)
        diff = abs(ce - carbon_equivalent)
        if diff < best_diff:
            best_diff = diff
            best_grade = grade_name

    if best_grade is None:
        raise ValueError("No CCT grades available in dataset")
    return best_grade
