"""Helpers for loading bundled defect acceptance criteria via the DataRegistry."""

from __future__ import annotations

import json
from typing import Any

from feaweld.data.registry import DataRegistry

_STANDARD_KEYS: dict[str, str] = {
    "ISO 5817": "defect_acceptance/iso5817",
    "ISO5817": "defect_acceptance/iso5817",
    "ASME BPVC IX": "defect_acceptance/asme_bpvc_ix",
    "ASME_BPVC_IX": "defect_acceptance/asme_bpvc_ix",
    "AWS D1.1": "defect_acceptance/aws_d1_1_table61",
    "AWS_D1_1": "defect_acceptance/aws_d1_1_table61",
    "BS 7910": "defect_fat/bs7910_annex_f",
    "BS7910": "defect_fat/bs7910_annex_f",
}


def load_acceptance_criteria(
    standard: str, registry: DataRegistry | None = None
) -> dict[str, Any]:
    """Load a bundled defect acceptance / downgrade table as a dict.

    Args:
        standard: Standard name, e.g. ``"ISO 5817"`` or ``"BS 7910"``.
        registry: Optional DataRegistry instance (defaults to a fresh scan).

    Raises:
        KeyError: If the standard is unknown or the file is missing.
    """
    reg = registry or DataRegistry()
    key = _STANDARD_KEYS.get(standard) or _STANDARD_KEYS.get(standard.upper())
    if key is None:
        raise KeyError(
            f"Unknown defect standard: {standard!r}. "
            f"Known: {sorted(set(_STANDARD_KEYS.keys()))}"
        )
    path = reg.get_dataset_path(key)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
