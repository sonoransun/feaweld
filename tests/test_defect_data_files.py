"""Tests for bundled defect acceptance / FAT downgrade data files."""

from __future__ import annotations

import json
from pathlib import Path

from feaweld.data.registry import DataRegistry
from feaweld.defects.loader import load_acceptance_criteria


def _data_path(*parts: str) -> Path:
    import feaweld.data as data_pkg

    return Path(data_pkg.__file__).parent.joinpath(*parts)


def test_iso5817_has_three_levels():
    path = _data_path("defect_acceptance", "iso5817.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["standard"] == "ISO 5817"
    levels = data["levels"]
    for lvl in ("B", "C", "D"):
        assert lvl in levels
        fields = levels[lvl]
        for f in (
            "pore_max_d_over_t",
            "pore_absolute_max_mm",
            "undercut_max_mm",
            "cluster_area_max_mm2",
            "slag_length_max_mm",
            "lof_max_length_mm",
        ):
            assert f in fields


def test_bs7910_tables_monotone():
    path = _data_path("defect_fat", "bs7910_annex_f.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    pore = data["pore_downgrade"]
    ratios = [row["d_over_t"] for row in pore]
    factors = [row["factor"] for row in pore]
    assert ratios == sorted(ratios)
    assert all(factors[i] >= factors[i + 1] for i in range(len(factors) - 1))
    slag = data["slag_downgrade"]
    lengths = [row["length_mm"] for row in slag]
    sfactors = [row["factor"] for row in slag]
    assert lengths == sorted(lengths)
    assert all(sfactors[i] >= sfactors[i + 1] for i in range(len(sfactors) - 1))


def test_loader_via_registry():
    registry = DataRegistry()
    data = load_acceptance_criteria("ISO 5817", registry=registry)
    assert isinstance(data, dict)
    assert "levels" in data
    assert set(data["levels"].keys()) == {"B", "C", "D"}
