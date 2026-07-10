"""Tests for pipeline/comparison: metric extraction, deltas, and reports.

Pins MetricSet.from_workflow_result on synthetic WorkflowResult objects,
ComparisonTable row/text/delta output, the value formatter, stress field
differencing, and end-to-end comparison report generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.pipeline.comparison import (
    ComparisonTable,
    MetricSet,
    _fmt_value,
    compute_stress_field_difference,
    generate_comparison_report,
)
from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh() -> FEMesh:
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    return FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TRI3)


def _make_stress(sigma_yy) -> StressField:
    vals = np.zeros((len(sigma_yy), 6))
    vals[:, 1] = sigma_yy
    return StressField(values=vals)


def _make_result(name: str = "case", *, stress=None, displacement=None,
                 **wf_kwargs) -> WorkflowResult:
    mesh = _make_mesh()
    fea = None
    if stress is not None or displacement is not None:
        fea = FEAResults(mesh=mesh, stress=stress, displacement=displacement)
    return WorkflowResult(
        case=AnalysisCase(name=name), mesh=mesh, fea_results=fea, **wf_kwargs,
    )


def _make_study_results(errors=None):
    from feaweld.pipeline.study import StudyResults

    cases, results = {}, {}
    for name, scale in [("thin", 1.0), ("medium", 1.5), ("thick", 2.0)]:
        wf = _make_result(
            name,
            stress=_make_stress(np.array([50.0, 100.0, 150.0, 200.0]) * scale),
            fatigue_results={"hotspot_linear": {"stress_range": 100.0 * scale,
                                                "life": 2.0e6 / scale}},
        )
        cases[name] = wf.case
        results[name] = wf
    return StudyResults(
        study_name="plate_sweep", cases=cases, results=results,
        errors=errors or {}, elapsed_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# Tests: MetricSet.from_workflow_result
# ---------------------------------------------------------------------------

class TestMetricSetExtraction:
    def test_empty_result_all_none(self):
        m = MetricSet.from_workflow_result(WorkflowResult(case=AnalysisCase(name="e")))
        assert all(v is None for v in m.to_dict().values())

    def test_mesh_counts_only(self):
        m = MetricSet.from_workflow_result(_make_result())
        assert m.n_nodes == 4
        assert m.n_elements == 2
        assert m.max_von_mises is None
        assert m.max_displacement is None

    def test_stress_metrics(self):
        # Uniaxial σ_yy per node: von Mises, Tresca, and P1 all equal σ_yy.
        stress = _make_stress([50.0, 100.0, 150.0, 200.0])
        m = MetricSet.from_workflow_result(_make_result(stress=stress))
        assert m.max_von_mises == pytest.approx(200.0)
        assert m.mean_von_mises == pytest.approx(125.0)
        assert m.max_tresca == pytest.approx(200.0)
        assert m.max_principal_1 == pytest.approx(200.0)

    def test_displacement_metric(self):
        disp = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ])
        m = MetricSet.from_workflow_result(_make_result(displacement=disp))
        assert m.max_displacement == pytest.approx(5.0)

    def test_fixture_results(self, uniform_stress_results):
        wf = WorkflowResult(
            case=AnalysisCase(name="u"),
            mesh=uniform_stress_results.mesh,
            fea_results=uniform_stress_results,
        )
        m = MetricSet.from_workflow_result(wf)
        assert m.max_von_mises == pytest.approx(100.0)
        assert m.mean_von_mises == pytest.approx(100.0)
        assert m.max_displacement == pytest.approx(0.005)

    def test_safety_factor_with_allowable(self):
        stress = _make_stress([50.0, 100.0, 150.0, 200.0])
        m = MetricSet.from_workflow_result(
            _make_result(stress=stress), allowable_stress=400.0,
        )
        assert m.safety_factor == pytest.approx(2.0)

    def test_safety_factor_none_without_allowable(self):
        stress = _make_stress([50.0, 100.0, 150.0, 200.0])
        m = MetricSet.from_workflow_result(_make_result(stress=stress))
        assert m.safety_factor is None

    def test_safety_factor_none_for_zero_stress(self):
        # max_von_mises must be > 0 for a safety factor to be computed.
        stress = _make_stress([0.0, 0.0, 0.0, 0.0])
        m = MetricSet.from_workflow_result(
            _make_result(stress=stress), allowable_stress=400.0,
        )
        assert m.safety_factor is None

    def test_min_life_selected_across_methods(self):
        wf = _make_result(fatigue_results={
            "hotspot_linear": {"life": 5e6},
            "notch": {"life": 2e6},
            "nominal": {"life": 8e6},
        })
        m = MetricSet.from_workflow_result(wf)
        assert m.fatigue_life == pytest.approx(2e6)

    def test_fatigue_entries_without_life_ignored(self):
        wf = _make_result(fatigue_results={
            "nominal": {"stress_range": 100.0},
            "rainflow": [(50.0, 0.0, 1.0)],
            "hotspot_linear": {"life": 3e6},
        })
        m = MetricSet.from_workflow_result(wf)
        assert m.fatigue_life == pytest.approx(3e6)

    def test_fatigue_life_without_fea(self):
        wf = WorkflowResult(
            case=AnalysisCase(name="f"),
            fatigue_results={"nominal": {"life": 1e6}},
        )
        m = MetricSet.from_workflow_result(wf)
        assert m.fatigue_life == pytest.approx(1e6)
        assert m.max_von_mises is None

    def test_hotspot_stress_extracted_case_insensitive(self):
        wf = _make_result(postprocess_results={
            "HotSpot_quadratic": {"max_stress": 123.4},
        })
        m = MetricSet.from_workflow_result(wf)
        assert m.hotspot_stress == pytest.approx(123.4)

    def test_hotspot_ignores_non_dict_and_missing_key(self):
        wf = _make_result(postprocess_results={
            "hotspot_note": "failed",
            "hotspot_linear": {"stress_range": 90.0},
        })
        m = MetricSet.from_workflow_result(wf)
        assert m.hotspot_stress is None

    def test_to_dict_completeness(self):
        d = MetricSet().to_dict()
        assert set(d) == {
            "max_von_mises", "mean_von_mises", "max_displacement", "max_tresca",
            "fatigue_life", "safety_factor", "hotspot_stress", "max_principal_1",
            "n_nodes", "n_elements",
        }


# ---------------------------------------------------------------------------
# Tests: ComparisonTable
# ---------------------------------------------------------------------------

class TestComparisonTable:
    @staticmethod
    def _table() -> ComparisonTable:
        return ComparisonTable(
            case_names=["a", "b"],
            metrics={
                "a": MetricSet(max_von_mises=100.0, fatigue_life=2e6, n_nodes=4),
                "b": MetricSet(max_von_mises=130.0, fatigue_life=1e6, n_nodes=4),
            },
        )

    def test_to_rows_structure(self):
        rows = self._table().to_rows()
        assert [row["case"] for row in rows] == ["a", "b"]
        assert list(rows[0])[0] == "case"
        assert rows[0]["max_von_mises"] == 100.0
        assert rows[1]["fatigue_life"] == 1e6
        assert rows[0]["safety_factor"] is None

    def test_text_table_formatting(self):
        text = self._table().to_text_table()
        lines = text.split("\n")
        assert len(lines) == 4  # header + separator + 2 case rows
        assert "case" in lines[0]
        assert "max_von_mises" in lines[0]
        assert "-+-" in lines[1]
        assert "100.00" in lines[2]
        assert "2.00e+06" in lines[2]

    def test_text_table_dashes_for_none(self):
        table = ComparisonTable(case_names=["x"], metrics={"x": MetricSet()})
        text = table.to_text_table()
        data_line = text.split("\n")[2]
        assert data_line.split(" | ")[1:] == [
            "-".rjust(len(col)) for col in text.split("\n")[0].split(" | ")[1:]
        ]

    def test_text_table_empty(self):
        table = ComparisonTable(case_names=[], metrics={})
        assert table.to_text_table() == "(no data)"

    def test_delta_from_baseline_values(self):
        deltas = self._table().delta_from_baseline("a")
        assert len(deltas) == 1
        row = deltas[0]
        assert row["case"] == "b"
        assert row["max_von_mises_delta"] == pytest.approx(30.0)
        assert row["max_von_mises_pct"] == pytest.approx(30.0)
        assert row["fatigue_life_delta"] == pytest.approx(-1e6)
        assert row["fatigue_life_pct"] == pytest.approx(-50.0)

    def test_delta_excludes_baseline_row(self):
        deltas = self._table().delta_from_baseline("b")
        assert [row["case"] for row in deltas] == ["a"]

    def test_delta_none_metric_gives_none(self):
        deltas = self._table().delta_from_baseline("a")
        assert deltas[0]["safety_factor_delta"] is None
        assert deltas[0]["safety_factor_pct"] is None

    def test_delta_zero_baseline_pct_is_zero(self):
        table = ComparisonTable(
            case_names=["base", "other"],
            metrics={
                "base": MetricSet(max_von_mises=0.0),
                "other": MetricSet(max_von_mises=10.0),
            },
        )
        row = table.delta_from_baseline("base")[0]
        assert row["max_von_mises_delta"] == pytest.approx(10.0)
        assert row["max_von_mises_pct"] == 0.0

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self._table().delta_from_baseline("nope")


# ---------------------------------------------------------------------------
# Tests: _fmt_value
# ---------------------------------------------------------------------------

class TestFmtValue:
    def test_none_dash(self):
        assert _fmt_value(None) == "-"

    def test_int_passthrough(self):
        assert _fmt_value(42) == "42"

    def test_float_fixed(self):
        assert _fmt_value(123.456) == "123.46"
        assert _fmt_value(0.0) == "0.00"
        assert _fmt_value(0.01) == "0.01"

    def test_float_scientific_large(self):
        assert _fmt_value(1e5) == "1.00e+05"
        assert _fmt_value(2.5e6) == "2.50e+06"

    def test_float_scientific_small(self):
        assert _fmt_value(0.005) == "5.00e-03"
        assert _fmt_value(-0.005) == "-5.00e-03"

    def test_infinity(self):
        assert _fmt_value(float("inf")) == "inf"

    def test_string_passthrough(self):
        assert _fmt_value("case_a") == "case_a"


# ---------------------------------------------------------------------------
# Tests: compute_stress_field_difference
# ---------------------------------------------------------------------------

class TestStressFieldDifference:
    def test_matched_fields(self):
        a = _make_stress([100.0, 150.0, 200.0, 250.0])
        b = _make_stress([80.0, 100.0, 150.0, 200.0])
        diff = compute_stress_field_difference(a, b)
        assert isinstance(diff, StressField)
        np.testing.assert_allclose(diff.values[:, 1], [20.0, 50.0, 50.0, 50.0])
        np.testing.assert_allclose(diff.values[:, [0, 2, 3, 4, 5]], 0.0)

    def test_identical_fields_zero_difference(self):
        a = _make_stress([100.0, 150.0, 200.0, 250.0])
        diff = compute_stress_field_difference(a, a)
        np.testing.assert_allclose(diff.von_mises, 0.0)

    def test_mismatched_node_counts_raise(self):
        a = _make_stress([100.0, 150.0, 200.0, 250.0])
        b = _make_stress([100.0, 150.0])
        with pytest.raises(ValueError, match="Incompatible stress field shapes"):
            compute_stress_field_difference(a, b)


# ---------------------------------------------------------------------------
# Tests: generate_comparison_report
# ---------------------------------------------------------------------------

class TestGenerateComparisonReport:
    def test_writes_html_with_case_names(self, tmp_path):
        sr = _make_study_results()
        path = generate_comparison_report(sr, tmp_path)
        assert path == str(tmp_path / "plate_sweep_comparison.html")
        html = open(path, encoding="utf-8").read()
        assert "<!DOCTYPE html>" in html
        for name in ("thin", "medium", "thick"):
            assert name in html
        assert "plate_sweep" in html

    def test_delta_section_with_baseline(self, tmp_path):
        sr = _make_study_results()
        html = open(generate_comparison_report(sr, tmp_path, baseline="thin"),
                    encoding="utf-8").read()
        assert "Deltas vs. Baseline" in html

    def test_unknown_baseline_skips_delta_section(self, tmp_path):
        # Unlike ComparisonTable.delta_from_baseline, the report silently
        # drops the delta table for an unknown baseline.
        sr = _make_study_results()
        html = open(generate_comparison_report(sr, tmp_path, baseline="nope"),
                    encoding="utf-8").read()
        assert "Deltas vs. Baseline" not in html

    def test_failed_cases_listed(self, tmp_path):
        sr = _make_study_results(errors={"broken": "solver exploded"})
        html = open(generate_comparison_report(sr, tmp_path),
                    encoding="utf-8").read()
        assert "Failed Cases" in html
        assert "solver exploded" in html

    def test_creates_nested_output_dir(self, tmp_path):
        out = tmp_path / "reports" / "nested"
        sr = _make_study_results()
        path = generate_comparison_report(sr, out)
        assert out.is_dir()
        assert path.startswith(str(out))
