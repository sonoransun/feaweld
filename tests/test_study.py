"""Tests for parametric study management, comparison, and visualization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from feaweld.core.types import (
    ElementType, FEAResults, FEMesh, StressField, LoadCase,
)
from feaweld.pipeline.workflow import (
    AnalysisCase, WorkflowResult, MaterialConfig, GeometryConfig,
    LoadConfig, MeshConfig, PostProcessConfig,
)
from feaweld.pipeline.study import (
    Study, StudyConfig, StudyResults, ParameterSweep,
    _set_nested_attr, _get_nested_attr, load_study, save_study,
)
from feaweld.pipeline.comparison import (
    MetricSet, ComparisonTable, compute_stress_field_difference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_case() -> AnalysisCase:
    return AnalysisCase(
        name="base",
        load=LoadConfig(axial_force=10000.0),
        mesh=MeshConfig(global_size=2.0),
        material=MaterialConfig(base_metal="A36"),
    )


def _make_mock_result(case: AnalysisCase, max_stress: float = 100.0) -> WorkflowResult:
    """Create a synthetic WorkflowResult without running FEA."""
    nodes = np.array([[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elems, element_type=ElementType.TRI3)
    stress_vals = np.zeros((3, 6))
    stress_vals[:, 0] = [max_stress * 0.5, max_stress, max_stress * 0.8]
    return WorkflowResult(
        case=case,
        mesh=mesh,
        fea_results=FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
            displacement=np.array([[0, 0, 0], [0.01, 0, 0], [0.005, 0.02, 0]], dtype=float),
        ),
        fatigue_results={"test": {"stress_range": max_stress, "life": 1e7 / max_stress}},
    )


# ---------------------------------------------------------------------------
# Tests: Nested attribute helpers
# ---------------------------------------------------------------------------

class TestNestedAttr:
    def test_set_top_level(self):
        case = _make_base_case()
        updated = _set_nested_attr(case, "name", "changed")
        assert updated.name == "changed"
        assert case.name == "base"  # original unchanged

    def test_set_nested_load(self):
        case = _make_base_case()
        updated = _set_nested_attr(case, "load.axial_force", 99999.0)
        assert updated.load.axial_force == 99999.0
        assert case.load.axial_force == 10000.0

    def test_set_nested_mesh(self):
        case = _make_base_case()
        updated = _set_nested_attr(case, "mesh.global_size", 0.5)
        assert updated.mesh.global_size == 0.5

    def test_set_nested_material(self):
        case = _make_base_case()
        updated = _set_nested_attr(case, "material.base_metal", "304SS")
        assert updated.material.base_metal == "304SS"

    def test_get_nested(self):
        case = _make_base_case()
        assert _get_nested_attr(case, "load.axial_force") == 10000.0
        assert _get_nested_attr(case, "name") == "base"
        assert _get_nested_attr(case, "mesh.global_size") == 2.0


# ---------------------------------------------------------------------------
# Tests: Study case generation
# ---------------------------------------------------------------------------

class TestStudyCaseGeneration:
    def test_vary_single_param(self):
        s = Study("test", _make_base_case())
        s.vary("load.axial_force", [10000, 50000, 100000])
        cases = s._generate_cases("grid")
        assert len(cases) == 3

    def test_vary_grid(self):
        s = Study("test", _make_base_case())
        s.vary("load.axial_force", [10000, 50000])
        s.vary("mesh.global_size", [1.0, 2.0, 4.0])
        cases = s._generate_cases("grid")
        assert len(cases) == 6  # 2 x 3

    def test_vary_oat(self):
        s = Study("test", _make_base_case())
        s.vary("load.axial_force", [10000, 50000, 100000])
        cases = s._generate_cases("one_at_a_time")
        # baseline + 2 variants (10000 is baseline value, so only 50000 and 100000 are new)
        assert len(cases) == 3  # baseline + 2

    def test_add_case(self):
        s = Study("test", _make_base_case())
        custom = AnalysisCase(name="custom")
        s.add_case("my_case", custom)
        cases = s._generate_cases("grid")
        assert "my_case" in cases

    def test_case_names_descriptive(self):
        s = Study("test", _make_base_case())
        s.vary("load.axial_force", [10000, 50000])
        cases = s._generate_cases("grid")
        names = list(cases.keys())
        assert any("axial_force=10000" in n for n in names)
        assert any("axial_force=50000" in n for n in names)

    def test_case_values_applied(self):
        s = Study("test", _make_base_case())
        s.vary("load.axial_force", [10000, 50000])
        cases = s._generate_cases("grid")
        values = {c.load.axial_force for c in cases.values()}
        assert values == {10000, 50000}

    def test_chaining(self):
        s = (
            Study("test", _make_base_case())
            .vary("load.axial_force", [10000, 50000])
            .vary("mesh.global_size", [1.0, 2.0])
        )
        cases = s._generate_cases("grid")
        assert len(cases) == 4

    def test_no_sweeps_returns_baseline(self):
        s = Study("test", _make_base_case())
        cases = s._generate_cases("grid")
        assert len(cases) == 1
        assert "baseline" in cases


# ---------------------------------------------------------------------------
# Tests: Study execution (mocked)
# ---------------------------------------------------------------------------

class TestStudyExecution:
    @patch("feaweld.pipeline.study._run_single_case")
    def test_run_parallel(self, mock_run):
        base = _make_base_case()
        mock_run.side_effect = lambda case: _make_mock_result(case)

        s = Study("test", base).vary("load.axial_force", [10000, 50000])
        results = s.run(max_workers=1, mode="grid")

        assert results.n_cases == 2
        assert results.n_succeeded == 2
        assert results.n_failed == 0
        assert results.elapsed_seconds >= 0

    @patch("feaweld.pipeline.study.run_analysis")
    def test_run_sequential(self, mock_run):
        base = _make_base_case()
        mock_run.side_effect = lambda case: _make_mock_result(case)

        s = Study("test", base).vary("load.axial_force", [10000, 50000])
        results = s.run(max_workers=1, mode="grid")

        assert results.n_succeeded == 2

    @patch("feaweld.pipeline.study.run_analysis")
    def test_error_isolation(self, mock_run):
        base = _make_base_case()
        call_count = [0]

        def side_effect(case):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated failure")
            return _make_mock_result(case)

        mock_run.side_effect = side_effect

        s = Study("test", base).vary("load.axial_force", [10000, 50000])
        results = s.run(max_workers=1, mode="grid")

        assert results.n_succeeded == 1
        assert results.n_failed == 1

    @patch("feaweld.pipeline.study.run_analysis")
    def test_progress_callback(self, mock_run):
        base = _make_base_case()
        mock_run.side_effect = lambda case: _make_mock_result(case)

        progress_calls = []
        s = Study("test", base).vary("load.axial_force", [10000, 50000, 100000])
        s.run(max_workers=1, mode="grid",
              progress_callback=lambda name, done, total: progress_calls.append((name, done, total)))

        assert len(progress_calls) == 3
        assert progress_calls[-1][1] == 3  # last call: done=3
        assert progress_calls[-1][2] == 3  # total=3


# ---------------------------------------------------------------------------
# Tests: StudyResults container
# ---------------------------------------------------------------------------

class TestStudyResults:
    def test_properties(self):
        sr = StudyResults(study_name="test")
        assert sr.n_cases == 0
        assert sr.n_succeeded == 0
        assert len(sr) == 0

    def test_getitem(self):
        result = _make_mock_result(_make_base_case())
        sr = StudyResults(
            study_name="test",
            cases={"a": _make_base_case()},
            results={"a": result},
        )
        assert sr["a"] is result
        assert "a" in sr

    def test_iteration(self):
        sr = StudyResults(
            study_name="test",
            results={"a": _make_mock_result(_make_base_case()),
                     "b": _make_mock_result(_make_base_case())},
        )
        names = [name for name, _ in sr]
        assert len(names) == 2


# ---------------------------------------------------------------------------
# Tests: MetricSet
# ---------------------------------------------------------------------------

class TestMetricSet:
    def test_from_workflow_result(self):
        result = _make_mock_result(_make_base_case(), max_stress=200.0)
        m = MetricSet.from_workflow_result(result)
        assert m.max_von_mises is not None
        assert m.max_von_mises > 0
        assert m.mean_von_mises is not None
        assert m.max_displacement is not None
        assert m.fatigue_life is not None
        assert m.n_nodes == 3
        assert m.n_elements == 1

    def test_missing_data(self):
        result = WorkflowResult(case=_make_base_case())
        m = MetricSet.from_workflow_result(result)
        assert m.max_von_mises is None
        assert m.fatigue_life is None
        assert m.max_displacement is None

    def test_to_dict(self):
        m = MetricSet(max_von_mises=100.0, fatigue_life=1e6)
        d = m.to_dict()
        assert d["max_von_mises"] == 100.0
        assert d["fatigue_life"] == 1e6
        assert "safety_factor" in d


# ---------------------------------------------------------------------------
# Tests: ComparisonTable
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def _make_table(self):
        metrics = {
            "case_a": MetricSet(max_von_mises=100.0, fatigue_life=1e6),
            "case_b": MetricSet(max_von_mises=200.0, fatigue_life=5e5),
        }
        return ComparisonTable(case_names=["case_a", "case_b"], metrics=metrics)

    def test_to_rows(self):
        table = self._make_table()
        rows = table.to_rows()
        assert len(rows) == 2
        assert rows[0]["case"] == "case_a"
        assert rows[0]["max_von_mises"] == 100.0

    def test_to_text_table(self):
        table = self._make_table()
        text = table.to_text_table()
        assert "case_a" in text
        assert "case_b" in text
        assert "100.00" in text

    def test_delta_from_baseline(self):
        table = self._make_table()
        deltas = table.delta_from_baseline("case_a")
        assert len(deltas) == 1
        assert deltas[0]["case"] == "case_b"
        assert deltas[0]["max_von_mises_delta"] == 100.0
        assert deltas[0]["max_von_mises_pct"] == pytest.approx(100.0)

    def test_delta_baseline_not_found(self):
        table = self._make_table()
        with pytest.raises(ValueError, match="not found"):
            table.delta_from_baseline("nonexistent")


# ---------------------------------------------------------------------------
# Tests: Stress field difference
# ---------------------------------------------------------------------------

class TestStressFieldDifference:
    def test_compatible(self):
        a = StressField(values=np.array([[100, 0, 0, 0, 0, 0]], dtype=float))
        b = StressField(values=np.array([[60, 0, 0, 0, 0, 0]], dtype=float))
        diff = compute_stress_field_difference(a, b)
        assert diff.values[0, 0] == pytest.approx(40.0)

    def test_incompatible(self):
        a = StressField(values=np.zeros((3, 6)))
        b = StressField(values=np.zeros((5, 6)))
        with pytest.raises(ValueError, match="Incompatible"):
            compute_stress_field_difference(a, b)


# ---------------------------------------------------------------------------
# Tests: YAML I/O
# ---------------------------------------------------------------------------

class TestYAMLIO:
    def test_save_load_roundtrip(self, tmp_path):
        config = StudyConfig(
            name="roundtrip_test",
            base_case=_make_base_case(),
            parameters=[
                ParameterSweep(name="load.axial_force", values=[10000, 50000]),
            ],
            mode="grid",
            max_workers=2,
        )
        path = tmp_path / "study.yaml"
        save_study(config, path)
        assert path.exists()

        loaded = load_study(path)
        assert loaded.name == "roundtrip_test"
        assert len(loaded.parameters) == 1
        assert loaded.parameters[0].name == "load.axial_force"
        assert loaded.mode == "grid"


# ---------------------------------------------------------------------------
# Tests: Comparison visualization (matplotlib)
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


class TestComparisonVisualization:
    def _make_study_results(self):
        cases = {}
        results = {}
        for force in [10000, 50000, 100000]:
            name = f"axial_force={force}"
            case = _set_nested_attr(_make_base_case(), "load.axial_force", float(force))
            cases[name] = case
            results[name] = _make_mock_result(case, max_stress=force / 100.0)
        return StudyResults(
            study_name="test_viz",
            cases=cases,
            results=results,
            elapsed_seconds=1.0,
        )

    def test_plot_metric_comparison(self):
        from feaweld.visualization.comparison import plot_metric_comparison
        sr = self._make_study_results()
        fig = plot_metric_comparison(sr, "max_von_mises", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_parameter_sensitivity(self):
        from feaweld.visualization.comparison import plot_parameter_sensitivity
        sr = self._make_study_results()
        fig = plot_parameter_sensitivity(sr, "load.axial_force", "max_von_mises", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_stress_difference(self):
        from feaweld.visualization.comparison import plot_stress_difference
        mesh = FEMesh(
            nodes=np.array([[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=float),
            elements=np.array([[0, 1, 2]], dtype=np.int64),
            element_type=ElementType.TRI3,
        )
        a = StressField(values=np.array([[100, 0, 0, 0, 0, 0],
                                          [150, 0, 0, 0, 0, 0],
                                          [120, 0, 0, 0, 0, 0]], dtype=float))
        b = StressField(values=np.array([[80, 0, 0, 0, 0, 0],
                                          [100, 0, 0, 0, 0, 0],
                                          [90, 0, 0, 0, 0, 0]], dtype=float))
        fig = plot_stress_difference(mesh, a, b, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_stress_envelope(self):
        from feaweld.visualization.comparison import plot_stress_envelope
        sr = self._make_study_results()
        fig = plot_stress_envelope(sr, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_dashboard(self):
        from feaweld.visualization.comparison import comparison_dashboard
        sr = self._make_study_results()
        fig = comparison_dashboard(sr, show=False)
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 4
        plt.close(fig)
