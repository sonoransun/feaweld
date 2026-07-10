"""Tests for HTML report generation and the figure-spec registry.

Covers Jinja2 template loading/rendering, end-to-end report generation
(escaping, conditional sections), the declarative FIGURE_SPECS availability
predicates, one-shot Plotly CDN injection, graceful degradation without
pyvista, and the parametric-study comparison report.

Matplotlib drives the embedded figures, so the module skips when matplotlib
is not installed.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.pipeline.workflow import (
    AnalysisCase, GeometryConfig, PostProcessConfig, WorkflowResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(n_side: int = 4) -> FEMesh:
    xs = np.linspace(0, 10, n_side)
    ys = np.linspace(0, 10, n_side)
    xx, yy = np.meshgrid(xs, ys)
    n_nodes = n_side * n_side
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_nodes)])
    elems = []
    for j in range(n_side - 1):
        for i in range(n_side - 1):
            n0 = j * n_side + i
            elems.append([n0, n0 + 1, n0 + n_side])
            elems.append([n0 + 1, n0 + n_side + 1, n0 + n_side])
    return FEMesh(
        nodes=nodes,
        elements=np.array(elems, dtype=np.int64),
        element_type=ElementType.TRI3,
    )


def _make_stress(n_nodes: int) -> StressField:
    vals = np.zeros((n_nodes, 6))
    vals[:, 0] = np.linspace(50, 250, n_nodes)
    vals[:, 1] = np.linspace(30, 150, n_nodes)
    return StressField(values=vals)


def _make_workflow_result(name: str = "case1", **kwargs) -> WorkflowResult:
    mesh = _make_mesh()
    stress = _make_stress(mesh.n_nodes)
    case = AnalysisCase(
        name=name,
        postprocess=PostProcessConfig(sn_curve="IIW_FAT90"),
        **{k: v for k, v in kwargs.items() if k in ("output_dir",)},
    )
    wf_kwargs = {k: v for k, v in kwargs.items() if k not in ("output_dir",)}
    return WorkflowResult(
        case=case,
        mesh=mesh,
        fea_results=FEAResults(mesh=mesh, stress=stress),
        **wf_kwargs,
    )


def _make_study_results():
    from feaweld.pipeline.study import StudyResults

    mesh = _make_mesh()
    cases = {}
    results = {}
    for name, leg, scale in [("leg6", 6.0, 1.0), ("leg8", 8.0, 1.15), ("leg10", 10.0, 1.3)]:
        vals = np.zeros((mesh.n_nodes, 6))
        vals[:, 0] = np.linspace(50, 250, mesh.n_nodes) * scale
        case = AnalysisCase(
            name=name,
            geometry=GeometryConfig(weld_leg_size=leg),
            postprocess=PostProcessConfig(sn_curve="IIW_FAT90"),
        )
        results[name] = WorkflowResult(
            case=case, mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=StressField(values=vals)),
            fatigue_results={"hotspot_linear": {"stress_range": 100.0 * scale,
                                                "life": 2.0e6 / scale}},
        )
        cases[name] = case
    return StudyResults(
        study_name="leg_sweep", cases=cases, results=results,
        errors={}, elapsed_seconds=2.0,
    )


# ---------------------------------------------------------------------------
# Template loading / rendering
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_env_loads_all_templates(self):
        from feaweld.pipeline.report import _get_env
        env = _get_env()
        for name in ("base.html.j2", "report.html.j2", "comparison.html.j2"):
            assert env.get_template(name) is not None

    def test_report_template_renders_minimal_context(self):
        from feaweld.pipeline.report import _get_env
        template = _get_env().get_template("report.html.j2")
        html = template.render(
            title="Minimal", body_class="report", version="0.0.0",
            timestamp="now",
            case={"name": "Minimal", "description": "", "joint_type": "fillet_t",
                  "base_metal": "A36", "weld_metal": "E70XX",
                  "solver_type": "linear_elastic", "backend": "auto",
                  "success": True},
            mesh=None, fea_rows=[], postprocess_rows=[], fatigue_rows=[],
            probabilistic_rows=[], figures=[], interactive_figures=[],
            plotly_cdn=False, warnings=[], errors=[],
        )
        assert "<!DOCTYPE html>" in html
        assert "Minimal" in html


# ---------------------------------------------------------------------------
# End-to-end report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_writes_file(self, tmp_path):
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(output_dir=str(tmp_path))
        path = generate_report(wf)
        import os
        assert os.path.isfile(path)
        assert wf.report_path == path

    def test_escapes_case_name(self, tmp_path):
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(name="<b>x", output_dir=str(tmp_path))
        path = generate_report(wf)
        html = open(path).read()
        assert "&lt;b&gt;x" in html
        assert "<b>x" not in html

    def test_no_probabilistic_section_when_absent(self, tmp_path):
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(output_dir=str(tmp_path))
        assert not wf.probabilistic_results
        html = open(generate_report(wf)).read()
        assert "Probabilistic Analysis" not in html

    def test_warnings_box_present(self, tmp_path):
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(
            output_dir=str(tmp_path), warnings=["singularity flagged at weld toe"],
        )
        html = open(generate_report(wf)).read()
        assert "Warnings" in html
        assert "singularity flagged at weld toe" in html


# ---------------------------------------------------------------------------
# FIGURE_SPECS availability predicates (no rendering)
# ---------------------------------------------------------------------------

class TestFigureSpecPredicates:
    def _specs(self):
        from feaweld.visualization.report_figures import FIGURE_SPECS
        return {spec.key: spec for spec in FIGURE_SPECS}

    def _base(self, **kwargs) -> WorkflowResult:
        mesh = _make_mesh()
        return WorkflowResult(case=AnalysisCase(name="p"), mesh=mesh, **kwargs)

    def test_rainflow_toggle(self):
        spec = self._specs()["rainflow"]
        present = self._base()
        present.postprocess_results = {"rainflow": [(50.0, 0.0, 1.0), (80.0, 0.0, 0.5)]}
        assert spec.available(present) is True
        assert spec.available(self._base()) is False

    def test_rainflow_matrix_toggle(self):
        spec = self._specs()["rainflow_matrix"]
        present = self._base()
        present.postprocess_results = {"rainflow": [
            (50.0, 10.0, 1.0), (80.0, 20.0, 1.0),
            (60.0, 15.0, 0.5), (100.0, 25.0, 1.0),
        ]}
        assert spec.available(present) is True
        # A 2-D matrix needs at least 4 cycles to be worth drawing.
        few = self._base()
        few.postprocess_results = {"rainflow": [(50.0, 10.0, 1.0)]}
        assert spec.available(few) is False
        assert spec.available(self._base()) is False

    def test_damage_per_block_toggle(self):
        spec = self._specs()["damage_per_block"]
        cycles = [(50.0, 0.0, 1.0), (80.0, 0.0, 0.5)]
        present = self._base()
        present.postprocess_results = {"rainflow": cycles}
        # Default case carries a resolvable IIW_FAT90 curve.
        assert spec.available(present) is True
        assert spec.available(self._base()) is False
        bad_curve = self._base()
        bad_curve.postprocess_results = {"rainflow": cycles}
        bad_curve.case.postprocess.sn_curve = "NOT_A_STANDARD"
        assert spec.available(bad_curve) is False

    def test_haigh_diagram_toggle(self):
        from feaweld.pipeline.workflow import FatigueConfig

        spec = self._specs()["haigh_diagram"]
        cycles = [(50.0, 10.0, 1.0), (80.0, 20.0, 0.5)]

        on = self._base()
        on.case.fatigue = FatigueConfig(mean_stress_correction="goodman")
        on.postprocess_results = {"rainflow": cycles}
        assert spec.available(on) is True

        # Default correction is "none" — no Haigh diagram.
        no_correction = self._base()
        no_correction.postprocess_results = {"rainflow": cycles}
        assert no_correction.case.fatigue.mean_stress_correction == "none"
        assert spec.available(no_correction) is False

        no_cycles = self._base()
        no_cycles.case.fatigue = FatigueConfig(mean_stress_correction="gerber")
        assert spec.available(no_cycles) is False

        # Unresolvable sigma_u degrades to unavailable, never raises.
        bad_material = self._base()
        bad_material.case.fatigue = FatigueConfig(mean_stress_correction="goodman")
        bad_material.case.material.base_metal = "unobtainium"
        bad_material.postprocess_results = {"rainflow": cycles}
        assert spec.available(bad_material) is False

    def test_sobol_toggle(self):
        spec = self._specs()["sobol_indices"]
        present = self._base()
        present.probabilistic_results = {
            "sobol": {"first_order": {"a": 0.5}, "total": {"a": 0.6}}
        }
        assert spec.available(present) is True
        assert spec.available(self._base()) is False

    def test_mc_histogram_toggle(self):
        spec = self._specs()["mc_histogram"]
        present = self._base()
        present.probabilistic_results = {"results": np.array([1.0, 2.0, 3.0])}
        assert spec.available(present) is True
        assert spec.available(self._base()) is False

    def test_temperature_history_toggle(self):
        spec = self._specs()["temperature_history"]
        mesh = _make_mesh()
        temp2d = np.tile(np.linspace(20, 500, mesh.n_nodes), (5, 1))
        ts = np.linspace(0.0, 4.0, 5)
        present = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(mesh=mesh, temperature=temp2d, time_steps=ts),
        )
        assert spec.available(present) is True
        # A steady (1-D) temperature field is not a history.
        steady = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(
                mesh=mesh, temperature=np.linspace(20, 500, mesh.n_nodes),
            ),
        )
        assert spec.available(steady) is False

    def test_deformed_shape_toggle(self):
        spec = self._specs()["deformed_shape"]
        mesh = _make_mesh()
        present = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(
                mesh=mesh, displacement=np.zeros((mesh.n_nodes, 3)),
                stress=_make_stress(mesh.n_nodes),
            ),
        )
        assert spec.available(present) is True
        absent = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=_make_stress(mesh.n_nodes)),
        )
        assert spec.available(absent) is False

    def test_nodal_stress_specs_reject_cell_located_stress(self):
        from feaweld.visualization.report_figures import _stress_is_nodal

        mesh = _make_mesh()
        # Cell-located stress has one row per element, not per node; the
        # pyvista/tricontourf builders would crash on it.
        assert mesh.n_elements != mesh.n_nodes
        cell_stress = StressField(values=np.zeros((mesh.n_elements, 6)))
        wf = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=cell_stress),
        )
        assert _stress_is_nodal(wf) is False

        specs = self._specs()
        for key in ("stress_contour_3d", "stress_contour_2d",
                    "fatigue_life_map", "damage_map"):
            assert specs[key].available(wf) is False, key
        # The plain histogram works for any row count.
        assert specs["stress_distribution"].available(wf) is True

        # Nodal stress re-enables the stress-coloured contours.
        nodal = WorkflowResult(
            case=AnalysisCase(name="p"), mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=_make_stress(mesh.n_nodes)),
        )
        assert _stress_is_nodal(nodal) is True
        assert specs["stress_contour_3d"].available(nodal) is True
        assert specs["stress_contour_2d"].available(nodal) is True

    def test_life_field_caps_infinite_and_huge_life(self):
        from feaweld.visualization.report_figures import _life_field

        mesh = _make_mesh()
        # Zero stress → life well beyond the S-N cutoff (infinite / huge).
        wf = WorkflowResult(
            case=AnalysisCase(
                name="p", postprocess=PostProcessConfig(sn_curve="IIW_FAT90"),
            ),
            mesh=mesh,
            fea_results=FEAResults(
                mesh=mesh, stress=StressField(values=np.zeros((mesh.n_nodes, 6))),
            ),
        )
        life = _life_field(wf)
        assert life.shape == (mesh.n_nodes,)
        assert np.all(np.isfinite(life))
        assert float(np.max(life)) <= 1e12
        assert float(np.min(life)) >= 1.0


# ---------------------------------------------------------------------------
# Spectrum-fatigue figure builders and report smoke test
# ---------------------------------------------------------------------------

class TestSpectrumFigures:
    _CYCLES = [
        (50.0, 10.0, 1.0),
        (80.0, 20.0, 1.0),
        (120.0, 30.0, 0.5),
        (60.0, 15.0, 1.0),
        (100.0, 25.0, 1.0),
        (40.0, 5.0, 1.0),
    ]

    def _spectrum_wf(self, correction: str = "goodman", **kwargs) -> WorkflowResult:
        from feaweld.pipeline.workflow import FatigueConfig
        wf = _make_workflow_result(**kwargs)
        wf.case.fatigue = FatigueConfig(mean_stress_correction=correction)
        wf.postprocess_results = {"rainflow": list(self._CYCLES)}
        return wf

    def _build(self, key: str, wf: WorkflowResult):
        from feaweld.visualization.report_figures import FIGURE_SPECS
        spec = {s.key: s for s in FIGURE_SPECS}[key]
        assert spec.available(wf) is True
        return spec.build(wf)

    @pytest.mark.parametrize(
        "key", ["rainflow_matrix", "damage_per_block", "haigh_diagram"],
    )
    def test_builder_returns_figure(self, key):
        import matplotlib.pyplot as plt
        fig = self._build(key, self._spectrum_wf())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_haigh_builder_gerber(self):
        import matplotlib.pyplot as plt
        fig = self._build("haigh_diagram", self._spectrum_wf(correction="gerber"))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_report_includes_spectrum_figures(self, tmp_path):
        from feaweld.pipeline.report import generate_report
        from feaweld.visualization.report_figures import generate_report_figures

        wf = self._spectrum_wf(output_dir=str(tmp_path))
        figures = generate_report_figures(wf)
        for key in ("rainflow", "rainflow_matrix", "damage_per_block",
                    "haigh_diagram"):
            assert key in figures, key

        html = open(generate_report(wf)).read()
        assert "Rainflow Range-Mean Matrix" in html
        assert "Damage by Stress-Range Bin" in html
        assert "Haigh Mean-Stress Diagram" in html

    def test_report_omits_haigh_without_correction(self, tmp_path):
        from feaweld.pipeline.report import generate_report

        wf = _make_workflow_result(output_dir=str(tmp_path))
        wf.postprocess_results = {"rainflow": list(self._CYCLES)}
        assert wf.case.fatigue.mean_stress_correction == "none"

        html = open(generate_report(wf)).read()
        assert "Rainflow Range-Mean Matrix" in html
        assert "Haigh Mean-Stress Diagram" not in html


# ---------------------------------------------------------------------------
# Plotly CDN injection (once, opt-in)
# ---------------------------------------------------------------------------

class TestPlotlyCDN:
    def test_cdn_injected_once_when_interactive(self, tmp_path):
        pytest.importorskip("plotly")
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(output_dir=str(tmp_path))
        html = open(generate_report(wf, interactive=True)).read()
        assert html.count("cdn.plot.ly") == 1

    def test_cdn_absent_when_static(self, tmp_path):
        pytest.importorskip("plotly")
        from feaweld.pipeline.report import generate_report
        wf = _make_workflow_result(output_dir=str(tmp_path))
        html = open(generate_report(wf, interactive=False)).read()
        assert html.count("cdn.plot.ly") == 0


# ---------------------------------------------------------------------------
# Graceful degradation without pyvista
# ---------------------------------------------------------------------------

class TestNoPyvista:
    def test_report_generates_without_pyvista(self, tmp_path, monkeypatch):
        import os

        # A None entry in sys.modules makes `import pyvista` raise ImportError,
        # simulating an environment where the optional dep is not installed.
        monkeypatch.setitem(sys.modules, "pyvista", None)

        from feaweld.visualization.report_figures import generate_report_figures
        from feaweld.pipeline.report import generate_report

        wf = _make_workflow_result(output_dir=str(tmp_path))

        figures = generate_report_figures(wf)
        pyvista_keys = {
            "mesh_preview", "stress_contour_3d", "deformed_shape",
            "temperature_field", "fatigue_life_map", "damage_map",
        }
        assert not (pyvista_keys & set(figures))
        # Matplotlib figures still render.
        assert "stress_distribution" in figures

        path = generate_report(wf)
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_generates_with_baseline_and_sensitivity(self, tmp_path):
        import os
        from feaweld.pipeline.comparison import generate_comparison_report

        sr = _make_study_results()
        path = generate_comparison_report(sr, str(tmp_path), baseline="leg6")
        assert os.path.isfile(path)
        html = open(path).read()
        assert "Deltas vs. Baseline" in html
        assert "Sensitivity:" in html

    def test_no_delta_section_without_baseline(self, tmp_path):
        import os
        from feaweld.pipeline.comparison import generate_comparison_report

        sr = _make_study_results()
        path = generate_comparison_report(sr, str(tmp_path))
        assert os.path.isfile(path)
        html = open(path).read()
        assert "Deltas vs. Baseline" not in html

    def test_metric_column_survives_failed_first_case(self):
        from feaweld.pipeline.comparison import (
            ComparisonTable, MetricSet, _metric_table_ctx,
        )

        mesh = _make_mesh()
        # First case failed (no FEA results) → all-None metrics; the second
        # case carries a stress field. The metric columns must reflect the
        # union over all rows, not just the (empty) first one.
        r_fail = WorkflowResult(case=AnalysisCase(name="fail"))
        r_ok = WorkflowResult(
            case=AnalysisCase(name="ok"), mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=_make_stress(mesh.n_nodes)),
        )
        table = ComparisonTable(
            case_names=["fail", "ok"],
            metrics={
                "fail": MetricSet.from_workflow_result(r_fail),
                "ok": MetricSet.from_workflow_result(r_ok),
            },
        )
        ctx = _metric_table_ctx(table)
        assert ctx is not None
        assert "Max VM (MPa)" in ctx["headers"]
        # Both cases keep a row.
        assert len(ctx["rows"]) == 2
