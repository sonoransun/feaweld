"""Tests for 2D visualization, annotations, dashboards, and report figures.

Matplotlib-dependent tests are skipped when matplotlib is not installed.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from feaweld.core.types import (
    ElementType, FEAResults, FEMesh, StressField,
    WeldGroupShape, WeldGroupProperties, WeldLineDefinition,
    SNCurve, SNSegment, SNStandard,
)

# Guard: skip all tests if matplotlib is not available
matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(n_side: int = 5) -> FEMesh:
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
        node_sets={"weld_toe": np.array([0, 1, 2])},
    )


def _make_stress(n_nodes: int) -> StressField:
    vals = np.zeros((n_nodes, 6))
    vals[:, 0] = np.linspace(50, 250, n_nodes)  # sigma_xx
    vals[:, 1] = np.linspace(30, 150, n_nodes)
    return StressField(values=vals)


def _make_sn_curve() -> SNCurve:
    return SNCurve(
        name="TestFAT90",
        standard=SNStandard.IIW,
        segments=[
            SNSegment(m=3.0, C=90.0**3 * 2e6, stress_threshold=0.0),
        ],
        cutoff_cycles=1e7,
    )


# ---------------------------------------------------------------------------
# Tests: 2D Plots
# ---------------------------------------------------------------------------

class TestPlots2D:
    def test_plot_through_thickness(self):
        from feaweld.postprocess.linearization import LinearizationResult
        from feaweld.visualization.plots_2d import plot_through_thickness

        z = np.linspace(0, 10, 20)
        total = np.zeros((20, 6))
        total[:, 0] = 100 + 50 * (z / 10)  # linearly varying

        result = LinearizationResult(
            membrane=np.array([125, 0, 0, 0, 0, 0], dtype=float),
            bending=np.array([25, 0, 0, 0, 0, 0], dtype=float),
            peak=np.array([5, 0, 0, 0, 0, 0], dtype=float),
            z_coords=z,
            total_stress=total,
            membrane_scalar=125.0,
            bending_scalar=25.0,
            peak_scalar=5.0,
            membrane_plus_bending_scalar=150.0,
        )

        fig = plot_through_thickness(result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_hotspot_extrapolation(self):
        from feaweld.postprocess.hotspot import HotSpotResult, HotSpotType
        from feaweld.visualization.plots_2d import plot_hotspot_extrapolation

        result = HotSpotResult(
            hot_spot_stress=220.0,
            reference_stresses=[180.0, 150.0],
            reference_distances=[4.0, 10.0],
            extrapolation_type=HotSpotType.TYPE_A,
            weld_toe_location=np.array([0.0, 0.0, 0.0]),
        )
        fig = plot_hotspot_extrapolation(result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_dong_decomposition(self):
        from feaweld.postprocess.dong import DongResult
        from feaweld.visualization.plots_2d import plot_dong_decomposition

        result = DongResult(
            membrane_stress=np.array([100.0, 110.0, 105.0]),
            bending_stress=np.array([30.0, 25.0, 35.0]),
            structural_stress=np.array([130.0, 135.0, 140.0]),
            bending_ratio=np.array([0.23, 0.19, 0.25]),
        )
        fig = plot_dong_decomposition(result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sn_curve_basic(self):
        from feaweld.visualization.plots_2d import plot_sn_curve
        curve = _make_sn_curve()
        fig = plot_sn_curve(curve, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sn_curve_with_operating_point(self):
        from feaweld.visualization.plots_2d import plot_sn_curve
        curve = _make_sn_curve()
        fig = plot_sn_curve(curve, stress_range=100.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_stress_along_path(self):
        from feaweld.visualization.plots_2d import plot_stress_along_path
        d = np.linspace(0, 50, 20)
        s = 200 - 3 * d
        fig = plot_stress_along_path(d, s, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.parametrize("shape", [
        WeldGroupShape.LINE,
        WeldGroupShape.BOX,
        WeldGroupShape.CIRCULAR,
        WeldGroupShape.C_SHAPE,
        WeldGroupShape.L_SHAPE,
        WeldGroupShape.PARALLEL,
        WeldGroupShape.I_SHAPE,
        WeldGroupShape.T_SHAPE,
        WeldGroupShape.U_SHAPE,
    ])
    def test_plot_weld_group_geometry(self, shape):
        from feaweld.visualization.plots_2d import plot_weld_group_geometry
        fig = plot_weld_group_geometry(shape, d=100.0, b=50.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_asme_check(self):
        from feaweld.postprocess.nominal import StressCategorization
        from feaweld.visualization.plots_2d import plot_asme_check

        cat = StressCategorization(
            membrane=120.0, bending=40.0, peak=15.0,
            total=175.0, stress_intensity=170.0,
        )
        fig = plot_asme_check(cat, S_m=150.0, S_y=250.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cross_section_stress(self):
        from feaweld.visualization.plots_2d import plot_cross_section_stress
        # Need a mesh with non-zero z variation for cross-section slicing
        n = 5
        xs = np.linspace(0, 10, n)
        zs = np.linspace(0, 10, n)
        xx, zz = np.meshgrid(xs, zs)
        nodes = np.column_stack([xx.ravel(), np.full(n * n, 5.0), zz.ravel()])
        elems = []
        for j in range(n - 1):
            for i in range(n - 1):
                n0 = j * n + i
                elems.append([n0, n0 + 1, n0 + n])
                elems.append([n0 + 1, n0 + n + 1, n0 + n])
        mesh = FEMesh(
            nodes=nodes, elements=np.array(elems, dtype=np.int64),
            element_type=ElementType.TRI3,
        )
        stress = _make_stress(mesh.n_nodes)
        fig = plot_cross_section_stress(mesh, stress, y_level=5.0, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Annotations
# ---------------------------------------------------------------------------

class TestTheme:
    def test_get_cmap_returns_strings(self):
        from feaweld.visualization.theme import get_cmap, _CMAP_REGISTRY
        for purpose in _CMAP_REGISTRY:
            result = get_cmap(purpose)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_get_cmap_unknown_raises(self):
        from feaweld.visualization.theme import get_cmap
        with pytest.raises(KeyError):
            get_cmap("nonexistent_purpose")

    def test_apply_feaweld_style(self):
        from feaweld.visualization.theme import apply_feaweld_style
        apply_feaweld_style()  # should not raise

    def test_color_constants_are_hex(self):
        from feaweld.visualization import theme
        for name in ("FEAWELD_BLUE", "FEAWELD_RED", "FEAWELD_ORANGE",
                     "FEAWELD_GREEN", "FEAWELD_DARK", "FEAWELD_GRAY"):
            val = getattr(theme, name)
            assert val.startswith("#") and len(val) == 7


class TestAnnotations:
    def test_find_critical_points(self):
        from feaweld.visualization.annotations import find_critical_points
        mesh = _make_mesh()
        stress = _make_stress(mesh.n_nodes)
        points = find_critical_points(mesh, stress, n_max=3)
        assert len(points) == 3
        # First point should be the max stress
        vm = stress.von_mises
        assert points[0].value == pytest.approx(float(np.max(vm)), rel=0.01)

    def test_find_critical_points_with_weld_line(self):
        from feaweld.visualization.annotations import find_critical_points
        mesh = _make_mesh()
        stress = _make_stress(mesh.n_nodes)
        weld_line = WeldLineDefinition(
            name="test",
            node_ids=np.array([0, 1, 2]),
            plate_thickness=10.0,
            normal_direction=np.array([0.0, 1.0, 0.0]),
        )
        points = find_critical_points(mesh, stress, n_max=5, weld_line=weld_line)
        # Should include weld toe point
        categories = [p.category for p in points]
        assert "weld_toe" in categories

    def test_safety_factor_field(self):
        from feaweld.visualization.annotations import safety_factor_field
        mesh = _make_mesh()
        stress = _make_stress(mesh.n_nodes)
        sf = safety_factor_field(mesh, stress, allowable=200.0)
        assert sf.shape == (mesh.n_nodes,)
        assert np.all(sf >= 0)
        assert np.all(sf <= 10.0)

    def test_format_engineering_value(self):
        from feaweld.visualization.annotations import format_engineering_value
        assert format_engineering_value(245.3, "MPa") == "245.3 MPa"
        assert "e+" in format_engineering_value(1.23e6, "cycles")
        assert format_engineering_value(0.0, "mm") == "0.0 mm"

    def test_annotate_2d(self):
        from feaweld.visualization.annotations import CriticalPoint, annotate_2d
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 100])
        points = [
            CriticalPoint(np.array([5, 50, 0]), 50.0, "Test", "warning", "stress"),
        ]
        annotate_2d(ax, points)  # should not crash
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    def _make_workflow_result(self):
        from feaweld.pipeline.workflow import (
            AnalysisCase, WorkflowResult, MaterialConfig, GeometryConfig,
            PostProcessConfig,
        )
        mesh = _make_mesh()
        stress = _make_stress(mesh.n_nodes)
        case = AnalysisCase(
            name="test_dashboard",
            postprocess=PostProcessConfig(sn_curve="IIW_FAT90"),
        )
        return WorkflowResult(
            case=case,
            mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=stress),
            fatigue_results={"hotspot_linear": {"stress_range": 100.0, "life": 1.46e6}},
        )

    def test_engineering_dashboard(self):
        from feaweld.visualization.dashboard import engineering_dashboard
        wf = self._make_workflow_result()
        fig = engineering_dashboard(wf, show=False)
        assert isinstance(fig, plt.Figure)
        # 2x3 = 6 subplots
        assert len(fig.get_axes()) == 6
        plt.close(fig)

    def test_fatigue_dashboard(self):
        from feaweld.visualization.dashboard import fatigue_dashboard
        wf = self._make_workflow_result()
        fig = fatigue_dashboard(wf, show=False)
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 4
        plt.close(fig)

    def test_comparison_view(self):
        from feaweld.visualization.dashboard import comparison_view
        mesh = _make_mesh()
        s1 = _make_stress(mesh.n_nodes)
        s2 = _make_stress(mesh.n_nodes)
        fig = comparison_view(
            [("Case A", mesh, s1), ("Case B", mesh, s2)],
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_postprocess_summary(self):
        from feaweld.visualization.dashboard import postprocess_summary
        wf = self._make_workflow_result()
        wf.postprocess_results = {"hotspot": {"max_stress": 200.0, "life": 1e6}}
        fig = postprocess_summary(wf, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: Report Figures
# ---------------------------------------------------------------------------

class TestReportFigures:
    def test_figure_to_base64(self):
        from feaweld.visualization.report_figures import figure_to_base64
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        b64 = figure_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100
        # Verify it's valid base64 that decodes to PNG
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"

    def test_html_img_tag(self):
        from feaweld.visualization.report_figures import html_img_tag
        tag = html_img_tag("AAAA", alt="test", width="50%")
        assert "data:image/png;base64,AAAA" in tag
        assert 'alt="test"' in tag
        assert "50%" in tag

    def test_generate_report_figures_minimal(self):
        from feaweld.visualization.report_figures import generate_report_figures
        from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult, PostProcessConfig

        mesh = _make_mesh()
        stress = _make_stress(mesh.n_nodes)
        case = AnalysisCase(
            name="test_figs",
            postprocess=PostProcessConfig(sn_curve="IIW_FAT90"),
        )
        wf = WorkflowResult(
            case=case,
            mesh=mesh,
            fea_results=FEAResults(mesh=mesh, stress=stress),
        )

        figures = generate_report_figures(wf)
        assert isinstance(figures, dict)
        # Should at least have the stress distribution
        assert "stress_distribution" in figures
        # Dashboard should be generated
        assert "engineering_dashboard" in figures

    def test_generate_report_figures_empty_result(self):
        from feaweld.visualization.report_figures import generate_report_figures
        from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult

        wf = WorkflowResult(case=AnalysisCase(name="empty"))
        figures = generate_report_figures(wf)
        assert isinstance(figures, dict)
