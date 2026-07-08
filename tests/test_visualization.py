"""Tests for the visualization and export modules.

PyVista-dependent tests are skipped when pyvista is not installed.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
import types

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField

# Guard: skip pyvista-dependent tests when the library is not available.
pyvista = pytest.importorskip("pyvista")

from feaweld.visualization.export import export_gltf, export_png, export_vtk
from feaweld.visualization.fatigue_maps import plot_damage, plot_fatigue_life
from feaweld.visualization.report_figures import plotter_to_base64
from feaweld.visualization.stress_plots import (
    plot_deformed,
    plot_stress_field,
    plot_temperature_field,
    resolve_component,
    resolve_grid,
    stress_field_to_pyvista,
)
from feaweld.visualization import enhanced_3d as e3


def _can_screenshot() -> bool:
    """True when an off-screen render backend is available on this machine."""
    try:
        pl = pyvista.Plotter(off_screen=True)
        pl.add_mesh(pyvista.Sphere())
        pl.screenshot(return_img=True)
        pl.close()
        return True
    except Exception:
        return False


requires_render = pytest.mark.skipif(
    not _can_screenshot(), reason="no off-screen render backend"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tri_mesh(n_side: int = 5) -> FEMesh:
    """Create a small triangulated plate mesh."""
    xs = np.linspace(0, 1, n_side)
    ys = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(xs, ys)
    n_nodes = n_side * n_side
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_nodes)])
    elems: list[list[int]] = []
    for j in range(n_side - 1):
        for i in range(n_side - 1):
            n0 = j * n_side + i
            n1 = n0 + 1
            n2 = n0 + n_side
            n3 = n2 + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    return FEMesh(
        nodes=nodes,
        elements=np.array(elems, dtype=np.int64),
        element_type=ElementType.TRI3,
    )


def _make_stress(n_nodes: int) -> StressField:
    """Generate a simple uniaxial stress field."""
    vals = np.zeros((n_nodes, 6), dtype=np.float64)
    vals[:, 0] = np.linspace(50, 200, n_nodes)  # sigma_xx
    return StressField(values=vals)


def _make_results(mesh: FEMesh) -> FEAResults:
    stress = _make_stress(mesh.n_nodes)
    disp = np.random.default_rng(42).normal(size=(mesh.n_nodes, 3)) * 0.01
    temp = np.linspace(20, 500, mesh.n_nodes)
    return FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=stress,
        temperature=temp,
    )


def _make_tet_mesh(n: int = 4) -> FEMesh:
    """A structured cube split into TET4 elements, with weld/toe sets.

    Four nodes per axis (64 nodes) keeps the default-median threshold split
    non-empty in both directions, so the ``above``/``below`` filters both
    produce renderable sub-meshes.
    """
    xs = np.linspace(0.0, 1.0, n)
    idx: dict[tuple[int, int, int], int] = {}
    pts: list[list[float]] = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                idx[(i, j, k)] = len(pts)
                pts.append([xs[i], xs[j], xs[k]])
    tets: list[list[int]] = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                c = [
                    idx[(i, j, k)], idx[(i + 1, j, k)],
                    idx[(i, j + 1, k)], idx[(i + 1, j + 1, k)],
                    idx[(i, j, k + 1)], idx[(i + 1, j, k + 1)],
                    idx[(i, j + 1, k + 1)], idx[(i + 1, j + 1, k + 1)],
                ]
                tets += [
                    [c[0], c[1], c[3], c[7]], [c[0], c[1], c[7], c[5]],
                    [c[0], c[5], c[7], c[4]], [c[0], c[3], c[2], c[7]],
                    [c[0], c[2], c[6], c[7]],
                ]
    return FEMesh(
        nodes=np.array(pts, dtype=np.float64),
        elements=np.array(tets, dtype=np.int64),
        element_type=ElementType.TET4,
        element_sets={"weld": np.array([0, 1, 2, 3, 4], dtype=np.int64)},
        node_sets={"toe": np.array([0, 1, 2], dtype=np.int64)},
    )


def _make_gradient_stress(mesh: FEMesh) -> StressField:
    """A spatially coherent stress field: sigma_yy rises with the y coordinate."""
    y = mesh.nodes[:, 1]
    vals = np.zeros((mesh.n_nodes, 6), dtype=np.float64)
    vals[:, 1] = 50.0 + 150.0 * (y - y.min()) / max(float(np.ptp(y)), 1e-9)
    vals[:, 0] = 0.2 * vals[:, 1]
    return StressField(values=vals)


# ---------------------------------------------------------------------------
# stress_field_to_pyvista
# ---------------------------------------------------------------------------


class TestStressFieldToPyvista:

    def test_creates_valid_grid(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        grid = stress_field_to_pyvista(mesh, stress)
        assert grid.n_points == mesh.n_nodes
        assert grid.n_cells == mesh.n_elements
        # Check that expected point-data arrays exist.
        assert "von_mises" in grid.point_data
        assert "tresca" in grid.point_data
        assert "principal_1" in grid.point_data

    def test_point_data_shapes(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        grid = stress_field_to_pyvista(mesh, stress)
        assert grid.point_data["von_mises"].shape == (mesh.n_nodes,)
        assert grid.point_data["stress_xx"].shape == (mesh.n_nodes,)


# ---------------------------------------------------------------------------
# export_vtk
# ---------------------------------------------------------------------------


class TestExportVTK:

    def test_writes_file(self) -> None:
        mesh = _make_tri_mesh()
        results = _make_results(mesh)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.vtu")
            export_vtk(mesh, results, path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0

    def test_vtk_extension(self) -> None:
        mesh = _make_tri_mesh()
        results = _make_results(mesh)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.vtk")
            export_vtk(mesh, results, path)
            assert os.path.isfile(path)

    def test_minimal_results(self) -> None:
        """Export should work even with only a mesh and no fields."""
        mesh = _make_tri_mesh()
        results = FEAResults(mesh=mesh)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "minimal.vtu")
            export_vtk(mesh, results, path)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Plot functions (show=False for headless)
# ---------------------------------------------------------------------------


class TestPlotFunctions:
    """Ensure plot functions run without errors when show=False."""

    def test_plot_stress_field(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        plotter = plot_stress_field(mesh, stress, component="von_mises", show=False)
        assert plotter is not None

    def test_plot_stress_field_components(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        for comp in ("von_mises", "tresca", "xx", "principal_1"):
            plotter = plot_stress_field(mesh, stress, component=comp, show=False)
            assert plotter is not None

    def test_plot_stress_field_invalid_component(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        with pytest.raises(ValueError, match="Unknown component"):
            plot_stress_field(mesh, stress, component="nonexistent", show=False)

    def test_plot_deformed_no_stress(self) -> None:
        mesh = _make_tri_mesh()
        disp = np.random.default_rng(0).normal(size=(mesh.n_nodes, 3)) * 0.01
        plotter = plot_deformed(mesh, disp, scale=5.0, show=False)
        assert plotter is not None

    def test_plot_deformed_with_stress(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        disp = np.random.default_rng(0).normal(size=(mesh.n_nodes, 3)) * 0.01
        plotter = plot_deformed(mesh, disp, stress=stress, show=False)
        assert plotter is not None

    def test_plot_temperature_field(self) -> None:
        mesh = _make_tri_mesh()
        temp = np.linspace(20, 800, mesh.n_nodes)
        plotter = plot_temperature_field(mesh, temp, show=False)
        assert plotter is not None

    def test_plot_fatigue_life(self) -> None:
        mesh = _make_tri_mesh()
        life = np.random.default_rng(1).uniform(1e3, 1e7, mesh.n_nodes)
        plotter = plot_fatigue_life(mesh, life, show=False)
        assert plotter is not None

    def test_plot_fatigue_life_no_log(self) -> None:
        mesh = _make_tri_mesh()
        life = np.random.default_rng(1).uniform(1e3, 1e7, mesh.n_nodes)
        plotter = plot_fatigue_life(mesh, life, log_scale=False, show=False)
        assert plotter is not None

    def test_plot_damage(self) -> None:
        mesh = _make_tri_mesh()
        damage = np.random.default_rng(2).uniform(0, 1.5, mesh.n_nodes)
        plotter = plot_damage(mesh, damage, show=False)
        assert plotter is not None


class TestGoldakRender:
    def test_goldak_isosurface_renders(self) -> None:
        from feaweld.solver.thermal import GoldakHeatSource
        from feaweld.visualization.thermal_plots import render_goldak_source

        source = GoldakHeatSource(
            power=4000.0, a_f=5.0, a_r=10.0, b=4.0, c=3.0,
            travel_speed=5.0,
            start_position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "goldak.png")
            plotter = render_goldak_source(
                source, t=2.0, grid_points=18,
                show=False, screenshot=out,
            )
            assert plotter is not None
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0


class TestDamageAnimation:
    def test_damage_animation_gif_writes(self) -> None:
        matplotlib = pytest.importorskip("matplotlib")
        from feaweld.visualization.fatigue_plots import animate_damage_evolution
        from feaweld.core.types import SNCurve, SNSegment, SNStandard

        sn = SNCurve(
            name="TestFAT90",
            standard=SNStandard.IIW,
            segments=[SNSegment(m=3.0, C=90.0**3 * 2e6, stress_threshold=0.0)],
            cutoff_cycles=1e7,
        )
        blocks = [
            [(50.0, 0.0, 100.0)],
            [(80.0, 0.0, 50.0)],
            [(120.0, 0.0, 20.0)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "damage.gif")
            path = animate_damage_evolution(blocks, sn, out, fps=5)
            assert os.path.exists(str(path))
            assert os.path.getsize(str(path)) > 0


# ---------------------------------------------------------------------------
# Enhanced 3-D plotting (clipping / threshold / iso / vectors / weld / SED /
# preview / annotated).  All run off-screen with show=False; a plotter is
# returned and closed.  These build VTK filter pipelines and add meshes but do
# not render, so they need pyvista (module-level importorskip) but not a GL
# backend.
# ---------------------------------------------------------------------------


class TestEnhanced3D:

    def test_clipping_default_origin(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_stress_with_clipping(mesh, stress, show=False)
        assert plotter is not None
        plotter.close()

    def test_clipping_explicit_origin(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_stress_with_clipping(
            mesh, stress, clip_origin=(0.5, 0.5, 0.5), show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_threshold_above(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_stress_threshold(mesh, stress, above=True, show=False)
        assert plotter is not None
        plotter.close()

    def test_threshold_below(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_stress_threshold(mesh, stress, above=False, show=False)
        assert plotter is not None
        plotter.close()

    def test_threshold_explicit_value(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_stress_threshold(
            mesh, stress, threshold=100.0, above=True, show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_iso_single_value(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_iso_surface(mesh, stress, iso_values=[120.0], show=False)
        assert plotter is not None
        plotter.close()

    def test_iso_multiple_values(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_iso_surface(
            mesh, stress, iso_values=[80.0, 120.0, 160.0], show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_iso_default_values(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_iso_surface(mesh, stress, show=False)
        assert plotter is not None
        plotter.close()

    def test_iso_out_of_range_no_crash(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        # An iso-level well outside the field yields an empty contour that is
        # skipped rather than crashing.
        plotter = e3.plot_iso_surface(mesh, stress, iso_values=[1e9], show=False)
        assert plotter is not None
        plotter.close()

    def test_force_vectors_happy_path(self) -> None:
        mesh = _make_tet_mesh()
        vectors = np.ones((mesh.n_nodes, 3), dtype=np.float64)
        plotter = e3.plot_force_vectors(mesh, vectors, show=False)
        assert plotter is not None
        plotter.close()

    def test_force_vectors_shape_mismatch(self) -> None:
        mesh = _make_tet_mesh()
        bad = np.ones((mesh.n_nodes + 1, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="vectors must have shape"):
            e3.plot_force_vectors(mesh, bad, show=False)

    def test_weld_region_present(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_weld_region_highlight(
            mesh, stress, weld_region="weld", show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_weld_region_absent(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        # Unknown region name: annotated with a warning label, no crash.
        plotter = e3.plot_weld_region_highlight(
            mesh, stress, weld_region="does_not_exist", show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_sed_control_volume_no_result(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_sed_control_volume(
            mesh, (0.5, 0.5, 0.5), 0.4, stress=stress, show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_sed_control_volume_with_result_stub(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        sed_result = types.SimpleNamespace(
            sed_field=np.linspace(0.1, 0.5, mesh.n_nodes),
            averaged_sed=0.31,
        )
        plotter = e3.plot_sed_control_volume(
            mesh, (0.5, 0.5, 0.5), 0.4,
            sed_result=sed_result, stress=stress, show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_mesh_preview_with_sets(self) -> None:
        mesh = _make_tet_mesh()
        plotter = e3.plot_mesh_preview(
            mesh, highlight_sets={"weld": "tomato"}, show=False,
        )
        assert plotter is not None
        plotter.close()

    def test_mesh_preview_no_highlight(self) -> None:
        mesh = _make_tet_mesh()
        plotter = e3.plot_mesh_preview(mesh, show=False)
        assert plotter is not None
        plotter.close()

    def test_annotated_femesh_path(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        plotter = e3.plot_annotated_stress(mesh, stress, show=False)
        assert plotter is not None
        plotter.close()

    def test_annotated_grid_path(self) -> None:
        mesh = _make_tet_mesh()
        stress = _make_gradient_stress(mesh)
        grid = stress_field_to_pyvista(mesh, stress)
        plotter = e3.plot_annotated_stress(grid, show=False)
        assert plotter is not None
        plotter.close()


class TestGridAcceptance:
    """resolve_grid / resolve_component contract and grid-input passthrough."""

    def test_plot_stress_field_accepts_grid(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        grid = stress_field_to_pyvista(mesh, stress)
        plotter = plot_stress_field(grid, show=False)
        assert plotter is not None
        plotter.close()

    def test_resolve_grid_passthrough_identity(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        grid = stress_field_to_pyvista(mesh, stress)
        assert resolve_grid(grid) is grid

    def test_resolve_grid_builds_from_femesh(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        grid = resolve_grid(mesh, stress)
        assert grid.n_points == mesh.n_nodes
        assert "von_mises" in grid.point_data

    def test_resolve_component_known(self) -> None:
        assert resolve_component("von_mises") == "von_mises"
        assert resolve_component("xx") == "stress_xx"

    def test_resolve_component_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown component"):
            resolve_component("bogus")


@requires_render
class TestExportAndScreenshots:
    """Screenshot / export paths that require an actual off-screen GL backend."""

    def test_plotter_to_base64_png_magic(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        plotter = plot_stress_field(mesh, stress, show=False)
        b64 = plotter_to_base64(plotter)
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"

    def test_export_png_writes_nested(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        plotter = plot_stress_field(mesh, stress, show=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "shot.png")
            export_png(plotter, path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        plotter.close()

    def test_export_gltf_writes(self) -> None:
        mesh = _make_tri_mesh()
        stress = _make_stress(mesh.n_nodes)
        plotter = plot_stress_field(mesh, stress, show=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.gltf")
            export_gltf(plotter, path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        plotter.close()
