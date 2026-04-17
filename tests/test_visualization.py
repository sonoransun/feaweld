"""Tests for the visualization and export modules.

PyVista-dependent tests are skipped when pyvista is not installed.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField

# Guard: skip pyvista-dependent tests when the library is not available.
pyvista = pytest.importorskip("pyvista")

from feaweld.visualization.export import export_vtk
from feaweld.visualization.fatigue_maps import plot_damage, plot_fatigue_life
from feaweld.visualization.stress_plots import (
    plot_deformed,
    plot_stress_field,
    plot_temperature_field,
    stress_field_to_pyvista,
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
