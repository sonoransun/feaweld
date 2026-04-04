"""Tests for feaweld.mesh — mesh generation, quality, and conversion."""

from __future__ import annotations

import numpy as np
import pytest

# Guard: skip if gmsh is not available
gmsh = pytest.importorskip("gmsh")

from feaweld.core.types import ElementType, FEMesh
from feaweld.geometry.joints import FilletTJoint
from feaweld.mesh.generator import WeldMeshConfig, generate_mesh
from feaweld.mesh.quality import aspect_ratio, jacobian_quality, mesh_quality_report

requires_gmsh = pytest.mark.requires_gmsh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_joint() -> FilletTJoint:
    return FilletTJoint(
        base_width=100.0,
        base_thickness=10.0,
        web_height=50.0,
        web_thickness=10.0,
        weld_leg_size=6.0,
    )


def _quick_config() -> WeldMeshConfig:
    """Coarse mesh config for fast tests."""
    return WeldMeshConfig(
        global_size=5.0,
        weld_toe_size=1.0,
        weld_region_size=2.0,
        refinement_distance=10.0,
        element_order=1,
        element_type_2d="tri",
    )


@pytest.fixture(autouse=True)
def _gmsh_cleanup():
    """Ensure gmsh is finalized after each test."""
    if gmsh.is_initialized():
        gmsh.finalize()
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

@requires_gmsh
class TestGenerateMesh:
    def test_produces_valid_femesh(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        assert isinstance(mesh, FEMesh)
        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_correct_dimension(self):
        mesh = generate_mesh(_make_joint(), _quick_config(), dim=2)
        # All nodes should have a z-coordinate of 0 for a 2D cross-section
        assert mesh.nodes.shape[1] == 3  # Gmsh always returns 3D coords

    def test_element_type_tri(self):
        cfg = _quick_config()
        cfg.element_type_2d = "tri"
        mesh = generate_mesh(_make_joint(), cfg)
        assert mesh.element_type in (ElementType.TRI3, ElementType.TRI6)

    def test_physical_groups_propagated(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        # At least some physical groups should have been transferred
        # (either as element sets or node sets)
        has_groups = len(mesh.physical_groups) > 0 or len(mesh.node_sets) > 0
        assert has_groups, "No physical groups or node sets in the mesh"

    def test_quadratic_elements(self):
        cfg = _quick_config()
        cfg.element_order = 2
        mesh = generate_mesh(_make_joint(), cfg)
        assert mesh.element_type in (ElementType.TRI6, ElementType.QUAD8)


# ---------------------------------------------------------------------------
# Mesh quality
# ---------------------------------------------------------------------------

@requires_gmsh
class TestMeshQuality:
    def test_aspect_ratio_computed(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        ar = aspect_ratio(mesh)
        assert ar.shape == (mesh.n_elements,)
        assert np.all(ar > 0)

    def test_jacobian_quality_computed(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        jq = jacobian_quality(mesh)
        assert jq.shape == (mesh.n_elements,)
        # Most elements should have reasonable quality
        assert np.mean(jq > 0.1) > 0.5

    def test_quality_report(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        report = mesh_quality_report(mesh)
        assert "aspect_ratio" in report
        assert "jacobian" in report
        assert "n_poor_elements" in report
        assert "poor_element_indices" in report
        assert report["aspect_ratio"]["min"] > 0
        assert report["aspect_ratio"]["max"] >= report["aspect_ratio"]["min"]


# ---------------------------------------------------------------------------
# Mesh conversion (meshio round-trip)
# ---------------------------------------------------------------------------

@requires_gmsh
class TestMeshConversion:
    def test_meshio_roundtrip(self):
        from feaweld.mesh.convert import femesh_to_meshio, meshio_to_femesh

        original = generate_mesh(_make_joint(), _quick_config())
        mio = femesh_to_meshio(original)

        # Check meshio object
        assert len(mio.cells) == 1
        assert mio.points.shape[0] == original.n_nodes

        # Convert back
        recovered = meshio_to_femesh(mio)
        assert recovered.n_nodes == original.n_nodes
        assert recovered.n_elements == original.n_elements
        np.testing.assert_allclose(recovered.nodes, original.nodes)
        np.testing.assert_array_equal(recovered.elements, original.elements)

    def test_vtk_export(self, tmp_path):
        from feaweld.mesh.convert import femesh_to_vtk

        mesh = generate_mesh(_make_joint(), _quick_config())
        vtk_file = str(tmp_path / "test_mesh.vtu")
        femesh_to_vtk(mesh, vtk_file)

        import pathlib
        assert pathlib.Path(vtk_file).exists()
        assert pathlib.Path(vtk_file).stat().st_size > 0
