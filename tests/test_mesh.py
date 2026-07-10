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


def _make_3d_joint(length: float = 30.0) -> FilletTJoint:
    return FilletTJoint(
        base_width=60.0,
        base_thickness=10.0,
        web_height=30.0,
        web_thickness=8.0,
        weld_leg_size=6.0,
        length=length,
        dimension=3,
    )


def _quick_3d_config(**over) -> WeldMeshConfig:
    """Very coarse 3D mesh config for fast tests."""
    kw = dict(
        global_size=9.0,
        weld_toe_size=2.5,
        weld_region_size=4.0,
        refinement_distance=8.0,
        element_order=1,
        element_type_2d="tri",
    )
    kw.update(over)
    return WeldMeshConfig(**kw)


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

    def test_weld_toe_node_set_populated(self):
        joint = _make_joint()
        cfg = _quick_config()
        mesh = generate_mesh(joint, cfg)

        assert "weld_toe" in mesh.node_sets
        toe_nodes = mesh.node_sets["weld_toe"]
        assert toe_nodes.dtype == np.int64
        assert len(toe_nodes) > 0
        # No duplicates.
        assert len(np.unique(toe_nodes)) == len(toe_nodes)

        # Every analytic toe point has a set node within one toe-element size.
        toe_pts = np.array(joint.get_weld_toe_points())
        toe_coords = mesh.nodes[toe_nodes]
        for p in toe_pts:
            nearest = np.linalg.norm(toe_coords - p, axis=1).min()
            assert nearest <= cfg.weld_toe_size

    def test_quadratic_elements(self):
        cfg = _quick_config()
        cfg.element_order = 2
        mesh = generate_mesh(_make_joint(), cfg)
        assert mesh.element_type in (ElementType.TRI6, ElementType.QUAD8)


# ---------------------------------------------------------------------------
# 3D mesh generation (extruded joints)
# ---------------------------------------------------------------------------

@requires_gmsh
class TestGenerateMesh3D:
    def test_produces_tet4_mesh(self):
        mesh = generate_mesh(_make_3d_joint(), _quick_3d_config())
        assert mesh.ndim == 3
        assert mesh.element_type == ElementType.TET4
        assert mesh.n_elements > 0
        # The extrusion spans the full weld length along z.
        assert np.ptp(mesh.nodes[:, 2]) == pytest.approx(30.0)

    def test_node_sets_from_physical_groups(self):
        joint = _make_3d_joint()
        mesh = generate_mesh(joint, _quick_3d_config())
        n_toes = len(joint.get_weld_toe_points())
        expected = {"bottom", "top", "weld_toe"} | {
            f"weld_toe_{i}" for i in range(n_toes)
        }
        assert expected <= set(mesh.node_sets)
        for name in expected:
            assert len(mesh.node_sets[name]) > 0

    def test_toe_nodes_on_analytic_lines(self):
        joint = _make_3d_joint()
        mesh = generate_mesh(joint, _quick_3d_config())
        for i, ((x0, y0, z0), (_x1, _y1, z1)) in enumerate(
                joint.get_weld_toe_lines()):
            coords = mesh.nodes[mesh.node_sets[f"weld_toe_{i}"]]
            np.testing.assert_allclose(coords[:, 0], x0, atol=1e-6)
            np.testing.assert_allclose(coords[:, 1], y0, atol=1e-6)
            assert coords[:, 2].min() >= z0 - 1e-6
            assert coords[:, 2].max() <= z1 + 1e-6

    def test_dim_joint_mismatch_raises(self):
        with pytest.raises(ValueError, match="dim"):
            generate_mesh(_make_3d_joint(), _quick_3d_config(), dim=2)
        with pytest.raises(ValueError, match="dim"):
            generate_mesh(_make_joint(), _quick_config(), dim=3)

    def test_hex_raises_not_implemented(self):
        cfg = _quick_3d_config(element_type_3d="hex")
        with pytest.raises(NotImplementedError, match="hex"):
            generate_mesh(_make_3d_joint(), cfg)

    def test_second_order_gives_tet10(self):
        cfg = _quick_3d_config(element_order=2, optimize=False)
        mesh = generate_mesh(_make_3d_joint(), cfg)
        assert mesh.element_type == ElementType.TET10
        assert mesh.elements.shape[1] == 10

    def test_quality_report_on_tets(self):
        mesh = generate_mesh(_make_3d_joint(), _quick_3d_config())
        report = mesh_quality_report(mesh)
        assert report["aspect_ratio"]["min"] > 0
        assert report["aspect_ratio"]["max"] >= report["aspect_ratio"]["min"]
        assert report["jacobian"]["min"] >= 0.0
        assert "n_poor_elements" in report


# ---------------------------------------------------------------------------
# Orphan-node filtering (size-field construction side effects)
# ---------------------------------------------------------------------------

def _assert_no_orphans_or_duplicates(mesh: FEMesh) -> None:
    """Every node is referenced by an element; no coincident node pairs."""
    from scipy.spatial import cKDTree

    used = np.unique(mesh.elements)
    np.testing.assert_array_equal(
        used, np.arange(mesh.n_nodes),
        err_msg="mesh contains nodes referenced by no element",
    )
    pairs = cKDTree(mesh.nodes).query_pairs(1e-9)
    assert not pairs, f"{len(pairs)} coincident node pairs in the mesh"
    # Node sets must only reference surviving (element-attached) nodes.
    for name, ids in mesh.node_sets.items():
        assert ids.min() >= 0 and ids.max() < mesh.n_nodes, name


@requires_gmsh
class TestOrphanNodeFiltering:
    """The free OCC points (2D) and lines (3D) feeding the weld-toe
    ``Distance`` size fields are meshed by Gmsh into nodes attached to no
    element — exact coordinate twins of the real toe nodes.  A real solver
    would give those orphans zero stress, corrupting every nearest-node
    stress sampler at the weld toe, so extraction must drop them."""

    def test_3d_mesh_has_no_orphan_or_duplicate_nodes(self):
        joint = _make_3d_joint()
        mesh = generate_mesh(joint, _quick_3d_config())
        _assert_no_orphans_or_duplicates(mesh)
        # Specifically: no zero-stress twins along the swept toe lines.
        from scipy.spatial import cKDTree
        for i in range(len(joint.get_weld_toe_lines())):
            toe_coords = mesh.nodes[mesh.node_sets[f"weld_toe_{i}"]]
            assert not cKDTree(toe_coords).query_pairs(1e-9)

    def test_2d_mesh_has_no_orphan_or_duplicate_nodes(self):
        mesh = generate_mesh(_make_joint(), _quick_config())
        _assert_no_orphans_or_duplicates(mesh)

    def test_2d_weld_toe_fallback_lands_on_real_nodes(self):
        """The kd-tree fallback must resolve each analytic toe point to the
        element-attached node at that geometry vertex, not an orphan twin."""
        from scipy.spatial import cKDTree

        joint = _make_joint()
        mesh = generate_mesh(joint, _quick_config())
        toe_pts = np.asarray(joint.get_weld_toe_points())
        dist, idx = cKDTree(mesh.nodes).query(toe_pts)
        # Toe points are geometry vertices, so a real node sits exactly there.
        assert np.all(dist < 1e-9)
        assert set(idx.tolist()) <= set(mesh.node_sets["weld_toe"].tolist())


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
