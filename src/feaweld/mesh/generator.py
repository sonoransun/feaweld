"""Mesh generation for weld joint geometries using the Gmsh Python API.

This module provides configurable mesh generation with automatic refinement
near weld toes, which is critical for accurate stress concentration
calculations in fatigue assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import ElementType, FEMesh
from feaweld.geometry.joints import JointGeometry, _ensure_gmsh_initialized


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WeldMeshConfig:
    """Meshing parameters for weld-joint models.

    Attributes
    ----------
    global_size : float
        Background mesh element size (mm).
    weld_toe_size : float
        Target element size at weld toe locations (mm).
    weld_region_size : float
        Target element size inside the weld region (mm).
    refinement_distance : float
        Distance (mm) over which the size field transitions from
        *weld_toe_size* to *global_size*.
    element_order : int
        1 for linear, 2 for quadratic elements.
    element_type_2d : str
        ``"tri"`` for triangles, ``"quad"`` for quadrilaterals.
    element_type_3d : str
        ``"tet"`` for tetrahedra, ``"hex"`` for hexahedra.
    algorithm_2d : int
        Gmsh 2-D meshing algorithm index (6 = Frontal-Delaunay).
    algorithm_3d : int
        Gmsh 3-D meshing algorithm index (1 = Delaunay).
    optimize : bool
        Run Gmsh mesh optimisation passes after generation.
    """

    global_size: float = 2.0
    weld_toe_size: float = 0.2
    weld_region_size: float = 0.5
    refinement_distance: float = 5.0
    element_order: int = 2
    element_type_2d: str = "tri"
    element_type_3d: str = "tet"
    algorithm_2d: int = 6
    algorithm_3d: int = 1
    optimize: bool = True


# ---------------------------------------------------------------------------
# Gmsh element-type code to our ElementType enum
# ---------------------------------------------------------------------------

_GMSH_TO_ELEMENT_TYPE: dict[int, ElementType] = {
    2: ElementType.TRI3,
    3: ElementType.QUAD4,
    4: ElementType.TET4,
    5: ElementType.HEX8,
    9: ElementType.TRI6,
    10: ElementType.QUAD8,
    11: ElementType.TET10,
    17: ElementType.HEX20,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_mesh(
    joint: JointGeometry,
    config: WeldMeshConfig | None = None,
    model_name: str = "weld_joint",
    dim: int = 2,
    finalize: bool = True,
) -> FEMesh:
    """Generate a finite-element mesh for *joint*.

    Parameters
    ----------
    joint:
        Joint geometry instance.  ``build()`` will be called internally.
    config:
        Meshing configuration.  If *None*, defaults are used.
    model_name:
        Gmsh model name.
    dim:
        Mesh dimension (2 or 3).
    finalize:
        If *True*, call ``gmsh.finalize()`` when done.  Set to *False*
        if you want to keep Gmsh alive for further operations (e.g.
        visualisation).

    Returns
    -------
    FEMesh
        Solver-agnostic mesh data structure.
    """
    if config is None:
        config = WeldMeshConfig()

    _ensure_gmsh_initialized()

    # 1. Build geometry
    joint.build(model_name=model_name)

    # 2. Size fields for weld-toe refinement
    toe_points = joint.get_weld_toe_points()
    _apply_size_fields(toe_points, config)

    # 3. Global meshing options
    gmsh.option.setNumber("Mesh.Algorithm", config.algorithm_2d)
    if dim == 3:
        gmsh.option.setNumber("Mesh.Algorithm3D", config.algorithm_3d)

    gmsh.option.setNumber("Mesh.ElementOrder", config.element_order)

    if config.element_type_2d == "quad":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    else:
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # 4. Generate
    gmsh.model.mesh.generate(dim)

    if config.optimize:
        if config.element_order == 1:
            gmsh.model.mesh.optimize("Laplace2D" if dim == 2 else "")
        else:
            gmsh.model.mesh.optimize("HighOrder")

    # 5. Extract into FEMesh
    mesh = extract_mesh_from_gmsh(dim=dim)

    if finalize:
        gmsh.finalize()

    return mesh


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

def extract_mesh_from_gmsh(dim: int = 2) -> FEMesh:
    """Read the current Gmsh model mesh into an :class:`FEMesh`.

    Must be called while a Gmsh session is active and a mesh has been
    generated.
    """
    # -- Nodes ---------------------------------------------------------------
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_coords is flat [x1,y1,z1, x2,y2,z2, ...]
    n_nodes = len(node_tags)
    coords = np.array(node_coords, dtype=np.float64).reshape(n_nodes, 3)

    # Build a mapping from Gmsh tag -> 0-based index
    tag_to_idx = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    for i, t in enumerate(node_tags):
        tag_to_idx[int(t)] = i

    # -- Elements ------------------------------------------------------------
    # We collect the dominant element type for the requested dimension.
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim)

    if len(elem_types) == 0:
        raise RuntimeError(
            f"No {dim}-D elements found in the Gmsh model. "
            "Did mesh generation succeed?"
        )

    # Use the first (most common) element type at this dimension
    primary_type_idx = 0
    gmsh_etype = int(elem_types[primary_type_idx])
    raw_conn = np.array(elem_node_tags[primary_type_idx], dtype=np.int64)
    n_elem_tags = np.array(elem_tags[primary_type_idx], dtype=np.int64)

    _, _, _, n_nodes_per_elem, _, _ = gmsh.model.mesh.getElementProperties(
        gmsh_etype
    )
    n_elems = len(n_elem_tags)
    connectivity = tag_to_idx[raw_conn].reshape(n_elems, n_nodes_per_elem)

    element_type = _GMSH_TO_ELEMENT_TYPE.get(gmsh_etype, ElementType.TRI3)

    # -- Physical groups → element sets --------------------------------------
    physical_groups: dict[str, NDArray[np.int64]] = {}
    node_sets: dict[str, NDArray[np.int64]] = {}

    for d in range(dim + 1):
        phys = gmsh.model.getPhysicalGroups(d)
        for pdim, ptag in phys:
            name = gmsh.model.getPhysicalName(pdim, ptag)
            if not name:
                name = f"group_{pdim}d_{ptag}"

            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(pdim, ptag)

            if pdim == dim:
                # Collect element indices belonging to this physical group
                elem_indices: list[int] = []
                for et in ent_tags:
                    e_types, e_tags, _ = gmsh.model.mesh.getElements(pdim, et)
                    for j, etype in enumerate(e_types):
                        if int(etype) == gmsh_etype:
                            for etag in e_tags[j]:
                                # Find index of this element tag in n_elem_tags
                                idx_arr = np.where(n_elem_tags == int(etag))[0]
                                elem_indices.extend(idx_arr.tolist())
                if elem_indices:
                    physical_groups[name] = np.array(
                        sorted(set(elem_indices)), dtype=np.int64
                    )
            else:
                # Lower-dimensional groups -> node sets
                ns: list[int] = []
                for et in ent_tags:
                    ntags, _, _ = gmsh.model.mesh.getNodes(pdim, et, True)
                    ns.extend(tag_to_idx[int(t)] for t in ntags)
                if ns:
                    node_sets[name] = np.array(
                        sorted(set(ns)), dtype=np.int64
                    )

    return FEMesh(
        nodes=coords,
        elements=connectivity,
        element_type=element_type,
        physical_groups=physical_groups,
        node_sets=node_sets,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_size_fields(
    toe_points: list[tuple[float, float, float]],
    config: WeldMeshConfig,
) -> None:
    """Set up Gmsh mesh size fields for weld-toe refinement."""
    if not toe_points:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.global_size)
        return

    field_ids: list[int] = []

    for px, py, pz in toe_points:
        # Point source for Distance field
        pt_tag = gmsh.model.occ.addPoint(px, py, pz)
        gmsh.model.occ.synchronize()

        # Distance field
        dist_id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [pt_tag])

        # Threshold field
        thresh_id = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_id, "InField", dist_id)
        gmsh.model.mesh.field.setNumber(thresh_id, "SizeMin", config.weld_toe_size)
        gmsh.model.mesh.field.setNumber(thresh_id, "SizeMax", config.global_size)
        gmsh.model.mesh.field.setNumber(thresh_id, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh_id, "DistMax", config.refinement_distance)

        field_ids.append(thresh_id)

    # Combine all threshold fields with Min
    if len(field_ids) == 1:
        bg_id = field_ids[0]
    else:
        bg_id = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(bg_id, "FieldsList", field_ids)

    gmsh.model.mesh.field.setAsBackgroundMesh(bg_id)

    # Disable default size constraints so background field takes over
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
