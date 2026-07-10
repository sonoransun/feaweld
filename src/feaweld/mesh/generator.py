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
from scipy.spatial import cKDTree

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
    dim: int | None = None,
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
        Mesh dimension (2 or 3).  With *None* (the default) the dimension
        is inferred from ``joint.dimension``; an explicit value that
        contradicts the joint raises ``ValueError``.
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

    joint_dim = int(getattr(joint, "dimension", 2))
    if dim is None:
        dim = joint_dim
    elif dim != joint_dim:
        raise ValueError(
            f"Requested mesh dim={dim} contradicts joint.dimension="
            f"{joint_dim}; build the joint with the matching 'dimension' "
            "(or omit 'dim' to infer it from the joint)."
        )

    if dim == 3 and config.element_type_3d == "hex":
        raise NotImplementedError(
            "hex meshing not yet supported for extruded joints; use 'tet'"
        )

    _ensure_gmsh_initialized()

    # 1. Build geometry
    joint.build(model_name=model_name)

    # 2. Size fields for weld-toe refinement.  Extruded joints refine
    #    around the swept toe lines rather than the section toe points.
    toe_points = joint.get_weld_toe_points()
    toe_lines = None
    if dim == 3:
        get_lines = getattr(joint, "get_weld_toe_lines", None)
        if get_lines is not None:
            toe_lines = get_lines()
    _apply_size_fields(toe_points, config, toe_lines=toe_lines)

    # 3. Global meshing options
    gmsh.option.setNumber("Mesh.Algorithm", config.algorithm_2d)
    if dim == 3:
        gmsh.option.setNumber("Mesh.Algorithm3D", config.algorithm_3d)

    gmsh.option.setNumber("Mesh.ElementOrder", config.element_order)

    if dim == 2 and config.element_type_2d == "quad":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    else:
        # No quad recombination on the boundary faces of a tet mesh.
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

    # 6. Attach the weld-toe node set.  Gmsh physical groups only tag the
    #    bottom/top boundaries, so the toe nodes needed by post-processing are
    #    recovered here from the joint's analytic toe coordinates.
    _attach_weld_toe_node_set(mesh, toe_points)

    if finalize:
        gmsh.finalize()

    return mesh


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

def extract_mesh_from_gmsh(dim: int = 2) -> FEMesh:
    """Read the current Gmsh model mesh into an [FEMesh][feaweld.core.types.FEMesh].

    Must be called while a Gmsh session is active and a mesh has been
    generated.

    Notes
    -----
    Nodes not referenced by any extracted element are dropped and the
    connectivity and node sets are remapped to the compacted numbering.
    The free OCC points (2D) and lines (3D) created for the weld-toe
    ``Distance`` size fields are meshed by Gmsh into nodes that belong to
    no solid element — along 3D toe lines they exactly duplicate every toe
    node.  A real solver assigns those orphans zero stress, which would
    silently corrupt every coordinate-based nearest-node sampler at the
    weld toe.
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

    # Sorted view of the element tags so group tags can be mapped to element
    # indices with searchsorted instead of a per-tag linear scan.
    tag_order = np.argsort(n_elem_tags).astype(np.int64)
    sorted_elem_tags = n_elem_tags[tag_order]

    for d in range(dim + 1):
        phys = gmsh.model.getPhysicalGroups(d)
        for pdim, ptag in phys:
            name = gmsh.model.getPhysicalName(pdim, ptag)
            if not name:
                name = f"group_{pdim}d_{ptag}"

            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(pdim, ptag)

            if pdim == dim:
                # Collect element indices belonging to this physical group
                group_tags: list[NDArray[np.int64]] = []
                for et in ent_tags:
                    e_types, e_tags, _ = gmsh.model.mesh.getElements(pdim, et)
                    for j, etype in enumerate(e_types):
                        if int(etype) == gmsh_etype:
                            group_tags.append(
                                np.asarray(e_tags[j], dtype=np.int64)
                            )
                if group_tags:
                    tags = np.concatenate(group_tags)
                    pos = np.searchsorted(sorted_elem_tags, tags)
                    pos = np.minimum(pos, len(sorted_elem_tags) - 1)
                    # Drop group tags absent from n_elem_tags (matches the
                    # empty np.where result of the per-tag scan).
                    valid = sorted_elem_tags[pos] == tags
                    if np.any(valid):
                        physical_groups[name] = np.unique(
                            tag_order[pos[valid]]
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

    # -- Drop orphan nodes ----------------------------------------------------
    # Only nodes referenced by the extracted connectivity are kept (element
    # indices in *physical_groups* are unaffected).  np.unique returns the
    # used indices sorted, so the relative node order is preserved and the
    # remapping below is monotone — already-sorted node sets stay sorted.
    used = np.unique(connectivity)
    if used.size < n_nodes:
        remap = np.full(n_nodes, -1, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        coords = coords[used]
        connectivity = remap[connectivity]
        filtered_sets: dict[str, NDArray[np.int64]] = {}
        for name, ids in node_sets.items():
            mapped = remap[ids]
            mapped = mapped[mapped >= 0]
            if mapped.size:
                filtered_sets[name] = mapped
        node_sets = filtered_sets

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

def _attach_weld_toe_node_set(
    mesh: FEMesh,
    toe_points: list[tuple[float, float, float]],
) -> None:
    """Map a joint's weld-toe coordinates to their nearest mesh nodes.

    Post-processing methods locate the weld toe through the ``"weld_toe"``
    node set.  For 2D sections Gmsh physical groups only tag
    ``bottom``/``top`` boundaries, so each analytic toe point is matched to
    its nearest mesh node via a k-d tree; the resulting node indices are
    de-duplicated and stored as an ``int64`` array on
    ``mesh.node_sets["weld_toe"]``.  Extruded 3D joints register toe-edge
    physical groups directly — a ``"weld_toe"`` set already extracted from
    those groups is kept as-is.

    Parameters
    ----------
    mesh : FEMesh
        The extracted mesh, modified in place.
    toe_points : list of tuple of float
        Weld-toe coordinates from ``JointGeometry.get_weld_toe_points()``.
        When empty (e.g. some butt configurations) the set is left absent.
    """
    if "weld_toe" in mesh.node_sets:
        return

    if not toe_points:
        return

    pts = np.asarray(toe_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return

    # Pad/truncate the toe coordinates to the mesh node dimensionality so the
    # k-d tree query is well-posed for both 2-D and 3-D meshes.
    ncol = mesh.nodes.shape[1]
    if pts.shape[1] < ncol:
        pts = np.hstack([pts, np.zeros((pts.shape[0], ncol - pts.shape[1]))])
    elif pts.shape[1] > ncol:
        pts = pts[:, :ncol]

    _, idx = cKDTree(mesh.nodes).query(pts)
    mesh.node_sets["weld_toe"] = np.unique(np.asarray(idx, dtype=np.int64))


def _apply_size_fields(
    toe_points: list[tuple[float, float, float]],
    config: WeldMeshConfig,
    toe_lines: list[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] | None = None,
) -> None:
    """Set up Gmsh mesh size fields for weld-toe refinement.

    In 2D each analytic toe point becomes a free OCC point feeding a
    ``Distance`` field.  For extruded 3D joints *toe_lines* provides the
    swept toe segments instead: each becomes a free OCC line whose
    ``Distance`` field is sampled finely enough to resolve
    ``weld_toe_size`` along its length.  The Threshold/Min combination is
    identical for both source kinds.
    """
    if toe_lines:
        dist_ids = _distance_fields_from_lines(toe_lines, config)
    elif toe_points:
        dist_ids = _distance_fields_from_points(toe_points)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.global_size)
        return

    field_ids: list[int] = []

    for dist_id in dist_ids:
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


def _distance_fields_from_points(
    toe_points: list[tuple[float, float, float]],
) -> list[int]:
    """One ``Distance`` field per analytic toe point (free OCC points)."""
    dist_ids: list[int] = []
    for px, py, pz in toe_points:
        # Point source for Distance field
        pt_tag = gmsh.model.occ.addPoint(px, py, pz)
        gmsh.model.occ.synchronize()

        dist_id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [pt_tag])
        dist_ids.append(dist_id)
    return dist_ids


def _distance_fields_from_lines(
    toe_lines: list[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    config: WeldMeshConfig,
) -> list[int]:
    """One ``Distance`` field per swept toe segment (free OCC lines).

    Each segment is rebuilt as a free OCC line so the field does not
    depend on the extruded model's edge tags; the field sampling density
    scales with segment length so the refinement band stays tight at
    ``weld_toe_size`` resolution.
    """
    curves: list[tuple[int, float]] = []
    for p0, p1 in toe_lines:
        t0 = gmsh.model.occ.addPoint(*p0)
        t1 = gmsh.model.occ.addPoint(*p1)
        curve = gmsh.model.occ.addLine(t0, t1)
        length = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        curves.append((curve, length))
    gmsh.model.occ.synchronize()

    dist_ids: list[int] = []
    for curve, length in curves:
        dist_id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", [curve])
        gmsh.model.mesh.field.setNumber(
            dist_id, "Sampling", max(20, int(length / config.weld_toe_size))
        )
        dist_ids.append(dist_id)
    return dist_ids
