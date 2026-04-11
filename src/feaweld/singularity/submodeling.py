"""Global-local submodeling with boundary condition transfer.

Submodeling (also known as the cut-boundary displacement method) takes
displacement results from a coarse global model and uses them as boundary
conditions on a refined local model.  This removes the need to globally
refine the mesh, focusing computational effort on the region of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SubmodelSpec:
    """Specification for a submodel region."""

    center: NDArray  # (3,) centre of the submodel sphere/box
    radius: float  # radius defining the extraction region
    mesh_size: float  # target element size in the submodel
    parent_results: FEAResults


# ---------------------------------------------------------------------------
# Boundary displacement extraction
# ---------------------------------------------------------------------------

def extract_boundary_displacements(
    parent_results: FEAResults,
    boundary_nodes: NDArray[np.int64],
) -> NDArray:
    """Interpolate displacement from the parent (global) model at submodel boundary nodes.

    For each boundary node the nearest parent node is found and its
    displacement is returned.  When *scipy* is available a Radial Basis
    Function (RBF) interpolation is used for smoother results; otherwise
    nearest-neighbour transfer is employed.

    Parameters
    ----------
    parent_results:
        Solved parent (global) model results that include displacements.
    boundary_nodes:
        Coordinates of the boundary nodes in the submodel, shape ``(n, 3)``.

    Returns
    -------
    NDArray
        Displacement vectors at the boundary nodes, shape ``(n, 3)``.
    """
    if parent_results.displacement is None:
        raise ValueError("Parent results must contain displacement data.")

    parent_coords = parent_results.mesh.nodes  # (n_p, 3)
    parent_disp = parent_results.displacement  # (n_p, 3)

    # Ensure boundary_nodes is 2-D
    boundary_nodes = np.asarray(boundary_nodes, dtype=np.float64)
    if boundary_nodes.ndim == 1:
        boundary_nodes = boundary_nodes.reshape(1, -1)

    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore[import-untyped]

        # Build one RBF interpolator for all three displacement components.
        rbf = RBFInterpolator(parent_coords, parent_disp, kernel="thin_plate_spline")
        return np.asarray(rbf(boundary_nodes), dtype=np.float64)
    except ImportError:
        # Fall back to nearest-neighbour transfer.
        return _nearest_neighbour_transfer(
            parent_coords, parent_disp, boundary_nodes
        )


# ---------------------------------------------------------------------------
# Submodel region extraction
# ---------------------------------------------------------------------------

def create_submodel_region(
    parent_mesh: FEMesh,
    center: NDArray,
    radius: float,
) -> tuple[NDArray, NDArray]:
    """Identify elements and nodes within *radius* of *center*.

    Parameters
    ----------
    parent_mesh:
        The global FE mesh.
    center:
        Centre point of the submodel region, shape ``(3,)``.
    radius:
        Inclusion radius.

    Returns
    -------
    (element_ids, node_ids)
        Integer arrays of element and node indices that lie within (or
        partially within) the extraction sphere.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    dist = np.linalg.norm(parent_mesh.nodes - center[np.newaxis, :], axis=1)
    node_mask = dist <= radius
    node_ids = np.nonzero(node_mask)[0].astype(np.int64)

    # An element is included if *any* of its nodes falls within the sphere.
    node_set = set(node_ids.tolist())
    elem_ids: list[int] = []
    for eid in range(parent_mesh.n_elements):
        conn = parent_mesh.elements[eid]
        if any(int(nid) in node_set for nid in conn):
            elem_ids.append(eid)

    return np.array(elem_ids, dtype=np.int64), node_ids


# ---------------------------------------------------------------------------
# Submodel solver
# ---------------------------------------------------------------------------

class SubmodelSolver:
    """Build, constrain, and solve a refined submodel.

    Parameters
    ----------
    parent_results:
        Solved global model.
    center:
        Centre of the submodel region.
    radius:
        Extraction radius.
    refinement_factor:
        How much finer the submodel mesh is compared to the parent.
    """

    def __init__(
        self,
        parent_results: FEAResults,
        center: NDArray,
        radius: float,
        refinement_factor: int = 4,
    ) -> None:
        self.parent_results = parent_results
        self.center = np.asarray(center, dtype=np.float64).ravel()
        self.radius = float(radius)
        self.refinement_factor = int(refinement_factor)

        # Derived quantities
        self._parent_h = _characteristic_element_size(parent_results.mesh)
        self._sub_h = self._parent_h / self.refinement_factor

        # Region from the parent mesh
        self._elem_ids, self._node_ids = create_submodel_region(
            parent_results.mesh, self.center, self.radius
        )

        self._submodel_mesh: FEMesh | None = None
        self._boundary_node_ids: NDArray | None = None

    # -- public interface ---------------------------------------------------

    def create_submodel_mesh(self) -> FEMesh:
        """Generate a refined mesh inside the submodel region.

        The mesh is constructed by subdividing elements from the parent
        model that fall within the extraction sphere.  Each parent
        element edge is divided by *refinement_factor*.

        Returns
        -------
        FEMesh
            The refined local mesh.
        """
        parent = self.parent_results.mesh

        # Collect unique node ids from the selected elements.
        sub_elems = parent.elements[self._elem_ids]
        unique_node_ids = np.unique(sub_elems)

        # Build a mapping from global node IDs to local (0-based) IDs.
        global_to_local = {int(g): l for l, g in enumerate(unique_node_ids)}

        sub_nodes = parent.nodes[unique_node_ids]
        sub_conn = np.vectorize(global_to_local.get)(sub_elems).astype(np.int64)

        # Refine by subdividing edges.
        refined_nodes, refined_conn = _refine_mesh(
            sub_nodes, sub_conn, self.refinement_factor
        )

        # Identify boundary nodes (those on the outer sphere).
        dist = np.linalg.norm(refined_nodes - self.center[np.newaxis, :], axis=1)
        tol = self._sub_h * 0.5
        boundary_mask = dist >= (self.radius - tol)
        self._boundary_node_ids = np.nonzero(boundary_mask)[0].astype(np.int64)

        self._submodel_mesh = FEMesh(
            nodes=refined_nodes,
            elements=refined_conn,
            element_type=parent.element_type,
            node_sets={"boundary": self._boundary_node_ids},
        )
        return self._submodel_mesh

    def apply_boundary_conditions(self) -> LoadCase:
        """Create displacement BCs on the submodel boundary from the parent model.

        Returns
        -------
        LoadCase
            A :class:`LoadCase` with displacement boundary conditions.
        """
        if self._submodel_mesh is None:
            self.create_submodel_mesh()
        assert self._submodel_mesh is not None
        assert self._boundary_node_ids is not None

        # Coordinates of boundary nodes in the submodel
        boundary_coords = self._submodel_mesh.nodes[self._boundary_node_ids]

        # Interpolate displacements from parent
        bc_displacements = extract_boundary_displacements(
            self.parent_results, boundary_coords
        )

        bc = BoundaryCondition(
            node_set="boundary",
            bc_type=LoadType.DISPLACEMENT,
            values=bc_displacements,
        )

        return LoadCase(
            name="submodel_bcs",
            constraints=[bc],
        )

    def solve(
        self,
        phase_field: bool = False,
        phase_field_config: "object | None" = None,
        phase_field_material: "object | None" = None,
    ) -> FEAResults:
        """Solve the submodel.

        This is a simplified linear-elastic solve.  For production use,
        this would delegate to the configured solver backend.

        Parameters
        ----------
        phase_field:
            If True, delegate to the variational phase-field fracture
            solver (Track A4) using the current submodel mesh.  The
            parent result's material is used unless ``phase_field_material``
            is provided.  A :class:`feaweld.fracture.FractureResult` is
            packaged into an :class:`FEAResults` and returned.
        phase_field_config:
            Optional :class:`feaweld.fracture.PhaseFieldConfig`; defaults
            to the dataclass defaults when omitted.
        phase_field_material:
            Optional :class:`feaweld.core.materials.Material` override for
            the phase-field solve.

        Returns
        -------
        FEAResults
            Results on the refined submodel mesh.
        """
        if self._submodel_mesh is None:
            self.create_submodel_mesh()
        assert self._submodel_mesh is not None

        load_case = self.apply_boundary_conditions()

        if phase_field:
            from feaweld.fracture import PhaseFieldConfig, solve_phase_field

            cfg = phase_field_config or PhaseFieldConfig()
            mat = phase_field_material or self.parent_results.metadata.get(
                "material"
            )
            if mat is None:
                raise ValueError(
                    "phase_field=True requires either phase_field_material "
                    "or parent_results.metadata['material'] to be set."
                )
            fr = solve_phase_field(self._submodel_mesh, mat, load_case, cfg)

            u_full = np.zeros((self._submodel_mesh.n_nodes, 3))
            if fr.displacement.shape[1] == 2:
                u_full[:, :2] = fr.displacement
            else:
                u_full[:, :] = fr.displacement
            return FEAResults(
                mesh=self._submodel_mesh,
                displacement=u_full,
                metadata={
                    "submodel": True,
                    "phase_field": True,
                    "fracture_result": fr,
                    "parent_mesh_size": self._parent_h,
                },
            )

        # Placeholder solve: interpolate parent stress onto submodel nodes.
        # A full solve would assemble and factor the stiffness matrix.
        sub_disp = extract_boundary_displacements(
            self.parent_results,
            self._submodel_mesh.nodes,
        )

        sub_stress: NDArray | None = None
        if self.parent_results.stress is not None:
            from feaweld.core.types import StressField

            parent_stress_vals = self.parent_results.stress.values  # (n_p, 6)
            parent_coords = self.parent_results.mesh.nodes

            # Nearest-neighbour stress transfer
            sub_stress_vals = _nearest_neighbour_transfer(
                parent_coords, parent_stress_vals, self._submodel_mesh.nodes
            )
            sub_stress = sub_stress_vals

        from feaweld.core.types import StressField as _SF

        return FEAResults(
            mesh=self._submodel_mesh,
            displacement=sub_disp,
            stress=_SF(values=sub_stress) if sub_stress is not None else None,
            metadata={"submodel": True, "parent_mesh_size": self._parent_h},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_neighbour_transfer(
    source_coords: NDArray,
    source_values: NDArray,
    target_coords: NDArray,
) -> NDArray:
    """Transfer field values from *source* to *target* via nearest neighbour."""
    n_t = target_coords.shape[0]
    chunk = max(1, min(1000, n_t))
    result = np.empty((n_t, source_values.shape[1]), dtype=np.float64)
    for start in range(0, n_t, chunk):
        end = min(start + chunk, n_t)
        diff = (
            target_coords[start:end, np.newaxis, :]
            - source_coords[np.newaxis, :, :]
        )
        dist_sq = np.sum(diff ** 2, axis=2)
        idx = np.argmin(dist_sq, axis=1)
        result[start:end] = source_values[idx]
    return result


def _characteristic_element_size(mesh: FEMesh) -> float:
    """Estimate average element edge length from bounding box and element count."""
    bbox = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)
    vol = float(np.prod(bbox + 1e-30))
    return (vol / max(mesh.n_elements, 1)) ** (1.0 / mesh.ndim)


def _refine_mesh(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Subdivide a mesh by inserting mid-edge nodes.

    For each refinement level, every element edge is bisected and the
    element is split into smaller elements of the same type.  This
    simplified implementation handles triangular and tetrahedral elements
    directly and falls back to a simple mid-point insertion for others.

    Parameters
    ----------
    nodes:
        Nodal coordinates ``(n_nodes, 3)``.
    elements:
        Element connectivity ``(n_elements, n_nodes_per_elem)``.
    factor:
        Target refinement factor.  The mesh is bisected
        ``ceil(log2(factor))`` times.

    Returns
    -------
    (refined_nodes, refined_elements)
    """
    import math as _math

    n_bisections = max(1, int(_math.ceil(_math.log2(factor))))
    cur_nodes = np.array(nodes, dtype=np.float64)
    cur_elems = np.array(elements, dtype=np.int64)

    for _ in range(n_bisections):
        cur_nodes, cur_elems = _bisect_once(cur_nodes, cur_elems)

    return cur_nodes, cur_elems


def _bisect_once(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """One level of uniform bisection."""
    nodes_per_elem = elements.shape[1]

    # Build a unique edge → midpoint map.
    edge_mid: dict[tuple[int, int], int] = {}
    new_nodes_list: list[NDArray] = [nodes.copy()]
    next_id = len(nodes)

    def get_mid(a: int, b: int) -> int:
        nonlocal next_id
        key = (min(a, b), max(a, b))
        if key in edge_mid:
            return edge_mid[key]
        mid = (nodes[a] + nodes[b]) / 2.0
        new_nodes_list.append(mid.reshape(1, -1))
        edge_mid[key] = next_id
        next_id += 1
        return edge_mid[key]

    new_elems: list[list[int]] = []

    if nodes_per_elem == 3:
        # Triangle → 4 triangles
        for tri in elements:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_elems.append([a, ab, ca])
            new_elems.append([ab, b, bc])
            new_elems.append([ca, bc, c])
            new_elems.append([ab, bc, ca])
    elif nodes_per_elem == 4 and nodes.shape[1] == 3:
        # Tetrahedron → 8 tetrahedra (simplified: 8-tet bisection)
        for tet in elements:
            n0, n1, n2, n3 = (int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]))
            m01 = get_mid(n0, n1)
            m02 = get_mid(n0, n2)
            m03 = get_mid(n0, n3)
            m12 = get_mid(n1, n2)
            m13 = get_mid(n1, n3)
            m23 = get_mid(n2, n3)
            # 8 sub-tetrahedra
            new_elems.append([n0, m01, m02, m03])
            new_elems.append([m01, n1, m12, m13])
            new_elems.append([m02, m12, n2, m23])
            new_elems.append([m03, m13, m23, n3])
            new_elems.append([m01, m02, m03, m13])
            new_elems.append([m01, m02, m12, m13])
            new_elems.append([m02, m03, m13, m23])
            new_elems.append([m02, m12, m13, m23])
    elif nodes_per_elem == 4 and nodes.shape[1] == 2:
        # Quad → 4 quads
        for quad in elements:
            a, b, c, d = (int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]))
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            cd = get_mid(c, d)
            da = get_mid(d, a)
            # centre
            centre_coord = (nodes[a] + nodes[b] + nodes[c] + nodes[d]) / 4.0
            new_nodes_list.append(centre_coord.reshape(1, -1))
            cen = next_id
            next_id += 1
            new_elems.append([a, ab, cen, da])
            new_elems.append([ab, b, bc, cen])
            new_elems.append([cen, bc, c, cd])
            new_elems.append([da, cen, cd, d])
    else:
        # Fallback: keep elements unchanged (no subdivision for exotic types)
        return nodes, elements

    refined_nodes = np.concatenate(new_nodes_list, axis=0)
    refined_elems = np.array(new_elems, dtype=np.int64)
    return refined_nodes, refined_elems
