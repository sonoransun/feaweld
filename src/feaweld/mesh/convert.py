"""Mesh format conversion utilities.

Converts between the internal :class:`~feaweld.core.types.FEMesh`
representation and external formats (meshio, VTK, DOLFINx).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import meshio

from feaweld.core.types import ElementType, FEMesh


# ---------------------------------------------------------------------------
# ElementType ↔ meshio cell-type mapping
# ---------------------------------------------------------------------------

_FEMESH_TO_MESHIO: dict[ElementType, str] = {
    ElementType.TRI3: "triangle",
    ElementType.TRI6: "triangle6",
    ElementType.QUAD4: "quad",
    ElementType.QUAD8: "quad8",
    ElementType.TET4: "tetra",
    ElementType.TET10: "tetra10",
    ElementType.HEX8: "hexahedron",
    ElementType.HEX20: "hexahedron20",
}

_MESHIO_TO_FEMESH: dict[str, ElementType] = {v: k for k, v in _FEMESH_TO_MESHIO.items()}


# ---------------------------------------------------------------------------
# FEMesh → meshio
# ---------------------------------------------------------------------------

def femesh_to_meshio(mesh: FEMesh) -> meshio.Mesh:
    """Convert an :class:`FEMesh` to a :class:`meshio.Mesh`.

    Physical groups that map to element indices are stored as
    ``meshio`` *cell_data* under the ``"physical"`` key.  Node sets are
    stored in *point_data* as integer masks.
    """
    cell_type = _FEMESH_TO_MESHIO[mesh.element_type]
    cells = [meshio.CellBlock(cell_type, mesh.elements)]

    # Cell data: physical group membership as an integer tag per element
    cell_data: dict[str, list[NDArray]] = {}
    if mesh.physical_groups:
        tags = np.zeros(mesh.n_elements, dtype=np.int64)
        for idx, (name, elem_ids) in enumerate(mesh.physical_groups.items(), start=1):
            tags[elem_ids] = idx
        cell_data["physical"] = [tags]

    # Point data: node set membership
    point_data: dict[str, NDArray] = {}
    for name, node_ids in mesh.node_sets.items():
        mask = np.zeros(mesh.n_nodes, dtype=np.int64)
        mask[node_ids] = 1
        point_data[name] = mask

    return meshio.Mesh(
        points=mesh.nodes,
        cells=cells,
        cell_data=cell_data,
        point_data=point_data,
    )


# ---------------------------------------------------------------------------
# meshio → FEMesh
# ---------------------------------------------------------------------------

def meshio_to_femesh(mio_mesh: meshio.Mesh) -> FEMesh:
    """Convert a :class:`meshio.Mesh` to an :class:`FEMesh`.

    Only the **first** cell block is imported.  If ``"physical"`` cell
    data is present it is converted back into named physical groups
    (``group_1``, ``group_2``, ...).
    """
    if not mio_mesh.cells:
        raise ValueError("The meshio.Mesh contains no cell blocks.")

    block = mio_mesh.cells[0]
    cell_type_str: str = block.type
    elements = np.asarray(block.data, dtype=np.int64)

    element_type = _MESHIO_TO_FEMESH.get(cell_type_str)
    if element_type is None:
        raise ValueError(
            f"Unsupported meshio cell type '{cell_type_str}'. "
            f"Supported types: {list(_MESHIO_TO_FEMESH.keys())}"
        )

    nodes = np.asarray(mio_mesh.points, dtype=np.float64)

    # Reconstruct physical groups from cell data
    physical_groups: dict[str, NDArray[np.int64]] = {}
    if "physical" in mio_mesh.cell_data:
        phys_tags = np.asarray(mio_mesh.cell_data["physical"][0], dtype=np.int64)
        for tag_val in np.unique(phys_tags):
            if tag_val == 0:
                continue  # unassigned
            name = f"group_{tag_val}"
            physical_groups[name] = np.where(phys_tags == tag_val)[0]

    # Reconstruct node sets from point data (integer masks)
    node_sets: dict[str, NDArray[np.int64]] = {}
    for key, arr in (mio_mesh.point_data or {}).items():
        arr = np.asarray(arr, dtype=np.int64)
        if arr.ndim == 1 and set(np.unique(arr)).issubset({0, 1}):
            nids = np.where(arr == 1)[0]
            if len(nids) > 0:
                node_sets[key] = nids

    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=element_type,
        physical_groups=physical_groups,
        node_sets=node_sets,
    )


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def femesh_to_vtk(mesh: FEMesh, filename: str) -> None:
    """Write *mesh* to a VTK file via meshio.

    The file format is inferred from the extension (``.vtk``,
    ``.vtu``, etc.).
    """
    mio = femesh_to_meshio(mesh)
    meshio.write(filename, mio)


# ---------------------------------------------------------------------------
# DOLFINx conversion (optional dependency)
# ---------------------------------------------------------------------------

def femesh_to_dolfinx(mesh: FEMesh):
    """Convert *mesh* to a DOLFINx mesh object.

    Requires ``fenics-dolfinx`` to be installed.  The function imports
    the package lazily so that the rest of feaweld is usable without it.

    Returns
    -------
    dolfinx.mesh.Mesh
    """
    try:
        from dolfinx import mesh as dfx_mesh
        from dolfinx.io import XDMFFile
        from mpi4py import MPI
        import basix
    except ImportError as exc:
        raise ImportError(
            "DOLFINx conversion requires the 'fenics-dolfinx' package. "
            "Install it with: pip install feaweld[fenics]"
        ) from exc

    # Map element type to basix cell type and degree
    _type_map = {
        ElementType.TRI3: (basix.CellType.triangle, 1),
        ElementType.TRI6: (basix.CellType.triangle, 2),
        ElementType.TET4: (basix.CellType.tetrahedron, 1),
        ElementType.TET10: (basix.CellType.tetrahedron, 2),
        ElementType.QUAD4: (basix.CellType.quadrilateral, 1),
        ElementType.QUAD8: (basix.CellType.quadrilateral, 2),
        ElementType.HEX8: (basix.CellType.hexahedron, 1),
        ElementType.HEX20: (basix.CellType.hexahedron, 2),
    }

    if mesh.element_type not in _type_map:
        raise ValueError(
            f"DOLFINx conversion not supported for element type "
            f"{mesh.element_type}"
        )

    cell_type, degree = _type_map[mesh.element_type]
    gdim = mesh.ndim  # geometric dimension

    # Determine topological dimension from cell type
    tdim = {
        basix.CellType.triangle: 2,
        basix.CellType.quadrilateral: 2,
        basix.CellType.tetrahedron: 3,
        basix.CellType.hexahedron: 3,
    }[cell_type]

    # Create basix coordinate element
    coord_elem = basix.ufl.element(
        basix.ElementFamily.P, cell_type, degree,
        basix.LagrangeVariant.equispaced, shape=(gdim,),
    )

    domain = dfx_mesh.create_mesh(
        MPI.COMM_WORLD,
        mesh.elements,
        mesh.nodes[:, :gdim],
        coord_elem,
    )

    return domain
