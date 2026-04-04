"""PyVista-based 3D stress visualization for FEA results.

All PyVista imports are deferred to inside the functions that need them
so that the rest of feaweld remains usable without the (optional) pyvista
dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import ElementType, FEMesh, StressField


# ---------------------------------------------------------------------------
# Element-type → VTK mapping
# ---------------------------------------------------------------------------

_VTK_CELL_TYPE: dict[ElementType, int] = {
    ElementType.TRI3: 5,      # VTK_TRIANGLE
    ElementType.TRI6: 22,     # VTK_QUADRATIC_TRIANGLE
    ElementType.QUAD4: 9,     # VTK_QUAD
    ElementType.QUAD8: 23,    # VTK_QUADRATIC_QUAD
    ElementType.TET4: 10,     # VTK_TETRA
    ElementType.TET10: 24,    # VTK_QUADRATIC_TETRA
    ElementType.HEX8: 12,     # VTK_HEXAHEDRON
    ElementType.HEX20: 25,    # VTK_QUADRATIC_HEXAHEDRON
}


# ---------------------------------------------------------------------------
# Mesh → PyVista conversion
# ---------------------------------------------------------------------------

def stress_field_to_pyvista(
    mesh: FEMesh,
    stress: StressField,
) -> Any:
    """Convert an :class:`FEMesh` and :class:`StressField` to a PyVista ``UnstructuredGrid``.

    Parameters
    ----------
    mesh:
        The finite-element mesh.
    stress:
        Stress field defined at nodes.

    Returns
    -------
    pyvista.UnstructuredGrid
        Grid with point-data arrays for each stress component and derived
        quantities (von Mises, Tresca, principal stresses).
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for visualization. "
            "Install with: pip install pyvista"
        ) from exc

    vtk_type = _VTK_CELL_TYPE.get(mesh.element_type)
    if vtk_type is None:
        raise ValueError(f"Unsupported element type: {mesh.element_type}")

    n_per_elem = mesh.elements.shape[1]
    n_elems = mesh.n_elements

    # Build the cells array expected by PyVista:
    # [n_per_elem, node0, node1, ..., n_per_elem, node0, ...]
    cells = np.empty((n_elems, n_per_elem + 1), dtype=np.int64)
    cells[:, 0] = n_per_elem
    cells[:, 1:] = mesh.elements
    cells = cells.ravel()

    cell_types = np.full(n_elems, vtk_type, dtype=np.uint8)
    points = np.asarray(mesh.nodes, dtype=np.float64)

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Attach stress components as point data.
    component_names = ["stress_xx", "stress_yy", "stress_zz",
                       "stress_xy", "stress_yz", "stress_xz"]
    for i, name in enumerate(component_names):
        grid.point_data[name] = stress.values[:, i]

    # Derived quantities
    grid.point_data["von_mises"] = stress.von_mises
    grid.point_data["tresca"] = stress.tresca

    principals = stress.principal  # (n, 3) sorted ascending
    grid.point_data["principal_1"] = principals[:, 2]  # max
    grid.point_data["principal_2"] = principals[:, 1]
    grid.point_data["principal_3"] = principals[:, 0]  # min

    return grid


# ---------------------------------------------------------------------------
# Stress contour plot
# ---------------------------------------------------------------------------

# Map user-friendly component names to point-data keys.
_COMPONENT_MAP: dict[str, str] = {
    "von_mises": "von_mises",
    "tresca": "tresca",
    "xx": "stress_xx",
    "yy": "stress_yy",
    "zz": "stress_zz",
    "xy": "stress_xy",
    "yz": "stress_yz",
    "xz": "stress_xz",
    "principal_1": "principal_1",
    "principal_2": "principal_2",
    "principal_3": "principal_3",
}


def plot_stress_field(
    mesh: FEMesh,
    stress: StressField,
    component: str = "von_mises",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot a stress component on the FE mesh.

    Parameters
    ----------
    mesh:
        FE mesh.
    stress:
        Stress field at nodes.
    component:
        Which scalar to display.  One of ``"von_mises"``, ``"tresca"``,
        ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ``"xz"``,
        ``"principal_1"``, ``"principal_3"``.
    show:
        If *True* display an interactive window.  Pass *False* for
        off-screen or headless use.
    **kwargs:
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
        The plotter instance (useful for further customisation or export).
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for visualization. "
            "Install with: pip install pyvista"
        ) from exc

    grid = stress_field_to_pyvista(mesh, stress)

    scalar_key = _COMPONENT_MAP.get(component)
    if scalar_key is None:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Choose from: {list(_COMPONENT_MAP)}"
        )

    plotter = pv.Plotter(off_screen=not show)
    plotter.add_mesh(
        grid,
        scalars=scalar_key,
        cmap=kwargs.pop("cmap", "jet"),
        show_scalar_bar=True,
        scalar_bar_args={"title": component.replace("_", " ").title()},
        **kwargs,
    )
    plotter.add_axes()

    if show:
        plotter.show()

    return plotter


# ---------------------------------------------------------------------------
# Deformed-shape plot
# ---------------------------------------------------------------------------

def plot_deformed(
    mesh: FEMesh,
    displacement: NDArray,
    scale: float = 10.0,
    stress: StressField | None = None,
    show: bool = True,
) -> Any:
    """Plot the deformed mesh shape, optionally coloured by stress.

    Parameters
    ----------
    mesh:
        Undeformed mesh.
    displacement:
        Nodal displacement array ``(n_nodes, 3)``.
    scale:
        Displacement magnification factor.
    stress:
        If provided, the deformed mesh is coloured by von-Mises stress.
    show:
        Display the interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for visualization. "
            "Install with: pip install pyvista"
        ) from exc

    # Build deformed coordinates
    deformed_nodes = mesh.nodes + scale * displacement

    deformed_mesh = FEMesh(
        nodes=deformed_nodes,
        elements=mesh.elements,
        element_type=mesh.element_type,
    )

    plotter = pv.Plotter(off_screen=not show)

    if stress is not None:
        grid = stress_field_to_pyvista(deformed_mesh, stress)
        plotter.add_mesh(
            grid,
            scalars="von_mises",
            cmap="jet",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Von Mises (MPa)"},
        )
    else:
        grid = _mesh_to_pyvista_grid(deformed_mesh)
        plotter.add_mesh(grid, color="steelblue", show_edges=True)

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# Temperature plot
# ---------------------------------------------------------------------------

def plot_temperature_field(
    mesh: FEMesh,
    temperature: NDArray,
    show: bool = True,
) -> Any:
    """Plot temperature contours on the FE mesh.

    Parameters
    ----------
    mesh:
        FE mesh.
    temperature:
        Nodal temperature array ``(n_nodes,)``.
    show:
        Display interactively.

    Returns
    -------
    pyvista.Plotter
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for visualization. "
            "Install with: pip install pyvista"
        ) from exc

    grid = _mesh_to_pyvista_grid(mesh)
    grid.point_data["Temperature"] = np.asarray(temperature, dtype=np.float64)

    plotter = pv.Plotter(off_screen=not show)
    plotter.add_mesh(
        grid,
        scalars="Temperature",
        cmap="coolwarm",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Temperature (\u00b0C)"},
    )
    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mesh_to_pyvista_grid(mesh: FEMesh) -> Any:
    """Create a bare PyVista UnstructuredGrid from an FEMesh (no field data)."""
    import pyvista as pv  # type: ignore[import-untyped]

    vtk_type = _VTK_CELL_TYPE.get(mesh.element_type)
    if vtk_type is None:
        raise ValueError(f"Unsupported element type: {mesh.element_type}")

    n_per_elem = mesh.elements.shape[1]
    n_elems = mesh.n_elements

    cells = np.empty((n_elems, n_per_elem + 1), dtype=np.int64)
    cells[:, 0] = n_per_elem
    cells[:, 1:] = mesh.elements
    cells = cells.ravel()

    cell_types = np.full(n_elems, vtk_type, dtype=np.uint8)
    points = np.asarray(mesh.nodes, dtype=np.float64)

    return pv.UnstructuredGrid(cells, cell_types, points)
