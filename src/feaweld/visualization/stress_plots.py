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
    """Convert an [FEMesh][feaweld.core.types.FEMesh] and [StressField][feaweld.core.types.StressField] to a PyVista ``UnstructuredGrid``.

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


def resolve_component(component: str) -> str:
    """Resolve a user-friendly stress component name to its point-data key.

    This is the single source of truth for the component-name mapping used
    across the 3-D visualization modules.

    Parameters
    ----------
    component : str
        One of ``"von_mises"``, ``"tresca"``, ``"xx"``, ``"yy"``, ``"zz"``,
        ``"xy"``, ``"yz"``, ``"xz"``, ``"principal_1"``, ``"principal_2"``,
        ``"principal_3"``.

    Returns
    -------
    str
        The PyVista ``point_data`` key for the requested component.

    Raises
    ------
    ValueError
        If *component* is not a recognised name.
    """
    key = _COMPONENT_MAP.get(component)
    if key is None:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Choose from: {list(_COMPONENT_MAP)}"
        )
    return key


def resolve_grid(mesh: Any, stress: StressField | None = None) -> Any:
    """Return a PyVista grid from either an FEMesh(+StressField) or a ready grid.

    Lets the 3-D plotting functions accept either the solver-agnostic
    ``(FEMesh, StressField)`` pair or an already-built PyVista ``DataSet``.
    A ready grid (detected via a ``point_data`` attribute) is returned
    unchanged; otherwise the mesh is converted, attaching the stress field
    when supplied and producing a bare grid when not.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        The finite-element mesh, or a PyVista grid to pass through untouched.
    stress : feaweld.core.types.StressField, optional
        Nodal stress field.  Used only when *mesh* is an [FEMesh][feaweld.core.types.FEMesh];
        when ``None`` a grid without stress arrays is produced.

    Returns
    -------
    pyvista.UnstructuredGrid or pyvista.DataSet
        A grid ready for rendering.
    """
    if hasattr(mesh, "point_data"):
        return mesh
    if stress is not None:
        return stress_field_to_pyvista(mesh, stress)
    return _mesh_to_pyvista_grid(mesh)


def plot_stress_field(
    mesh: Any,
    stress: StressField | None = None,
    component: str = "von_mises",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot a stress component on the FE mesh.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        FE mesh, or a ready PyVista grid carrying the component arrays.
    stress : feaweld.core.types.StressField, optional
        Stress field at nodes.  Required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh];
        ignored when *mesh* is already a grid.
    component : str
        Which scalar to display.  One of ``"von_mises"``, ``"tresca"``,
        ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ``"xz"``,
        ``"principal_1"``, ``"principal_3"``.
    show : bool
        If *True* display an interactive window.  Pass *False* for
        off-screen or headless use.
    **kwargs
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

    grid = resolve_grid(mesh, stress)

    scalar_key = resolve_component(component)

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)
    # Show mesh edges for smaller meshes to aid mesh quality assessment
    show_edges = grid.n_cells < 50_000
    plotter.add_mesh(
        grid,
        scalars=scalar_key,
        cmap=kwargs.pop("cmap", get_cmap("stress")),
        show_scalar_bar=True,
        scalar_bar_args={"title": component.replace("_", " ").title()},
        show_edges=show_edges,
        edge_color="gray",
        edge_opacity=0.15 if show_edges else 0.0,
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
    mesh: Any,
    displacement: NDArray | None = None,
    scale: float = 10.0,
    stress: StressField | None = None,
    show: bool = True,
) -> Any:
    """Plot the deformed mesh shape, optionally coloured by stress.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Undeformed mesh, or a ready (already-deformed) PyVista grid.
    displacement : numpy.ndarray, optional
        Nodal displacement array ``(n_nodes, 3)``.  Required when *mesh* is
        an [FEMesh][feaweld.core.types.FEMesh]; ignored when *mesh* is already a grid.
    scale : float
        Displacement magnification factor.
    stress : feaweld.core.types.StressField, optional
        If provided, the deformed mesh is coloured by von-Mises stress.
    show : bool
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

    if hasattr(mesh, "point_data"):
        # Already a PyVista grid; assume it carries the deformed geometry.
        grid = mesh
        colored = "von_mises" in grid.point_data
    else:
        # Build deformed coordinates from the FEMesh.
        deformed_nodes = mesh.nodes + scale * displacement
        deformed_mesh = FEMesh(
            nodes=deformed_nodes,
            elements=mesh.elements,
            element_type=mesh.element_type,
        )
        grid = resolve_grid(deformed_mesh, stress)
        colored = stress is not None

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    if colored:
        plotter.add_mesh(
            grid,
            scalars="von_mises",
            cmap=get_cmap("stress"),
            show_scalar_bar=True,
            scalar_bar_args={"title": "Von Mises (MPa)"},
        )
    else:
        plotter.add_mesh(grid, color="steelblue", show_edges=True)

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# Temperature plot
# ---------------------------------------------------------------------------

def plot_temperature_field(
    mesh: Any,
    temperature: NDArray | None = None,
    show: bool = True,
) -> Any:
    """Plot temperature contours on the FE mesh.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        FE mesh, or a ready PyVista grid (carrying a ``"Temperature"`` array
        when *temperature* is omitted).
    temperature : numpy.ndarray, optional
        Nodal temperature array ``(n_nodes,)``.  Attached to the grid when
        supplied; required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh].
    show : bool
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

    grid = resolve_grid(mesh)
    if temperature is not None:
        grid.point_data["Temperature"] = np.asarray(temperature, dtype=np.float64)

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)
    plotter.add_mesh(
        grid,
        scalars="Temperature",
        cmap=get_cmap("temperature"),
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
