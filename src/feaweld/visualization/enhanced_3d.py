"""Enhanced 3D visualization functions for feaweld FEA results.

Extends the base PyVista visualization in [stress_plots][feaweld.visualization.stress_plots] with clipping
planes, threshold filtering, iso-surfaces, force-vector glyphs, weld-region
highlighting, SED control-volume rendering, mesh previews, and annotated
stress plots.

All PyVista imports are deferred inside functions so the rest of feaweld
remains usable without the (optional) pyvista dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEMesh, StressField
from feaweld.visualization.theme import get_cmap, configure_plotter
from feaweld.visualization.stress_plots import resolve_component, resolve_grid


# ---------------------------------------------------------------------------
# 1. Clipping-plane stress plot
# ---------------------------------------------------------------------------

def plot_stress_with_clipping(
    mesh: Any,
    stress: StressField | None = None,
    clip_normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    clip_origin: tuple[float, float, float] | None = None,
    component: str = "von_mises",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot stress on a clipped cross-section of the mesh.

    A clipping plane defined by *clip_normal* and *clip_origin* slices the
    mesh.  The clipped region is shown with a filled stress contour while the
    full mesh outline is rendered as a translucent wireframe for context.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Finite-element mesh, or a ready PyVista grid.
    stress : feaweld.core.types.StressField, optional
        Nodal stress field.  Required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh].
    clip_normal : tuple of float
        Normal vector of the clipping plane.
    clip_origin : tuple of float, optional
        A point on the clipping plane.  Defaults to the mesh centroid.
    component : str
        Stress component to display.
    show : bool
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    scalar_key = resolve_component(component)

    grid = resolve_grid(mesh, stress)

    if clip_origin is None:
        clip_origin = tuple(grid.center)

    clipped = grid.clip(
        normal=clip_normal,
        origin=clip_origin,
    )

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Full mesh as translucent wireframe for spatial context.
    plotter.add_mesh(
        grid,
        style="wireframe",
        color="gray",
        opacity=0.15,
        line_width=0.5,
    )

    # Clipped region with stress contour.
    plotter.add_mesh(
        clipped,
        scalars=scalar_key,
        cmap=kwargs.pop("cmap", get_cmap("stress")),
        show_scalar_bar=True,
        scalar_bar_args={"title": component.replace("_", " ").title()},
        **kwargs,
    )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 2. Threshold stress plot
# ---------------------------------------------------------------------------

def plot_stress_threshold(
    mesh: Any,
    stress: StressField | None = None,
    threshold: float | None = None,
    component: str = "von_mises",
    above: bool = True,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Highlight regions where a stress component exceeds (or falls below) a threshold.

    The thresholded region is displayed opaque with a colour contour, while
    the full mesh is rendered underneath at low opacity for spatial reference.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Finite-element mesh, or a ready PyVista grid.
    stress : feaweld.core.types.StressField, optional
        Nodal stress field.  Required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh].
    threshold : float, optional
        Scalar threshold value (same units as the stress component).  When
        ``None`` the median of the component field is used.
    component : str
        Stress component to filter on.
    above : bool
        If *True*, show the region **above** *threshold*; otherwise show the
        region **below** it.
    show : bool
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    scalar_key = resolve_component(component)

    grid = resolve_grid(mesh, stress)
    grid.set_active_scalars(scalar_key)

    if threshold is None:
        threshold = float(np.median(np.asarray(grid.point_data[scalar_key])))

    thresholded = grid.threshold(
        value=threshold,
        scalars=scalar_key,
        invert=not above,
    )

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Full mesh as translucent background.
    plotter.add_mesh(
        grid,
        scalars=scalar_key,
        cmap=kwargs.pop("cmap", get_cmap("stress")),
        opacity=0.15,
        show_scalar_bar=False,
    )

    # Thresholded region, opaque with scalar bar. The region can be empty
    # (no cells satisfy the criterion) — pyvista refuses to plot an empty
    # mesh, so annotate instead of crashing.
    if thresholded.n_points > 0:
        plotter.add_mesh(
            thresholded,
            scalars=scalar_key,
            cmap=kwargs.pop("threshold_cmap", get_cmap("stress")),
            show_scalar_bar=True,
            scalar_bar_args={
                "title": f"{component.replace('_', ' ').title()} "
                         f"({'>' if above else '<'} {threshold:.1f})",
            },
            **kwargs,
        )
    else:
        plotter.add_text(
            f"No region with {component} "
            f"{'>' if above else '<'} {threshold:.1f}",
            font_size=12,
            color="gray",
        )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 3. Iso-surface plot
# ---------------------------------------------------------------------------

def plot_iso_surface(
    mesh: Any,
    stress: StressField | None = None,
    iso_values: list[float] | NDArray | None = None,
    component: str = "von_mises",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot iso-surfaces of a stress component inside the mesh volume.

    Each iso-value produces a surface of constant stress rendered with slight
    transparency.  The original mesh is shown as a translucent wireframe.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Finite-element mesh (volumetric elements recommended), or a ready grid.
    stress : feaweld.core.types.StressField, optional
        Nodal stress field.  Required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh].
    iso_values : list of float or numpy.ndarray, optional
        One or more iso-surface levels.  When ``None`` three levels spanning
        the inter-quartile range of the component field are used.
    component : str
        Stress component to contour.
    show : bool
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    scalar_key = resolve_component(component)

    grid = resolve_grid(mesh, stress)
    grid.set_active_scalars(scalar_key)

    if iso_values is None:
        field = np.asarray(grid.point_data[scalar_key])
        iso_values = np.linspace(
            float(np.percentile(field, 25)),
            float(np.percentile(field, 75)),
            3,
        )

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Original mesh as translucent wireframe.
    plotter.add_mesh(
        grid,
        style="wireframe",
        color="lightgray",
        opacity=0.1,
        line_width=0.5,
    )

    iso_values = np.atleast_1d(np.asarray(iso_values, dtype=np.float64))
    cmap = kwargs.pop("cmap", "coolwarm")

    for i, iso_val in enumerate(iso_values):
        contour = grid.contour(isosurfaces=[iso_val], scalars=scalar_key)
        if contour.n_points == 0:
            continue
        opacity = max(0.3, 0.8 - 0.15 * i)
        plotter.add_mesh(
            contour,
            opacity=opacity,
            cmap=cmap,
            scalars=scalar_key,
            clim=[float(iso_values.min()), float(iso_values.max())],
            show_scalar_bar=(i == 0),
            scalar_bar_args={"title": component.replace("_", " ").title()},
            **kwargs,
        )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 4. Force-vector plot
# ---------------------------------------------------------------------------

def plot_force_vectors(
    mesh: FEMesh,
    vectors: NDArray,
    scale: float = 1.0,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot force (or displacement) vectors as 3D arrows on the mesh.

    Parameters
    ----------
    mesh:
        Finite-element mesh.
    vectors:
        ``(n_nodes, 3)`` array of vector quantities (forces, displacements,
        etc.) defined at each node.
    scale:
        Scaling factor applied to arrow length.
    show:
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv
    from feaweld.visualization.stress_plots import _mesh_to_pyvista_grid

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape != (mesh.n_nodes, 3):
        raise ValueError(
            f"vectors must have shape ({mesh.n_nodes}, 3), "
            f"got {vectors.shape}"
        )

    magnitudes = np.linalg.norm(vectors, axis=1)

    # Filter out zero-magnitude vectors to avoid degenerate glyphs.
    nonzero_mask = magnitudes > 1e-30
    active_points = np.asarray(mesh.nodes[nonzero_mask], dtype=np.float64)
    active_vectors = vectors[nonzero_mask]
    active_mag = magnitudes[nonzero_mask]

    cloud = pv.PolyData(active_points)
    cloud["magnitude"] = active_mag
    cloud["vectors"] = active_vectors

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Add mesh as translucent surface for context.
    grid = _mesh_to_pyvista_grid(mesh)
    plotter.add_mesh(
        grid,
        color="lightgray",
        opacity=0.3,
        show_edges=True,
        edge_color="gray",
    )

    # Create arrow glyphs oriented by the vector field.
    cloud.set_active_vectors("vectors")
    glyphs = cloud.glyph(
        orient="vectors",
        scale="magnitude",
        factor=scale,
    )

    plotter.add_mesh(
        glyphs,
        scalars="magnitude",
        cmap=kwargs.pop("cmap", "plasma"),
        show_scalar_bar=True,
        scalar_bar_args={"title": "Magnitude"},
        **kwargs,
    )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 5. Weld-region highlight
# ---------------------------------------------------------------------------

def plot_weld_region_highlight(
    mesh: Any,
    stress: StressField | None = None,
    weld_region: str = "weld",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Highlight a weld region on the mesh, optionally coloured by stress.

    The function looks for *weld_region* in ``mesh.physical_groups`` and then
    ``mesh.element_sets``.  If found, the corresponding elements are extracted
    and rendered opaque while the remainder of the mesh is translucent.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Finite-element mesh, or a ready PyVista grid.  Region lookup requires
        an [FEMesh][feaweld.core.types.FEMesh]; a raw grid is rendered without a highlight.
    stress : feaweld.core.types.StressField, optional
        Optional stress field.  When provided, the weld sub-mesh is coloured
        by von-Mises stress; otherwise a solid highlight colour is used.
    weld_region : str
        Name of the physical group or element set identifying the weld.
    show : bool
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    # Build the full grid (with or without stress data).
    grid = resolve_grid(mesh, stress)

    # Locate the weld element indices (only possible for an FEMesh).
    weld_elem_ids: NDArray[np.int64] | None = None
    if hasattr(mesh, "physical_groups") and weld_region in mesh.physical_groups:
        weld_elem_ids = mesh.physical_groups[weld_region]
    elif hasattr(mesh, "element_sets") and weld_region in mesh.element_sets:
        weld_elem_ids = mesh.element_sets[weld_region]

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Base mesh translucent.
    plotter.add_mesh(
        grid,
        color="lightgray",
        opacity=0.2,
        show_edges=True,
        edge_color="silver",
    )

    if weld_elem_ids is not None and len(weld_elem_ids) > 0:
        weld_grid = grid.extract_cells(weld_elem_ids)

        if stress is not None:
            plotter.add_mesh(
                weld_grid,
                scalars="von_mises",
                cmap=kwargs.pop("cmap", get_cmap("stress")),
                show_scalar_bar=True,
                scalar_bar_args={"title": "Von Mises (MPa) - Weld"},
                **kwargs,
            )
        else:
            plotter.add_mesh(
                weld_grid,
                color=kwargs.pop("highlight_color", "tomato"),
                show_edges=True,
                edge_color="darkred",
                **kwargs,
            )
    else:
        # No matching group found; render full mesh normally and warn.
        plotter.add_text(
            f"Region '{weld_region}' not found in mesh groups",
            position="upper_left",
            font_size=10,
            color="orange",
        )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 6. SED control-volume visualization
# ---------------------------------------------------------------------------

def plot_sed_control_volume(
    mesh: Any,
    center_point: tuple[float, float, float] | NDArray,
    control_radius: float,
    sed_result: Any | None = None,
    stress: StressField | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Visualize the SED control volume around a critical point.

    Renders the mesh translucently and overlays a wireframe sphere
    representing the control volume.  If a stress field is given, the
    elements inside the sphere are coloured by strain energy density (via
    the SED field on *sed_result*) or by von-Mises stress.

    Parameters
    ----------
    mesh:
        Finite-element mesh.
    center_point:
        Center of the control volume (x, y, z).
    control_radius:
        Radius R_0 of the averaging control volume (mesh length units).
    sed_result:
        An [SEDResult][feaweld.postprocess.sed.SEDResult] instance.  If its
        ``sed_field`` attribute is populated the clipped region is coloured
        by SED; otherwise von-Mises stress is used (when *stress* is given).
    stress:
        Optional stress field for fallback colouring.
    show:
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    center = np.asarray(center_point, dtype=np.float64).ravel()

    # Build grid.
    grid = resolve_grid(mesh, stress)

    # Attach SED field if available.
    has_sed_field = False
    if sed_result is not None and getattr(sed_result, "sed_field", None) is not None:
        sed_field = np.asarray(sed_result.sed_field, dtype=np.float64)
        # sed_field may be per-element or per-node; handle both.
        if sed_field.shape[0] == grid.n_cells:
            grid.cell_data["SED"] = sed_field
        elif sed_field.shape[0] == grid.n_points:
            grid.point_data["SED"] = sed_field
        has_sed_field = True

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Full mesh translucent.
    plotter.add_mesh(
        grid,
        color="lightgray",
        opacity=0.15,
        show_edges=False,
    )

    # Wireframe sphere showing control volume boundary.
    sphere = pv.Sphere(
        radius=control_radius,
        center=center,
        theta_resolution=30,
        phi_resolution=30,
    )
    plotter.add_mesh(
        sphere,
        style="wireframe",
        color="blue",
        line_width=1.5,
        opacity=0.5,
    )

    # Clip mesh to the sphere region using the sphere as an implicit surface.
    clipped = grid.clip_surface(sphere, invert=False)

    if clipped.n_points > 0:
        if has_sed_field and "SED" in clipped.point_data:
            scalars = "SED"
            title = "SED (MJ/m\u00b3)"
        elif has_sed_field and "SED" in clipped.cell_data:
            scalars = "SED"
            title = "SED (MJ/m\u00b3)"
        elif stress is not None:
            scalars = "von_mises"
            title = "Von Mises (MPa)"
        else:
            scalars = None
            title = ""

        mesh_kwargs: dict[str, Any] = {}
        if scalars is not None:
            mesh_kwargs.update(
                scalars=scalars,
                cmap=kwargs.pop("cmap", "turbo"),
                show_scalar_bar=True,
                scalar_bar_args={"title": title},
            )
        else:
            mesh_kwargs.update(color="steelblue")

        plotter.add_mesh(clipped, **mesh_kwargs, **kwargs)

    # Annotation with averaged SED value.
    if sed_result is not None:
        avg_sed = getattr(sed_result, "averaged_sed", None)
        if avg_sed is not None:
            plotter.add_text(
                f"Averaged SED: {avg_sed:.4g} MJ/m\u00b3\n"
                f"R\u2080 = {control_radius:.3g} mm",
                position="upper_right",
                font_size=10,
                color="black",
            )

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 7. Mesh preview (no stress data needed)
# ---------------------------------------------------------------------------

def plot_mesh_preview(
    mesh: FEMesh,
    highlight_sets: dict[str, str] | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Preview the finite-element mesh without any field data.

    Renders the mesh with visible edges.  Named element sets or physical
    groups can be highlighted with custom colours via *highlight_sets*.  Node
    sets are rendered as small coloured spheres.

    Parameters
    ----------
    mesh:
        Finite-element mesh.
    highlight_sets:
        Optional mapping from group/set name to colour string.  The function
        searches ``mesh.physical_groups`` and ``mesh.element_sets`` for each
        name and highlights matching elements.
    show:
        Open an interactive window.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv
    from feaweld.visualization.stress_plots import _mesh_to_pyvista_grid

    grid = _mesh_to_pyvista_grid(mesh)

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    # Base mesh: light gray with edges.
    plotter.add_mesh(
        grid,
        color="whitesmoke",
        show_edges=True,
        edge_color="gray",
        opacity=0.8 if highlight_sets else 1.0,
        **kwargs,
    )

    # Highlight element groups.
    if highlight_sets:
        for group_name, color_str in highlight_sets.items():
            elem_ids: NDArray[np.int64] | None = None
            if group_name in mesh.physical_groups:
                elem_ids = mesh.physical_groups[group_name]
            elif group_name in mesh.element_sets:
                elem_ids = mesh.element_sets[group_name]

            if elem_ids is not None and len(elem_ids) > 0:
                sub_grid = grid.extract_cells(elem_ids)
                plotter.add_mesh(
                    sub_grid,
                    color=color_str,
                    show_edges=True,
                    edge_color="black",
                    label=group_name,
                )

    # Render node sets as point spheres.
    if mesh.node_sets:
        set_colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        for i, (set_name, node_ids) in enumerate(mesh.node_sets.items()):
            if len(node_ids) == 0:
                continue
            pts = pv.PolyData(np.asarray(mesh.nodes[node_ids], dtype=np.float64))
            color = set_colors[i % len(set_colors)]
            plotter.add_mesh(
                pts,
                color=color,
                point_size=8,
                render_points_as_spheres=True,
                label=set_name,
            )

    # Show legend if we added any labeled meshes.
    if highlight_sets or mesh.node_sets:
        plotter.add_legend(bcolor="white", face="circle", size=(0.15, 0.15))

    plotter.add_axes()
    plotter.add_axes_at_origin(labels_off=True, line_width=2)
    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# 8. Annotated stress plot
# ---------------------------------------------------------------------------

def _critical_points_from_grid(
    grid: Any,
    scalar_key: str,
    n_max: int,
) -> list[Any]:
    """Build critical-point annotations from a raw grid's scalar array.

    Used when [plot_annotated_stress][feaweld.visualization.enhanced_3d.plot_annotated_stress] is handed a ready PyVista grid
    (rather than an ``FEMesh``/``StressField`` pair), so the mesh-based
    [find_critical_points][feaweld.visualization.annotations.find_critical_points] cannot be
    called.  Selects the top *n_max* points by the *scalar_key* array.

    Parameters
    ----------
    grid : pyvista.DataSet
        Grid carrying the ``scalar_key`` point-data array.
    scalar_key : str
        Point-data key to rank points by.
    n_max : int
        Maximum number of points to return.

    Returns
    -------
    list of feaweld.visualization.annotations.CriticalPoint
        Ranked descending by value; the top point is ``"warning"`` severity,
        the rest ``"info"``.
    """
    from feaweld.visualization.annotations import (
        CriticalPoint,
        format_engineering_value,
    )

    values = np.asarray(grid.point_data[scalar_key], dtype=np.float64)
    points = np.asarray(grid.points, dtype=np.float64)
    top_indices = np.argsort(values)[::-1][:n_max]

    result: list[CriticalPoint] = []
    for rank, idx in enumerate(top_indices):
        val = float(values[idx])
        label = format_engineering_value(val, "MPa")
        label = f"Max: {label}" if rank == 0 else label
        result.append(CriticalPoint(
            location=points[idx].copy(),
            value=val,
            label=label,
            severity="warning" if rank == 0 else "info",
            category="stress",
        ))
    return result


def plot_annotated_stress(
    mesh: Any,
    stress: StressField | None = None,
    annotations: list[Any] | None = None,
    component: str = "von_mises",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Stress contour plot with critical-point annotations.

    If *annotations* is ``None``, critical points are detected automatically:
    via [find_critical_points][feaweld.visualization.annotations.find_critical_points] for an
    [FEMesh][feaweld.core.types.FEMesh], or from the grid's scalar array via
    `_critical_points_from_grid` when a ready grid is supplied.

    Parameters
    ----------
    mesh : feaweld.core.types.FEMesh or pyvista.DataSet
        Finite-element mesh, or a ready PyVista grid.
    stress : feaweld.core.types.StressField, optional
        Nodal stress field.  Required when *mesh* is an [FEMesh][feaweld.core.types.FEMesh].
    annotations : list of CriticalPoint, optional
        Pre-computed list of
        [CriticalPoint][feaweld.visualization.annotations.CriticalPoint].
        When ``None``, they are generated automatically.
    component : str
        Stress component to display.
    show : bool
        Open an interactive window.
    **kwargs
        Forwarded to ``plotter.add_mesh()``.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv
    from feaweld.visualization.annotations import (
        annotate_3d,
        find_critical_points,
    )

    scalar_key = resolve_component(component)
    n_max = kwargs.pop("n_max", 5)

    grid = resolve_grid(mesh, stress)

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)

    plotter.add_mesh(
        grid,
        scalars=scalar_key,
        cmap=kwargs.pop("cmap", get_cmap("stress")),
        show_scalar_bar=True,
        scalar_bar_args={"title": component.replace("_", " ").title()},
        **kwargs,
    )

    # Auto-detect critical points when none are supplied.
    if annotations is None:
        if hasattr(mesh, "point_data"):
            annotations = _critical_points_from_grid(grid, scalar_key, n_max)
        else:
            annotations = find_critical_points(mesh, stress, n_max=n_max)

    annotate_3d(plotter, annotations)

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter
