"""Thermal visualization — Goldak heat source and temperature fields.

Provides 3-D PyVista rendering of the Goldak double-ellipsoid heat
source at an instantaneous moment and along a traveling path. Temperature
field utilities from :mod:`feaweld.visualization.stress_plots`
(``plot_temperature_field``) cover solved fields; this module focuses on
the source-term itself, which is useful for validating welding-process
input before running a transient solve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.visualization.theme import configure_plotter, get_cmap


def _require_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for 3D thermal plots. "
            "Install with: pip install feaweld[viz]"
        ) from exc


def _source_grid(
    source: Any,
    t: float,
    extent: tuple[float, float, float] | None,
    n: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Evaluate a Goldak source on a structured grid centred on its current position."""
    if extent is None:
        ext_x = 3.0 * max(source.a_f, source.a_r)
        ext_y = 3.0 * source.b
        ext_z = 3.0 * source.c
    else:
        ext_x, ext_y, ext_z = extent

    centre = source.start_position + source.direction * source.travel_speed * t
    x = np.linspace(centre[0] - ext_x, centre[0] + ext_x, n)
    y = np.linspace(centre[1] - ext_y, centre[1] + ext_y, n)
    z = np.linspace(centre[2] - ext_z, centre[2] + ext_z, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    q = source.evaluate(X, Y, Z, t)
    return X, Y, Z, q


def render_goldak_source(
    source: Any,
    t: float = 0.0,
    *,
    grid_points: int = 40,
    extent: tuple[float, float, float] | None = None,
    iso_fraction: float = 0.1,
    mesh: Any = None,
    plotter: Any = None,
    show: bool = True,
    screenshot: str | None = None,
) -> Any:
    """Render the Goldak double-ellipsoid heat source as a 3-D iso-surface.

    The source is sampled on a structured grid centred on the current
    torch position (``start_position + direction * travel_speed * t``).
    An iso-surface at ``iso_fraction`` of the peak power density is
    rendered in the ``temperature`` colormap. If an optional ``mesh`` is
    supplied (an :class:`~feaweld.core.types.FEMesh`), its outline is
    drawn as a translucent wireframe for spatial context.

    Parameters
    ----------
    source : feaweld.solver.thermal.GoldakHeatSource
        Configured heat source.
    t : float
        Snapshot time (s).
    grid_points : int
        Number of sample points per axis (``n^3`` total samples).
    extent : tuple of three floats, optional
        Half-extents ``(Lx, Ly, Lz)`` of the sample box around the
        torch centre. Defaults to 3× the respective ellipsoid semi-axes.
    iso_fraction : float
        Iso-surface level as a fraction of peak power density (0 < frac ≤ 1).
    mesh : FEMesh, optional
        Structural mesh to overlay as a translucent wireframe.
    plotter : pyvista.Plotter, optional
        Re-use an existing plotter (useful for composing multiple plots).
    show : bool
        Open an interactive window. When ``False``, ``screenshot`` must
        be supplied or the caller is responsible for using the returned
        plotter.
    screenshot : str, optional
        If set, save a PNG to this path instead of showing interactively.

    Returns
    -------
    pyvista.Plotter
    """
    pv = _require_pyvista()

    X, Y, Z, q = _source_grid(source, t, extent, grid_points)

    grid = pv.StructuredGrid(X, Y, Z)
    grid["power_density"] = q.flatten(order="F")
    peak = float(q.max())

    if plotter is None:
        plotter = pv.Plotter(off_screen=screenshot is not None or not show)
        configure_plotter(plotter)

    # Full sample-box volume as faint bounds context.
    plotter.add_mesh(grid.outline(), color="gray", line_width=1.0)

    # Iso-surface at iso_fraction * peak.
    if peak > 0.0:
        contour_value = iso_fraction * peak
        iso = grid.contour(isosurfaces=[contour_value], scalars="power_density")
        if iso.n_points > 0:
            plotter.add_mesh(
                iso,
                scalars="power_density",
                cmap=get_cmap("temperature"),
                opacity=0.85,
                show_scalar_bar=True,
                scalar_bar_args={"title": "q (W/mm³)"},
            )

    # Optional: underlying mesh for spatial context.
    if mesh is not None:
        from feaweld.core.types import StressField
        from feaweld.visualization.stress_plots import stress_field_to_pyvista
        dummy = StressField(values=np.zeros((mesh.n_nodes, 6)))
        struct = stress_field_to_pyvista(mesh, dummy)
        plotter.add_mesh(struct, style="wireframe", color="gray", opacity=0.2)

    # Arrow showing travel direction.
    centre = source.start_position + source.direction * source.travel_speed * t
    arrow_length = 2.0 * max(source.a_f, source.a_r)
    plotter.add_arrows(
        centre.reshape(1, 3),
        (source.direction * arrow_length).reshape(1, 3),
        mag=1.0,
        color="black",
    )

    plotter.add_axes()

    if screenshot is not None:
        plotter.screenshot(screenshot)
    elif show:
        plotter.show()

    return plotter
