"""Fatigue life and damage contour plots.

Provides coloured contour maps for fatigue-life fields and Miner's
cumulative damage fields on FEA meshes.  PyVista is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEMesh


def plot_fatigue_life(
    mesh: FEMesh,
    life_field: NDArray,
    log_scale: bool = True,
    show: bool = True,
) -> Any:
    """Colour-map of predicted fatigue life on the mesh.

    Low-life regions are shown in red (critical) and high-life regions in
    blue (safe).

    Parameters
    ----------
    mesh:
        The FE mesh.
    life_field:
        Fatigue life at each node (cycles), shape ``(n_nodes,)``.
    log_scale:
        If *True* (default) the colour scale uses log10 of life.
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

    grid = _mesh_to_grid(mesh)

    life = np.asarray(life_field, dtype=np.float64)
    if log_scale:
        # Clamp to avoid log(0)
        life_safe = np.clip(life, 1.0, None)
        scalar_data = np.log10(life_safe)
        title = "log\u2081\u2080(Life [cycles])"
    else:
        scalar_data = life
        title = "Life [cycles]"

    grid.point_data["fatigue_life"] = scalar_data

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)
    plotter.add_mesh(
        grid,
        scalars="fatigue_life",
        cmap=get_cmap("fatigue_life"),
        show_scalar_bar=True,
        scalar_bar_args={"title": title},
    )
    plotter.add_axes()

    # Color interpretation annotation
    plotter.add_text(
        "Red = short life (critical)    Blue = long life (safe)",
        position="lower_left", font_size=8, color="black",
    )

    if show:
        plotter.show()
    return plotter


def plot_damage(
    mesh: FEMesh,
    damage_field: NDArray,
    show: bool = True,
) -> Any:
    """Contour plot of Miner's cumulative damage.

    Damage >= 1.0 indicates failure.

    Parameters
    ----------
    mesh:
        The FE mesh.
    damage_field:
        Damage at each node, shape ``(n_nodes,)``.
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

    grid = _mesh_to_grid(mesh)
    grid.point_data["damage"] = np.asarray(damage_field, dtype=np.float64)

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)
    plotter.add_mesh(
        grid,
        scalars="damage",
        cmap=get_cmap("damage"),
        clim=[0.0, max(float(np.max(damage_field)), 1.0)],
        show_scalar_bar=True,
        scalar_bar_args={"title": "Miner Damage D"},
    )
    plotter.add_axes()

    # Failure warning if any node exceeds D=1.0
    max_damage = float(np.max(damage_field))
    if max_damage >= 1.0:
        plotter.add_text(
            f"D \u2265 1.0 \u2014 FAILURE (max D = {max_damage:.2f})",
            position="upper_left", font_size=10, color="red",
        )

    if show:
        plotter.show()
    return plotter


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mesh_to_grid(mesh: FEMesh) -> Any:
    """Create a PyVista UnstructuredGrid from an FEMesh."""
    # Re-use the mapping from stress_plots to avoid duplication.
    from feaweld.visualization.stress_plots import _mesh_to_pyvista_grid

    return _mesh_to_pyvista_grid(mesh)
