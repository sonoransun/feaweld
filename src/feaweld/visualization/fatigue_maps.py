"""Fatigue life and damage contour plots.

Provides coloured contour maps for fatigue-life fields and Miner's
cumulative damage fields on FEA meshes.  PyVista is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEMesh

#: Upper bound applied to fatigue-life fields before plotting. Stress below
#: the S-N cutoff yields infinite life, which would drive the log10 colour
#: scale to infinity and flatten the map; clipping here preserves contrast.
_LIFE_CAP = 1e12


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

    raw = np.asarray(life_field, dtype=np.float64)
    # Detect top-end clipping (finite values above the cap or +inf) before
    # sanitising, so the scalar bar can flag it.
    capped_high = bool(np.any(raw >= _LIFE_CAP))
    # Clip both ends: the low clamp avoids log(0); the high clamp keeps
    # infinite-life nodes from driving the colour scale to infinity. NaN and
    # +inf are mapped to the cap so the log10 map stays finite.
    life = np.clip(
        np.nan_to_num(raw, nan=_LIFE_CAP, posinf=_LIFE_CAP, neginf=1.0),
        1.0, _LIFE_CAP,
    )
    if log_scale:
        scalar_data = np.log10(life)
        title = "log\u2081\u2080(Life [cycles])"
    else:
        scalar_data = life
        title = "Life [cycles]"
    if capped_high:
        title += f" (capped at {_LIFE_CAP:.0e})"

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

    raw = np.asarray(damage_field, dtype=np.float64)
    # Infinite damage (zero predicted life) or NaN would blow up the colour
    # limits and the max-damage readout. Map +inf/NaN to the largest finite
    # damage present (at least 1.0, so such nodes still colour as failed).
    finite = raw[np.isfinite(raw)]
    fill_high = max(float(np.max(finite)) if finite.size else 1.0, 1.0)
    damage = np.nan_to_num(raw, nan=0.0, posinf=fill_high, neginf=0.0)
    grid.point_data["damage"] = damage

    from feaweld.visualization.theme import get_cmap, configure_plotter

    plotter = pv.Plotter(off_screen=not show)
    configure_plotter(plotter)
    plotter.add_mesh(
        grid,
        scalars="damage",
        cmap=get_cmap("damage"),
        clim=[0.0, max(float(np.max(damage)), 1.0)],
        show_scalar_bar=True,
        scalar_bar_args={"title": "Miner Damage D"},
    )
    plotter.add_axes()

    # Failure warning if any node exceeds D=1.0
    max_damage = float(np.max(damage)) if damage.size else 0.0
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
