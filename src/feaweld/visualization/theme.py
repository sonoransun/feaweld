"""Centralized visual theme for feaweld plots and 3D visualizations.

Provides named colour constants, a semantic colormap registry, and
functions to apply consistent styling to Matplotlib figures and PyVista
plotters.  Import lazily in visualization modules to keep non-viz code
independent of matplotlib/pyvista.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Named colour palette
# ---------------------------------------------------------------------------

FEAWELD_BLUE = "#2980b9"
FEAWELD_RED = "#e74c3c"
FEAWELD_ORANGE = "#f39c12"
FEAWELD_GREEN = "#27ae60"
FEAWELD_DARK = "#1a5276"
FEAWELD_GRAY = "#7f8c8d"
FEAWELD_LIGHT_BG = "#fafafa"

# Severity colours (shared with annotations.py)
SEVERITY_INFO = "#27ae60"
SEVERITY_WARNING = "#f39c12"
SEVERITY_CRITICAL = "#e74c3c"

# ---------------------------------------------------------------------------
# Semantic colormap registry
# ---------------------------------------------------------------------------

_CMAP_REGISTRY: dict[str, str] = {
    "stress": "turbo",
    "temperature": "inferno",
    "fatigue_life": "RdYlBu",
    "damage": "YlOrRd",
    "diverging": "RdBu_r",
    "safety_factor": "RdYlGn",
    "displacement": "viridis",
}


def get_cmap(purpose: str) -> str:
    """Return the colormap name for a semantic *purpose*.

    Raises ``KeyError`` if *purpose* is not registered.
    """
    return _CMAP_REGISTRY[purpose]


# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def apply_feaweld_style() -> None:
    """Set Matplotlib rcParams for a consistent feaweld look.

    Safe to call multiple times; only the first call mutates rcParams.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        import matplotlib as mpl
    except ImportError:
        return

    rc = mpl.rcParams
    rc["figure.facecolor"] = FEAWELD_LIGHT_BG
    rc["axes.facecolor"] = "white"
    rc["axes.titlesize"] = 14
    rc["axes.titleweight"] = "bold"
    rc["axes.labelsize"] = 12
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["grid.linestyle"] = ":"
    rc["grid.alpha"] = 0.4
    rc["legend.framealpha"] = 0.9
    rc["legend.fontsize"] = "small"
    rc["figure.constrained_layout.use"] = False

    _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# PyVista plotter styling
# ---------------------------------------------------------------------------

def configure_plotter(plotter: Any) -> None:
    """Apply a consistent look to a PyVista ``Plotter``.

    Call immediately after ``pv.Plotter(...)`` creation.
    """
    try:
        plotter.set_background("white", top="lightgray")
    except Exception:
        pass
