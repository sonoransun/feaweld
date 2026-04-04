"""Engineering annotation helpers for 2D and 3D visualizations.

Identifies critical points (max stress, weld toes, singularities) and
provides functions to annotate both Matplotlib axes and PyVista plotters
with severity-coded labels and markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEMesh, StressField, WeldLineDefinition


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CriticalPoint:
    """A point of engineering significance to annotate on plots."""
    location: NDArray[np.float64]  # (3,) coordinates
    value: float                    # stress / damage / life value
    label: str                      # e.g. "Max VM: 245.3 MPa"
    severity: str = "info"          # "info", "warning", "critical"
    category: str = "stress"        # "stress", "singularity", "weld_toe", "fatigue"


_SEVERITY_COLORS = {
    "info": "#27ae60",       # green
    "warning": "#f39c12",    # orange
    "critical": "#e74c3c",   # red
}

_SEVERITY_MARKERS = {
    "info": "o",
    "warning": "^",
    "critical": "X",
}


# ---------------------------------------------------------------------------
# Critical point detection
# ---------------------------------------------------------------------------

def find_critical_points(
    mesh: FEMesh,
    stress: StressField,
    n_max: int = 5,
    weld_line: WeldLineDefinition | None = None,
    allowable: float | None = None,
) -> list[CriticalPoint]:
    """Identify the most significant locations in the stress field.

    Returns up to *n_max* critical points sorted by severity, including:
    - Global max von Mises stress location(s)
    - Weld toe peak stress (if *weld_line* provided)
    - Locations exceeding *allowable* (if given)

    Args:
        mesh: Finite-element mesh.
        stress: Stress field on the mesh.
        n_max: Maximum number of critical points to return.
        weld_line: Optional weld line for toe-specific checks.
        allowable: Optional allowable stress (MPa) for pass/fail marking.

    Returns:
        List of CriticalPoint sorted by descending severity then value.
    """
    points: list[CriticalPoint] = []
    vm = stress.von_mises

    # --- Global max stress ---
    top_indices = np.argsort(vm)[::-1][:n_max]
    for rank, idx in enumerate(top_indices):
        val = float(vm[idx])
        loc = mesh.nodes[idx].copy()

        if allowable is not None and val > allowable:
            severity = "critical"
        elif rank == 0:
            severity = "warning"
        else:
            severity = "info"

        label = format_engineering_value(val, "MPa")
        if rank == 0:
            label = f"Max \u03c3_vm: {label}"
        else:
            label = f"\u03c3_vm: {label}"

        points.append(CriticalPoint(
            location=loc,
            value=val,
            label=label,
            severity=severity,
            category="stress",
        ))

    # --- Weld toe peaks ---
    if weld_line is not None:
        toe_vm = vm[weld_line.node_ids]
        toe_max_idx = int(np.argmax(toe_vm))
        toe_node = weld_line.node_ids[toe_max_idx]
        toe_val = float(toe_vm[toe_max_idx])
        toe_loc = mesh.nodes[toe_node].copy()

        severity = "warning"
        if allowable is not None and toe_val > allowable:
            severity = "critical"

        points.append(CriticalPoint(
            location=toe_loc,
            value=toe_val,
            label=f"Weld toe: {format_engineering_value(toe_val, 'MPa')}",
            severity=severity,
            category="weld_toe",
        ))

    # Sort: critical first, then warning, then info; within severity by value desc
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    points.sort(key=lambda p: (severity_order.get(p.severity, 3), -p.value))

    return points[:n_max]


def singularity_warning_markers(
    singularities: list,  # list[SingularityInfo] from feaweld.singularity.detection
    mesh: FEMesh,
) -> list[CriticalPoint]:
    """Convert singularity detection results into annotation markers."""
    points = []
    for s in singularities:
        if not getattr(s, "is_singular", False):
            continue
        nid = getattr(s, "node_id", 0)
        val = getattr(s, "stress_value", 0.0)
        loc = mesh.nodes[nid].copy() if nid < mesh.n_nodes else np.zeros(3)
        points.append(CriticalPoint(
            location=loc,
            value=val,
            label=f"Singularity: {format_engineering_value(val, 'MPa')}",
            severity="critical",
            category="singularity",
        ))
    return points


# ---------------------------------------------------------------------------
# Computed fields
# ---------------------------------------------------------------------------

def safety_factor_field(
    mesh: FEMesh,
    stress: StressField,
    allowable: float,
) -> NDArray[np.float64]:
    """Compute nodal safety factor: SF = allowable / von_mises.

    Values > 1 are safe, < 1 indicate failure.
    Clamped to [0, 10] for visualization.
    """
    vm = stress.von_mises
    sf = np.where(vm > 1e-12, allowable / vm, 10.0)
    return np.clip(sf, 0.0, 10.0)


# ---------------------------------------------------------------------------
# Annotation rendering
# ---------------------------------------------------------------------------

def annotate_3d(
    plotter: Any,
    points: list[CriticalPoint],
    font_size: int = 12,
) -> None:
    """Add labeled markers to a PyVista plotter.

    Color-codes by severity: green = info, orange = warning, red = critical.
    """
    import pyvista as pv

    if not points:
        return

    coords = np.array([p.location for p in points])
    labels = [p.label for p in points]
    colors = [_SEVERITY_COLORS.get(p.severity, "#888888") for p in points]

    # Add point cloud
    cloud = pv.PolyData(coords)
    plotter.add_mesh(
        cloud,
        color="black",
        point_size=10,
        render_points_as_spheres=True,
    )

    # Add labels with individual colors
    for i, pt in enumerate(points):
        plotter.add_point_labels(
            pv.PolyData(pt.location.reshape(1, 3)),
            [pt.label],
            font_size=font_size,
            text_color=_SEVERITY_COLORS.get(pt.severity, "#333333"),
            point_size=0,
            shape_opacity=0.7,
            always_visible=True,
        )


def annotate_2d(
    ax: Any,
    points: list[CriticalPoint],
    font_size: int = 10,
) -> None:
    """Add annotations to a Matplotlib axes.

    Uses arrows pointing from label to location, severity color coded.
    Only annotates points whose (x, y) position is within the axes limits.
    """
    for pt in points:
        color = _SEVERITY_COLORS.get(pt.severity, "#333333")
        marker = _SEVERITY_MARKERS.get(pt.severity, "o")
        x, y = pt.location[0], pt.location[1]

        ax.plot(x, y, marker=marker, color=color, markersize=8, zorder=5)
        ax.annotate(
            pt.label,
            xy=(x, y),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=font_size,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=6,
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_engineering_value(value: float, unit: str, precision: int = 1) -> str:
    """Format a value with engineering notation for readability.

    Examples:
        format_engineering_value(245.3, "MPa") -> "245.3 MPa"
        format_engineering_value(1.23e6, "cycles") -> "1.23e+06 cycles"
        format_engineering_value(0.00123, "mm") -> "0.0 mm"
    """
    if abs(value) >= 1e5 or (abs(value) < 0.01 and value != 0):
        return f"{value:.{precision}e} {unit}"
    return f"{value:.{precision}f} {unit}"
