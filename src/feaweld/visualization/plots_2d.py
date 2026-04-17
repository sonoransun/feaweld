"""Matplotlib-based 2D plotting for feaweld FEA weld analysis.

All matplotlib imports are deferred via :func:`_require_matplotlib` so that
the rest of feaweld remains usable without the optional matplotlib dependency.
Every public function follows a consistent signature:

* ``show: bool = True`` -- call ``plt.show()`` when *True*.
* ``ax: Any = None`` -- reuse an existing Axes; create a new Figure when *None*.
* Returns the :class:`~matplotlib.figure.Figure` that owns the axes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helper -- deferred matplotlib import
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Import and return ``matplotlib.pyplot``, raising a friendly error if absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for 2D plots. "
            "Install with: pip install feaweld[viz]"
        ) from exc


def _von_mises_from_voigt(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Von Mises stress from an (n, 6) Voigt stress array."""
    return np.sqrt(
        0.5 * (
            (s[:, 0] - s[:, 1]) ** 2
            + (s[:, 1] - s[:, 2]) ** 2
            + (s[:, 2] - s[:, 0]) ** 2
            + 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
        )
    )


def _prepare_axes(plt, ax: Any, title: str | None):
    """Return ``(fig, ax)``; create a new figure when *ax* is ``None``."""
    from feaweld.visualization.theme import apply_feaweld_style
    apply_feaweld_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if title:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Through-thickness linearization plot
# ---------------------------------------------------------------------------


def plot_through_thickness(
    result: Any,
    title: str = "Through-Thickness Stress Linearization",
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Plot through-thickness stress decomposition.

    Parameters
    ----------
    result:
        A :class:`~feaweld.postprocess.linearization.LinearizationResult`.
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        An existing :class:`~matplotlib.axes.Axes`; if *None* a new figure
        is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title)

    z = result.z_coords
    total_vm = _von_mises_from_voigt(result.total_stress)

    # Linearized (membrane + bending) von Mises distribution
    linearized = result.linearized_stress  # (n_points, 6)
    linearized_vm = _von_mises_from_voigt(linearized)

    # Membrane: constant scalar across thickness
    membrane_line = np.full_like(z, result.membrane_scalar)

    # Membrane + bending: linear from (membrane - bending) to (membrane + bending)
    mb_line = np.linspace(
        result.membrane_scalar - result.bending_scalar,
        result.membrane_scalar + result.bending_scalar,
        len(z),
    )

    # Total stress -- solid black
    ax.plot(total_vm, z, "k-", linewidth=1.5, label="Total (von Mises)")

    # Membrane -- horizontal dashed blue
    ax.plot(membrane_line, z, "b--", linewidth=1.2, label="Membrane")

    # Membrane + bending -- diagonal dash-dot red
    ax.plot(mb_line, z, "r-.", linewidth=1.2, label="Membrane + Bending")

    # Shaded region between total and linearized (peak stress)
    ax.fill_betweenx(
        z,
        linearized_vm,
        total_vm,
        alpha=0.25,
        color="gray",
        label="Peak (nonlinear)",
    )

    # --- Engineering context ---
    # Reference lines at membrane and membrane+bending scalars
    ax.axvline(result.membrane_scalar, color="blue", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.axvline(result.membrane_plus_bending_scalar, color="red", linestyle=":", alpha=0.4, linewidth=0.8)

    # Decomposition equation box
    ax.text(
        0.98, 0.02,
        "$\\sigma_{total} = \\sigma_m + \\sigma_b + \\sigma_F$",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"),
    )

    # Surface labels
    ax.text(ax.get_xlim()[0], z[0], " Inner", fontsize=7, va="top", color="#555")
    ax.text(ax.get_xlim()[0], z[-1], " Outer", fontsize=7, va="bottom", color="#555")

    ax.set_xlabel("Stress (MPa)")
    ax.set_ylabel("Through-thickness position (mm)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Hot-spot stress extrapolation plot
# ---------------------------------------------------------------------------


def plot_hotspot_extrapolation(
    result: Any,
    title: str = "Hot-Spot Stress Extrapolation",
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Plot hot-spot stress extrapolation from reference points to weld toe.

    Parameters
    ----------
    result:
        A :class:`~feaweld.postprocess.hotspot.HotSpotResult`.
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title)

    ref_d = np.asarray(result.reference_distances, dtype=np.float64)
    ref_s = np.asarray(result.reference_stresses, dtype=np.float64)
    hs = result.hot_spot_stress

    # --- Schematic weld toe profile at x=0 ---
    stress_range = float(np.max(ref_s) - np.min(ref_s))
    y_base = float(np.min(ref_s)) - stress_range * 0.05
    weld_h = stress_range * 0.15
    weld_w = float(np.max(ref_d)) * 0.06
    ax.fill(
        [-weld_w, 0, 0, -weld_w],
        [y_base, y_base, y_base + weld_h, y_base],
        color="#bdc3c7", edgecolor="#7f8c8d", linewidth=1.0, zorder=1,
    )
    ax.text(-weld_w * 0.5, y_base + weld_h * 0.5, "weld\ntoe",
            fontsize=6, ha="center", va="center", color="#555")

    # Reference point markers
    ax.plot(ref_d, ref_s, "bo", markersize=8, label="Reference points")

    # Fit polynomial through reference points
    n_ref = len(ref_d)
    deg = 2 if n_ref >= 3 else 1
    coeffs = np.polyfit(ref_d, ref_s, deg=deg)
    poly = np.poly1d(coeffs)

    # Extrapolation line from weld toe (x=0) to beyond reference points
    x_fit = np.linspace(0, float(np.max(ref_d)) * 1.05, 200)
    y_fit = poly(x_fit)

    ax.plot(x_fit, y_fit, "b--", linewidth=1.2, label="Extrapolation line")

    # Hot-spot stress at weld toe
    ax.plot(0, hs, "r*", markersize=14, label=f"Hot-spot stress = {hs:.1f} MPa")
    ax.annotate(
        f"{hs:.1f} MPa",
        xy=(0, hs),
        xytext=(float(np.max(ref_d)) * 0.15, hs + stress_range * 0.12),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        fontsize=9,
        color="red",
        fontweight="bold",
    )

    # --- Horizontal reference line at hot-spot stress ---
    ax.axhline(hs, color="red", linestyle=":", alpha=0.3, linewidth=0.8)

    # --- IIW reference point distance labels ---
    extrap_type = getattr(result, "extrapolation_type", None)
    type_label = getattr(extrap_type, "value", "") if extrap_type else ""
    if type_label:
        ax.text(
            0.98, 0.98, f"IIW {type_label}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"),
        )

    ax.set_xlabel("Distance from weld toe (mm)")
    ax.set_ylabel("Stress (MPa)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. Dong decomposition (stacked bar + bending ratio)
# ---------------------------------------------------------------------------


def plot_dong_decomposition(
    result: Any,
    title: str = "Dong Structural Stress Decomposition",
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Stacked bar chart of membrane/bending stress with bending ratio overlay.

    Parameters
    ----------
    result:
        A :class:`~feaweld.postprocess.dong.DongResult`.
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title)

    n_nodes = len(result.membrane_stress)
    node_idx = np.arange(n_nodes)

    abs_mem = np.abs(result.membrane_stress)
    abs_bend = np.abs(result.bending_stress)

    from feaweld.visualization.theme import FEAWELD_BLUE, FEAWELD_ORANGE

    # Stacked bars
    bar_width = 0.7
    ax.bar(
        node_idx, abs_mem, width=bar_width,
        color=FEAWELD_BLUE, label="Membrane |$\\sigma_m$|",
    )
    ax.bar(
        node_idx, abs_bend, width=bar_width, bottom=abs_mem,
        color=FEAWELD_ORANGE, label="Bending |$\\sigma_b$|",
    )

    ax.set_xlabel("Node")
    ax.set_ylabel("Stress (MPa)")

    # Secondary y-axis for bending ratio
    ax2 = ax.twinx()
    ax2.plot(
        node_idx, result.bending_ratio,
        "r-o", markersize=4, linewidth=1.2, label="Bending ratio $r$",
    )
    ax2.set_ylabel("Bending ratio $r$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1.05)

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize="small")

    # --- Formula box ---
    ax.text(
        0.02, 0.98,
        "$\\sigma_s = \\sigma_m + \\sigma_b$\n"
        "$r = |\\sigma_b| / (|\\sigma_m| + |\\sigma_b|)$",
        transform=ax.transAxes, fontsize=7,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"),
    )

    ax.set_xticks(node_idx)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 4. S-N curve plot
# ---------------------------------------------------------------------------


def plot_sn_curve(
    curve: Any,
    stress_range: float | None = None,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Log-log S-N curve with optional fatigue-life evaluation point.

    Parameters
    ----------
    curve:
        A :class:`~feaweld.core.types.SNCurve`.
    stress_range:
        If given, compute life and mark on the plot.
    title:
        Plot title.  Defaults to the curve name.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    if title is None:
        title = f"S-N Curve: {curve.name}"
    fig, ax = _prepare_axes(plt, ax, title)

    # Determine overall stress range for plotting
    s_min, s_max = 1.0, 1000.0
    s_values = np.logspace(np.log10(s_min), np.log10(s_max), 500)

    # Plot each segment
    colors = plt.cm.tab10(np.linspace(0, 0.5, max(len(curve.segments), 1)))
    knee_points_s: list[float] = []
    knee_points_n: list[float] = []

    for idx, seg in enumerate(curve.segments):
        # Filter stress values belonging to this segment
        if idx == 0:
            upper_bound = s_max
        else:
            upper_bound = curve.segments[idx - 1].stress_threshold

        lower_bound = seg.stress_threshold

        mask = (s_values >= lower_bound) & (s_values <= upper_bound)
        s_seg = s_values[mask]
        if len(s_seg) == 0:
            continue

        n_seg = seg.C / (s_seg ** seg.m)

        ax.loglog(n_seg, s_seg, "-", linewidth=2.0, color=colors[idx],
                  label=f"Segment {idx + 1} (m={seg.m:.1f})")

        # Mark knee point (transition) at the lower threshold
        if seg.stress_threshold > 0:
            n_knee = seg.C / (seg.stress_threshold ** seg.m)
            knee_points_s.append(seg.stress_threshold)
            knee_points_n.append(n_knee)

    # --- Fatigue regime bands ---
    ax.axvspan(1e0, 1e4, alpha=0.04, color="red", zorder=0)
    ax.axvspan(1e4, 2e6, alpha=0.04, color="orange", zorder=0)
    ax.axvspan(2e6, 1e10, alpha=0.04, color="green", zorder=0)
    ax.text(3e1, s_min * 1.15, "LCF", fontsize=7, color="#888", ha="center")
    ax.text(1e5, s_min * 1.15, "HCF", fontsize=7, color="#888", ha="center")
    ax.text(1e8, s_min * 1.15, "Endurance", fontsize=7, color="#888", ha="center")

    # Draw knee point markers
    for n_k, s_k in zip(knee_points_n, knee_points_s):
        ax.plot(n_k, s_k, "D", color="darkgreen", markersize=8, zorder=5)
        # Vertical CAFL line at knee point
        ax.axvline(n_k, color="darkgreen", linestyle="--", alpha=0.3, linewidth=0.7)
        ax.text(n_k, s_max * 0.85, "CAFL", fontsize=7, color="darkgreen",
                ha="center", rotation=90, alpha=0.6)
    if knee_points_n:
        ax.plot([], [], "D", color="darkgreen", markersize=8, label="Knee point")

    # Mark a specific stress range evaluation
    if stress_range is not None and stress_range > 0:
        life = curve.life(stress_range)
        if np.isfinite(life):
            ax.plot(life, stress_range, "ro", markersize=10, zorder=6,
                    label=f"S={stress_range:.0f} MPa, N={life:.2e}")
            ax.annotate(
                f"N = {life:.2e}",
                xy=(life, stress_range),
                xytext=(life * 3, stress_range * 1.3),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9,
                color="red",
            )

    # --- Standard name annotation ---
    ax.text(
        0.98, 0.98, curve.name,
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"),
    )

    ax.set_xlabel("Fatigue Life $N$ (cycles)")
    ax.set_ylabel("Stress Range $S$ (MPa)")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 5. Stress along a path
# ---------------------------------------------------------------------------


def plot_stress_along_path(
    distances: NDArray[np.float64],
    stress_values: NDArray[np.float64],
    labels: dict[str, float] | None = None,
    title: str = "Stress Along Path",
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Simple x-y line plot of stress versus distance along a path.

    Parameters
    ----------
    distances:
        (n,) array of distance values (mm).
    stress_values:
        (n,) array of stress values (MPa).
    labels:
        Optional dict mapping label names to distance values.  A vertical
        marker and annotation is placed at each labelled position.
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title)

    from feaweld.visualization.theme import FEAWELD_BLUE

    ax.plot(distances, stress_values, color=FEAWELD_BLUE, linewidth=1.5, label="Stress")

    # --- Min / mean / max reference lines ---
    s_min, s_mean, s_max = float(np.min(stress_values)), float(np.mean(stress_values)), float(np.max(stress_values))
    ax.axhline(s_max, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(s_mean, color="#f39c12", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhline(s_min, color="#27ae60", linestyle="--", alpha=0.5, linewidth=0.8)
    x_right = float(np.max(distances))
    ax.text(x_right, s_max, f" max {s_max:.0f}", fontsize=7, va="bottom", color="#e74c3c")
    ax.text(x_right, s_mean, f" mean {s_mean:.0f}", fontsize=7, va="bottom", color="#f39c12")
    ax.text(x_right, s_min, f" min {s_min:.0f}", fontsize=7, va="top", color="#27ae60")

    if labels:
        for name, dist_val in labels.items():
            # Interpolate stress at the labelled distance
            stress_at = float(np.interp(dist_val, distances, stress_values))
            ax.plot(dist_val, stress_at, "ro", markersize=7)
            ax.annotate(
                f"{name}\n{stress_at:.1f} MPa",
                xy=(dist_val, stress_at),
                xytext=(dist_val, stress_at + 0.08 * (s_max - s_min)),
                fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Stress (MPa)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 6. Weld group geometry drawing
# ---------------------------------------------------------------------------


def plot_weld_group_geometry(
    shape: Any,
    d: float,
    b: float = 0.0,
    props: Any = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Draw the weld group outline with dimension annotations.

    Parameters
    ----------
    shape:
        A :class:`~feaweld.core.types.WeldGroupShape` enum value.
    d:
        Primary dimension (mm).
    b:
        Secondary dimension (mm).
    props:
        Optional :class:`~feaweld.core.types.WeldGroupProperties`.  When
        provided a text box with section properties is added.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, f"Weld Group: {shape.value if hasattr(shape, 'value') else shape}")

    # Import WeldGroupShape enum values for comparison
    from feaweld.core.types import WeldGroupShape

    lw = 3.0  # weld line width
    color = "black"

    # Coordinates are centred on the origin where practical.
    # Each shape is drawn with thick lines.
    if shape == WeldGroupShape.LINE:
        # Horizontal line of length d, centred at origin
        ax.plot([-d / 2, d / 2], [0, 0], color=color, linewidth=lw)
        _add_dim_arrow(ax, -d / 2, 0, d / 2, 0, f"d = {d:.0f}")
        cx, cy = 0.0, 0.0

    elif shape == WeldGroupShape.PARALLEL:
        # Two horizontal lines of length d, separated by b
        ax.plot([-d / 2, d / 2], [-b / 2, -b / 2], color=color, linewidth=lw)
        ax.plot([-d / 2, d / 2], [b / 2, b / 2], color=color, linewidth=lw)
        _add_dim_arrow(ax, -d / 2, -b / 2, d / 2, -b / 2, f"d = {d:.0f}")
        _add_dim_arrow(ax, d / 2 + d * 0.05, -b / 2, d / 2 + d * 0.05, b / 2, f"b = {b:.0f}")
        cx, cy = 0.0, 0.0

    elif shape == WeldGroupShape.C_SHAPE:
        # Channel open right: web on left, flanges top/bottom
        # Bottom flange: (0, 0) to (b, 0)
        # Top flange: (0, d) to (b, d)
        # Web: (0, 0) to (0, d)
        ax.plot([0, b], [0, 0], color=color, linewidth=lw)         # bottom flange
        ax.plot([0, b], [d, d], color=color, linewidth=lw)         # top flange
        ax.plot([0, 0], [0, d], color=color, linewidth=lw)         # web
        _add_dim_arrow(ax, 0, -d * 0.08, b, -d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, -b * 0.12, 0, -b * 0.12, d, f"d = {d:.0f}")
        x_bar = b ** 2 / (2.0 * b + d)
        cx, cy = x_bar, d / 2.0

    elif shape == WeldGroupShape.L_SHAPE:
        # Angle: bottom horizontal leg (length b) + left vertical leg (height d)
        # Corner at origin
        ax.plot([0, b], [0, 0], color=color, linewidth=lw)   # horizontal
        ax.plot([0, 0], [0, d], color=color, linewidth=lw)   # vertical
        _add_dim_arrow(ax, 0, -d * 0.08, b, -d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, -b * 0.12, 0, -b * 0.12, d, f"d = {d:.0f}")
        cx = b ** 2 / (2.0 * (b + d))
        cy = d ** 2 / (2.0 * (b + d))

    elif shape == WeldGroupShape.BOX:
        # Rectangle: width b, depth d, centred at origin
        hb, hd = b / 2.0, d / 2.0
        rect_x = [-hb, hb, hb, -hb, -hb]
        rect_y = [-hd, -hd, hd, hd, -hd]
        ax.plot(rect_x, rect_y, color=color, linewidth=lw)
        _add_dim_arrow(ax, -hb, -hd - d * 0.08, hb, -hd - d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, hb + b * 0.08, -hd, hb + b * 0.08, hd, f"d = {d:.0f}")
        cx, cy = 0.0, 0.0

    elif shape == WeldGroupShape.CIRCULAR:
        # Circle of diameter d centred at origin
        theta = np.linspace(0, 2 * np.pi, 200)
        r = d / 2.0
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=color, linewidth=lw)
        _add_dim_arrow(ax, -r, 0, r, 0, f"d = {d:.0f}")
        cx, cy = 0.0, 0.0

    elif shape == WeldGroupShape.I_SHAPE:
        # Two horizontal flanges of width b, separated by depth d
        # Centred at origin
        hb, hd = b / 2.0, d / 2.0
        ax.plot([-hb, hb], [-hd, -hd], color=color, linewidth=lw)  # bottom flange
        ax.plot([-hb, hb], [hd, hd], color=color, linewidth=lw)    # top flange
        _add_dim_arrow(ax, -hb, -hd - d * 0.08, hb, -hd - d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, hb + b * 0.08, -hd, hb + b * 0.08, hd, f"d = {d:.0f}")
        cx, cy = 0.0, 0.0

    elif shape == WeldGroupShape.T_SHAPE:
        # Bottom flange (length b) + web going up from centre (height d)
        # Flange centred on x-axis at y = 0
        hb = b / 2.0
        ax.plot([-hb, hb], [0, 0], color=color, linewidth=lw)  # flange
        ax.plot([0, 0], [0, d], color=color, linewidth=lw)      # web
        _add_dim_arrow(ax, -hb, -d * 0.08, hb, -d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, b * 0.08, 0, b * 0.08, d, f"d = {d:.0f}")
        cx = 0.0
        cy = d ** 2 / (2.0 * (b + d))

    elif shape == WeldGroupShape.U_SHAPE:
        # U-shape (open top): bottom flange (b) + two webs (d)
        # Bottom flange at y = 0 from x = 0 to x = b
        # Left web: (0, 0) to (0, d)
        # Right web: (b, 0) to (b, d)
        ax.plot([0, b], [0, 0], color=color, linewidth=lw)     # bottom
        ax.plot([0, 0], [0, d], color=color, linewidth=lw)     # left web
        ax.plot([b, b], [0, d], color=color, linewidth=lw)     # right web
        _add_dim_arrow(ax, 0, -d * 0.08, b, -d * 0.08, f"b = {b:.0f}")
        _add_dim_arrow(ax, -b * 0.12, 0, -b * 0.12, d, f"d = {d:.0f}")
        cx = b / 2.0
        cy = d ** 2 / (b + 2.0 * d)

    else:
        raise ValueError(f"Unsupported weld group shape: {shape}")

    # Mark centroid with coordinates
    ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2.5, zorder=5, label="Centroid")
    ax.text(cx, cy, f"  ({cx:.1f}, {cy:.1f})", fontsize=7, color="red", va="top")

    # Properties text box
    if props is not None:
        text = (
            f"$A_w$ = {props.A_w:.1f} mm\n"
            f"$S_x$ = {props.S_x:.1f} mm$^2$\n"
            f"$S_y$ = {props.S_y:.1f} mm$^2$\n"
            f"$J_w$ = {props.J_w:.1f} mm$^3$"
        )
        ax.text(
            0.98, 0.02, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def _add_dim_arrow(
    ax: Any,
    x0: float, y0: float,
    x1: float, y1: float,
    label: str,
) -> None:
    """Draw a two-headed dimension arrow with a centred label."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0),
    )
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    ax.text(
        mid_x, mid_y, label,
        fontsize=8, ha="center", va="bottom",
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )


# ---------------------------------------------------------------------------
# 7. ASME stress categorization check
# ---------------------------------------------------------------------------


def plot_asme_check(
    categorization: Any,
    S_m: float,
    S_y: float,
    title: str = "ASME VIII Div 2 Stress Check",
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Horizontal bar chart comparing stress categories against ASME limits.

    Parameters
    ----------
    categorization:
        A :class:`~feaweld.postprocess.nominal.StressCategorization`.
    S_m:
        Allowable stress intensity (MPa).
    S_y:
        Yield strength (MPa).
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title)

    # Define the three checks
    categories = ["Pm", "Pm+Pb", "PL+Pb+Q"]
    values = [
        abs(categorization.membrane),
        abs(categorization.membrane) + abs(categorization.bending),
        categorization.total,
    ]
    limits = [
        S_m,
        1.5 * S_m,
        max(3.0 * S_m, 2.0 * S_y),
    ]

    import matplotlib.colors as mcolors

    y_pos = np.arange(len(categories))
    limit_eqs = [
        f"$\\leq$ 1.0 $S_m$ = {limits[0]:.0f}",
        f"$\\leq$ 1.5 $S_m$ = {limits[1]:.0f}",
        f"$\\leq$ max(3$S_m$, 2$S_y$) = {limits[2]:.0f}",
    ]

    # Gradient colors based on utilization ratio (green->yellow->red)
    utilization_cmap = plt.cm.RdYlGn_r  # red=high, green=low
    bar_colors = []
    for v, lim in zip(values, limits):
        ratio = min(v / lim, 1.5) if lim > 0 else 1.5
        bar_colors.append(utilization_cmap(ratio / 1.5))

    ax.barh(y_pos, values, color=bar_colors, height=0.5, alpha=0.85, edgecolor="black")

    # Draw vertical dashed limit lines and annotate ratios + limit equations
    for i, (val, lim) in enumerate(zip(values, limits)):
        ax.axvline(lim, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ratio = val / lim if lim > 0 else float("inf")
        status = "PASS" if val < lim else "FAIL"
        status_color = "darkgreen" if val < lim else "darkred"
        ax.text(
            val + max(values) * 0.02, i,
            f"{ratio:.0%}  {status}",
            va="center", fontsize=9, fontweight="bold",
            color=status_color,
        )
        # Limit equation to the right of the limit line
        ax.text(
            lim, i + 0.30, limit_eqs[i],
            va="bottom", fontsize=7, color="gray", ha="center",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Stress (MPa)")
    ax.set_xlim(0, max(max(values), max(limits)) * 1.35)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 8. Cross-section stress contour
# ---------------------------------------------------------------------------


def plot_cross_section_stress(
    mesh: Any,
    stress: Any,
    y_level: float,
    component: str = "von_mises",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Plot a 2D cross-section stress contour at a given y-level.

    Finds nodes within a tolerance of *y_level*, projects them onto the
    x-z plane, and plots a filled contour coloured by the requested stress
    component.

    Parameters
    ----------
    mesh:
        A :class:`~feaweld.core.types.FEMesh`.
    stress:
        A :class:`~feaweld.core.types.StressField`.
    y_level:
        Y-coordinate of the cross-section slice (mm).
    component:
        Stress component to plot.  One of ``"von_mises"``, ``"tresca"``,
        ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ``"xz"``.
    title:
        Plot title.
    show:
        Call ``plt.show()`` when *True*.
    ax:
        Existing axes or *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from matplotlib.tri import Triangulation

    if title is None:
        title = f"Cross-Section Stress ({component}) at y = {y_level:.1f} mm"
    fig, ax = _prepare_axes(plt, ax, title)

    # Tolerance: fraction of mesh extent in y direction
    y_coords = mesh.nodes[:, 1]
    y_extent = float(np.ptp(y_coords))
    tol = max(y_extent * 0.02, 0.5)  # at least 0.5 mm

    # Find nodes near the y-level
    mask = np.abs(y_coords - y_level) <= tol
    node_indices = np.where(mask)[0]

    if len(node_indices) < 3:
        ax.text(
            0.5, 0.5, f"No nodes found near y = {y_level:.1f} mm",
            transform=ax.transAxes, ha="center", va="center", fontsize=12,
        )
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    x = mesh.nodes[node_indices, 0]
    z = mesh.nodes[node_indices, 2]

    # Extract stress component
    _COMP_INDEX = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}
    if component == "von_mises":
        values = stress.von_mises[node_indices]
    elif component == "tresca":
        values = stress.tresca[node_indices]
    elif component in _COMP_INDEX:
        values = stress.values[node_indices, _COMP_INDEX[component]]
    else:
        raise ValueError(
            f"Unknown stress component '{component}'. "
            f"Choose from: von_mises, tresca, {', '.join(_COMP_INDEX)}"
        )

    # Create Delaunay triangulation and plot filled contour
    triang = Triangulation(x, z)
    n_levels = 20
    from feaweld.visualization.theme import get_cmap
    contour = ax.tricontourf(triang, values, levels=n_levels, cmap=get_cmap("stress"))
    # Overlay contour lines with labels at key stress levels
    n_line_levels = min(8, n_levels)
    cs = ax.tricontour(triang, values, levels=n_line_levels, colors="black", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    fig.colorbar(contour, ax=ax, label=f"{component.replace('_', ' ').title()} (MPa)")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_aspect("equal")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Mesh convergence (Richardson / GCI)
# ---------------------------------------------------------------------------


def plot_mesh_convergence(
    result: Any,
    *,
    quantity_label: str = "quantity of interest",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
):
    """Plot mesh refinement convergence from a :class:`ConvergenceResult`.

    Produces a log-log plot of the quantity of interest versus element
    size across the refinement levels, with the Richardson-extrapolated
    zero-mesh-size value drawn as a horizontal asymptote and the Grid
    Convergence Index (GCI) band around the finest-mesh value.

    Parameters
    ----------
    result : feaweld.singularity.convergence.ConvergenceResult
        Output of
        :func:`feaweld.singularity.convergence.convergence_study`.
    quantity_label : str
        Label for the y-axis (e.g. ``"max von Mises stress (MPa)"``).
    title : str, optional
        Figure title.
    show : bool
        Call ``plt.show()`` when ``True``.
    ax : matplotlib Axes, optional
        Reuse an existing Axes. A new Figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.theme import (
        apply_feaweld_style, FEAWELD_BLUE, FEAWELD_RED, FEAWELD_GREEN,
    )

    plt = _require_matplotlib()
    apply_feaweld_style()

    sizes = np.asarray(result.mesh_sizes, dtype=float)
    values = np.asarray(result.stress_values, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Data points — coarsest (large size) to finest (small size)
    ax.plot(sizes, values, marker="o", markersize=7, linewidth=1.6,
            color=FEAWELD_BLUE, label="FEA result")

    # Richardson-extrapolated zero-mesh-size value
    extrap = float(result.extrapolated_value)
    ax.axhline(extrap, color=FEAWELD_GREEN, linestyle="--", linewidth=1.3,
               label=f"Richardson extrapolation = {extrap:.3g}")

    # GCI band around the finest-mesh value
    finest_value = values[int(np.argmin(sizes))]
    gci = float(result.gci)
    band = abs(finest_value) * gci
    ax.fill_between(
        sizes,
        finest_value - band,
        finest_value + band,
        color=FEAWELD_RED, alpha=0.15,
        label=f"GCI band ±{gci * 100:.1f} %",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Element size h (mm)")
    ax.set_ylabel(quantity_label)
    ax.invert_xaxis()  # finer meshes at the right
    ax.grid(True, which="both", alpha=0.3)

    status = "converged" if result.is_converged else "not converged"
    p = result.convergence_order
    order_str = f"order p ≈ {p:.2f}" if np.isfinite(p) else "order p → ∞"
    ax.set_title(
        title
        or f"Mesh convergence ({status}, {order_str}, GCI = {gci * 100:.2f} %)"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
