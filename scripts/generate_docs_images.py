#!/usr/bin/env python3
"""Generate all documentation images for feaweld.

Produces SVG files in docs/images/ for:
  - Conceptual diagrams explaining engineering methods
  - Example output gallery from actual feaweld visualization functions

Usage:
    python scripts/generate_docs_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure feaweld is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OUTPUT_DIR = _ROOT / "docs" / "images"

# Theme colours (duplicated to keep script standalone if theme isn't installed)
BLUE = "#2980b9"
RED = "#e74c3c"
ORANGE = "#f39c12"
GREEN = "#27ae60"
DARK = "#1a5276"
GRAY = "#7f8c8d"
LIGHT_BG = "#fafafa"

_STAGE_COLORS = [BLUE, "#16a085", "#8e44ad", RED, ORANGE, GREEN, DARK]


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": LIGHT_BG,
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "font.size": 10,
    })
    return plt


# ============================================================================
# Conceptual Diagrams
# ============================================================================

def generate_pipeline_diagram(out: Path) -> None:
    """Pipeline flow diagram: YAML -> ... -> Report."""
    plt = _setup_matplotlib()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")
    fig.patch.set_facecolor(LIGHT_BG)

    stages = [
        ("YAML\nConfig", 0),
        ("Geometry\n(Gmsh)", 2),
        ("Mesh", 4),
        ("Solve\n(FEniCSx /\nCalculiX)", 6),
        ("Post-\nProcess", 8),
        ("Fatigue", 10),
        ("Report\n(HTML)", 12),
    ]

    box_w, box_h = 1.6, 1.4
    for i, (label, x) in enumerate(stages):
        color = _STAGE_COLORS[i % len(_STAGE_COLORS)]
        box = FancyBboxPatch(
            (x - box_w / 2, -box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.9, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, 0, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + box_w / 2
        x1 = stages[i + 1][1] - box_w / 2
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))

    # Parametric study branch
    ax.annotate("Parametric\nStudy", xy=(8, -1.2), xytext=(6, -1.2),
                fontsize=8, ha="center", va="center", color=GRAY,
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, ls="--"))
    ax.text(9.5, -1.2, "Concurrent\ncases", fontsize=7, ha="center", color=GRAY,
            style="italic")

    fig.tight_layout()
    fig.savefig(out / "pipeline_overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_hotspot_concept(out: Path) -> None:
    """Hot-spot stress extrapolation conceptual diagram."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Hot-Spot Stress Extrapolation (IIW)", fontweight="bold")

    # Plate profile (bottom)
    plate_y = 0
    ax.fill_between([-2, 15], plate_y - 0.8, plate_y, color="#dfe6e9", edgecolor=GRAY)
    ax.text(7.5, plate_y - 0.4, "Base plate", fontsize=9, ha="center", color="#555")

    # Weld bead at x=0
    weld_x = [0, -1.2, -0.3, 0]
    weld_y = [plate_y, plate_y, plate_y + 1.8, plate_y + 2.0]
    ax.fill(weld_x, weld_y, color="#bdc3c7", edgecolor=GRAY, linewidth=1.5, hatch="//")
    ax.text(-0.8, plate_y + 0.9, "Weld", fontsize=8, ha="center", color="#555")

    # Stress distribution curve (rising near weld toe)
    x_stress = np.linspace(0.1, 14, 200)
    # Notch stress: exponential rise near weld toe + nominal
    nominal = 80
    notch_peak = 280
    stress = nominal + (notch_peak - nominal) * np.exp(-x_stress / 1.5)

    y_offset = plate_y + 2.5
    scale = 0.02
    ax.plot(x_stress, y_offset + stress * scale, color=DARK, linewidth=2, label="Actual stress")

    # Reference points (Type A: 0.4t and 1.0t at t=10mm)
    t = 10
    ref_dists = [0.4 * t, 1.0 * t]
    ref_stresses = [nominal + (notch_peak - nominal) * np.exp(-d / 1.5) for d in ref_dists]

    for d, s in zip(ref_dists, ref_stresses):
        ax.plot(d, y_offset + s * scale, "bo", markersize=10, zorder=5)
        ax.text(d, y_offset + s * scale + 0.25, f"{d:.0f} mm\n({s:.0f} MPa)",
                fontsize=7, ha="center", va="bottom", color=BLUE)

    # Extrapolation line
    coeffs = np.polyfit(ref_dists, ref_stresses, 1)
    x_ext = np.linspace(0, ref_dists[-1] * 1.1, 100)
    y_ext = np.polyval(coeffs, x_ext)
    ax.plot(x_ext, y_offset + y_ext * scale, "--", color=BLUE, linewidth=1.5,
            label="Linear extrapolation")

    # Hot-spot stress at x=0
    hs = np.polyval(coeffs, 0)
    ax.plot(0, y_offset + hs * scale, "r*", markersize=16, zorder=6)
    ax.annotate(f"Hot-spot stress\n= {hs:.0f} MPa", xy=(0, y_offset + hs * scale),
                xytext=(4, y_offset + hs * scale + 1.5),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                fontsize=10, fontweight="bold", color=RED)

    # Nominal stress line
    ax.axhline(y_offset + nominal * scale, color=GREEN, linestyle=":", alpha=0.6, linewidth=1)
    ax.text(13, y_offset + nominal * scale + 0.15, f"Nominal = {nominal} MPa",
            fontsize=8, color=GREEN, ha="right")

    # Labels
    ax.annotate("", xy=(0, plate_y - 1.2), xytext=(0, plate_y + 2.0),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, ls=":"))
    ax.text(0, plate_y - 1.4, "Weld toe", fontsize=8, ha="center", color=GRAY)

    # Type A / Type B note
    ax.text(0.02, 0.02,
            "Type A: 0.4t, 1.0t (linear)\n"
            "Type B: 4, 8, 12 mm (quadratic)",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"))

    ax.set_xlabel("Distance from weld toe (mm)")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlim(-2.5, 15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hotspot_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_linearization_concept(out: Path) -> None:
    """Through-thickness stress linearization conceptual diagram."""
    plt = _setup_matplotlib()

    fig, (ax_plate, ax_stress) = plt.subplots(1, 2, figsize=(12, 5),
                                               gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("Through-Thickness Stress Linearization (ASME VIII Div 2)", fontweight="bold")

    # Left panel: plate cross-section schematic
    ax_plate.set_title("Cross-Section", fontsize=11)
    plate_w, plate_h = 4, 10
    ax_plate.fill(
        [-plate_w / 2, plate_w / 2, plate_w / 2, -plate_w / 2],
        [0, 0, plate_h, plate_h],
        color="#dfe6e9", edgecolor=GRAY, linewidth=2,
    )
    # Linearization path arrow
    ax_plate.annotate("", xy=(0, plate_h), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax_plate.text(0.5, plate_h / 2, "SCL", fontsize=9, color=RED, fontweight="bold",
                  rotation=90, va="center")
    ax_plate.text(0, -0.5, "Inner", fontsize=9, ha="center", color="#555")
    ax_plate.text(0, plate_h + 0.5, "Outer", fontsize=9, ha="center", color="#555")
    ax_plate.set_xlim(-3, 3)
    ax_plate.set_ylim(-1.5, plate_h + 1.5)
    ax_plate.set_aspect("equal")
    ax_plate.axis("off")

    # Right panel: stress decomposition
    ax_stress.set_title("Stress Decomposition", fontsize=11)
    z = np.linspace(0, plate_h, 100)

    # Actual stress (wavy)
    sigma_m = 120
    sigma_b = 40
    actual = sigma_m + sigma_b * (2 * z / plate_h - 1) + 15 * np.sin(4 * np.pi * z / plate_h)

    # Components
    membrane = np.full_like(z, sigma_m)
    mb = sigma_m + sigma_b * (2 * z / plate_h - 1)

    ax_stress.plot(actual, z, "k-", linewidth=2, label="$\\sigma_{total}$ (actual)")
    ax_stress.plot(membrane, z, "b--", linewidth=1.5, label="$\\sigma_m$ (membrane)")
    ax_stress.plot(mb, z, "r-.", linewidth=1.5, label="$\\sigma_m + \\sigma_b$ (linearized)")
    ax_stress.fill_betweenx(z, mb, actual, alpha=0.2, color="gray", label="$\\sigma_F$ (peak)")

    # Annotations
    ax_stress.axvline(sigma_m, color="blue", linestyle=":", alpha=0.3)
    ax_stress.text(sigma_m + 2, plate_h * 0.15, f"$\\sigma_m$ = {sigma_m}", fontsize=8, color="blue")

    # Equation box
    ax_stress.text(
        0.98, 0.02,
        "$\\sigma_{total} = \\sigma_m + \\sigma_b + \\sigma_F$\n\n"
        "$\\sigma_m$ = membrane (average)\n"
        "$\\sigma_b$ = bending (linear)\n"
        "$\\sigma_F$ = peak (remainder)",
        transform=ax_stress.transAxes, fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", alpha=0.95, edgecolor="#bdc3c7"),
    )

    ax_stress.set_xlabel("Stress (MPa)")
    ax_stress.set_ylabel("Through-thickness position (mm)")
    ax_stress.legend(loc="upper left", fontsize=9)
    ax_stress.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / "linearization_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_goldak_concept(out: Path) -> None:
    """Goldak double-ellipsoid heat source diagram."""
    plt = _setup_matplotlib()
    from matplotlib.patches import Ellipse
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Goldak Double-Ellipsoid Heat Source Model", fontweight="bold")

    # Draw the two half-ellipsoids as overlapping ellipses (2D representation)
    # Front ellipsoid (smaller)
    af, ar, b_param, c = 6, 12, 8, 5

    # Rear (larger, lighter)
    rear = Ellipse((0, 0), 2 * ar, 2 * b_param, angle=0,
                    facecolor="#ffeaa7", edgecolor=ORANGE, linewidth=2, alpha=0.5, ls="--")
    ax.add_patch(rear)

    # Front (smaller, brighter)
    front = Ellipse((0, 0), 2 * af, 2 * b_param, angle=0,
                     facecolor="#fab1a0", edgecolor=RED, linewidth=2, alpha=0.6)
    ax.add_patch(front)

    # Heat intensity gradient (center dot)
    for r_frac in [0.7, 0.5, 0.3, 0.1]:
        circle = plt.Circle((0, 0), af * r_frac, color=RED,
                             alpha=0.15 + 0.2 * (1 - r_frac))
        ax.add_patch(circle)

    # Center point
    ax.plot(0, 0, "ko", markersize=6, zorder=5)
    ax.text(0.5, 0.5, "q_max", fontsize=9, color=DARK, fontweight="bold")

    # Dimension arrows
    # a_f (front semi-axis)
    ax.annotate("", xy=(af, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
    ax.text(af / 2, -0.8, "$a_f$", fontsize=11, ha="center", color=DARK, fontweight="bold")

    # a_r (rear semi-axis)
    ax.annotate("", xy=(-ar, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
    ax.text(-ar / 2, -0.8, "$a_r$", fontsize=11, ha="center", color=DARK, fontweight="bold")

    # b (width)
    ax.annotate("", xy=(0, b_param), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    ax.text(0.8, b_param / 2, "$b$", fontsize=11, color=BLUE, fontweight="bold")

    # Travel direction arrow
    ax.annotate("Travel direction", xy=(af + 3, 0), xytext=(af + 1, 0),
                fontsize=10, ha="left", va="center", color=GRAY, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))

    # Labels
    ax.text(af * 0.6, b_param * 0.55, "Front\nhalf", fontsize=8, ha="center",
            color=RED, alpha=0.8, style="italic")
    ax.text(-ar * 0.5, b_param * 0.55, "Rear\nhalf", fontsize=8, ha="center",
            color=ORANGE, alpha=0.8, style="italic")

    # Power equation
    ax.text(0.02, 0.02,
            "$q(x,y,z) = \\frac{6\\sqrt{3}\\,f\\,Q}{abc\\pi\\sqrt{\\pi}}"
            "\\exp\\left(-\\frac{3x^2}{a^2} - \\frac{3y^2}{b^2} - \\frac{3z^2}{c^2}\\right)$",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#bdc3c7"))

    ax.set_xlim(-ar - 3, af + 8)
    ax.set_ylim(-b_param - 2, b_param + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "goldak_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_joint_types(out: Path) -> None:
    """All 5 joint type cross-section schematics."""
    plt = _setup_matplotlib()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Parametric Weld Joint Types", fontsize=16, fontweight="bold")

    t = 10  # plate thickness
    w = 7   # weld leg size

    joint_data = [
        ("Fillet T-Joint", _draw_fillet_t),
        ("Butt Weld", _draw_butt),
        ("Lap Joint", _draw_lap),
        ("Corner Joint", _draw_corner),
        ("Cruciform", _draw_cruciform),
    ]

    for i, (name, draw_fn) in enumerate(joint_data):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.set_title(name, fontsize=12, fontweight="bold")
        draw_fn(ax, t, w)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide the 6th subplot
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.5, "feaweld supports\n5 parametric\njoint types",
                     ha="center", va="center", fontsize=12, color=GRAY,
                     transform=axes[1, 2].transAxes, style="italic")

    fig.tight_layout()
    fig.savefig(out / "joint_types.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def _draw_fillet_t(ax, t, w):
    # Base plate (horizontal)
    ax.fill([-30, 30, 30, -30], [0, 0, -t, -t], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Web (vertical)
    ax.fill([-t / 2, t / 2, t / 2, -t / 2], [0, 0, 30, 30], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Weld fillets
    ax.fill([t / 2, t / 2 + w, t / 2], [0, 0, w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.fill([-t / 2, -t / 2 - w, -t / 2], [0, 0, w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.text(t / 2 + w + 2, w / 2, f"a={w}", fontsize=7, color=DARK)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-t - 5, 35)


def _draw_butt(ax, t, w):
    gap = 2
    # Left plate
    ax.fill([-30, -gap / 2, -gap / 2, -30], [t, t, 0, 0], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Right plate
    ax.fill([gap / 2, 30, 30, gap / 2], [t, t, 0, 0], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Weld (V-groove)
    ax.fill([-gap / 2, gap / 2, gap / 2 + 3, -gap / 2 - 3],
            [0, 0, t, t], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    # Cap reinforcement
    ax.fill([-gap / 2 - 3, gap / 2 + 3, gap / 2 + 2, -gap / 2 - 2],
            [t, t, t + 2, t + 2], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.text(0, t / 2, "V", fontsize=9, ha="center", va="center", color=DARK, fontweight="bold")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-5, t + 8)


def _draw_lap(ax, t, w):
    overlap = 20
    # Bottom plate
    ax.fill([-30, 30, 30, -30], [0, 0, -t, -t], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Top plate (offset)
    ax.fill([-10, 30, 30, -10], [0, 0, t, t], color="#c8d6e5", edgecolor=GRAY, lw=1.5)
    # Fillet weld
    ax.fill([-10, -10, -10 - w], [0, t, 0], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.text(-10 - w - 2, t / 3, f"a", fontsize=8, color=DARK)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-t - 5, t + 5)


def _draw_corner(ax, t, w):
    # Horizontal plate
    ax.fill([-30, 0, 0, -30], [0, 0, -t, -t], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Vertical plate
    ax.fill([0, t, t, 0], [0, 0, 30, 30], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Fillet weld (outside corner)
    ax.fill([0, w, 0], [0, 0, -w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.set_xlim(-35, 20)
    ax.set_ylim(-t - 10, 35)


def _draw_cruciform(ax, t, w):
    # Base plate (horizontal)
    ax.fill([-30, 30, 30, -30], [0, 0, -t, -t], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Web (vertical, both sides)
    ax.fill([-t / 2, t / 2, t / 2, -t / 2], [0, 0, 25, 25], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    ax.fill([-t / 2, t / 2, t / 2, -t / 2], [-t, -t, -t - 25, -t - 25], color="#dfe6e9", edgecolor=GRAY, lw=1.5)
    # Top fillets
    ax.fill([t / 2, t / 2 + w, t / 2], [0, 0, w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.fill([-t / 2, -t / 2 - w, -t / 2], [0, 0, w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    # Bottom fillets
    ax.fill([t / 2, t / 2 + w, t / 2], [-t, -t, -t - w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.fill([-t / 2, -t / 2 - w, -t / 2], [-t, -t, -t - w], color="#bdc3c7", edgecolor=DARK, lw=1, hatch="//")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-t - 30, 30)


def generate_sn_concept(out: Path) -> None:
    """S-N curve fundamentals diagram."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("S-N Curve Fundamentals", fontweight="bold")

    # Classic IIW-style S-N curve
    N_knee = 2e6
    S_knee = 90  # FAT90 knee point
    m1, m2 = 3.0, 5.0

    # Above knee
    N_high = np.logspace(3, np.log10(N_knee), 200)
    C1 = S_knee ** m1 * N_knee
    S_high = (C1 / N_high) ** (1 / m1)

    # Below knee
    N_low = np.logspace(np.log10(N_knee), 9, 200)
    C2 = S_knee ** m2 * N_knee
    S_low = (C2 / N_low) ** (1 / m2)

    ax.loglog(N_high, S_high, color=BLUE, linewidth=2.5, label=f"Slope m = {m1:.0f}")
    ax.loglog(N_low, S_low, color=DARK, linewidth=2.5, linestyle="--", label=f"Slope m = {m2:.0f}")

    # Knee point
    ax.plot(N_knee, S_knee, "D", color=GREEN, markersize=12, zorder=5, label="Knee point (CAFL)")

    # Regime bands
    ax.axvspan(1e3, 1e4, alpha=0.06, color="red", zorder=0)
    ax.axvspan(1e4, N_knee, alpha=0.06, color="orange", zorder=0)
    ax.axvspan(N_knee, 1e9, alpha=0.06, color="green", zorder=0)

    ax.text(3e3, 15, "Low-Cycle\nFatigue", fontsize=8, ha="center", color="#888")
    ax.text(1.5e5, 15, "High-Cycle\nFatigue", fontsize=8, ha="center", color="#888")
    ax.text(3e7, 15, "Endurance\nRegion", fontsize=8, ha="center", color="#888")

    # CAFL line
    ax.axhline(S_knee, color=GREEN, linestyle=":", alpha=0.4, linewidth=0.8)
    ax.text(2e3, S_knee * 1.05, f"CAFL = {S_knee} MPa", fontsize=8, color=GREEN)

    # N = 2e6 vertical
    ax.axvline(N_knee, color=GREEN, linestyle=":", alpha=0.4, linewidth=0.8)
    ax.text(N_knee * 1.5, 500, f"N = 2$\\times$10$^6$", fontsize=8, color=GREEN, rotation=90)

    # Example operating point
    S_op = 120
    N_op = C1 / (S_op ** m1)
    ax.plot(N_op, S_op, "ro", markersize=10, zorder=6)
    ax.annotate(f"Operating point\nS = {S_op} MPa\nN = {N_op:.1e}",
                xy=(N_op, S_op), xytext=(N_op * 5, S_op * 1.8),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    # Slope annotation
    ax.text(2e4, 350, f"m = {m1:.0f}", fontsize=10, color=BLUE, fontweight="bold", rotation=-35)

    ax.set_xlabel("Fatigue Life $N$ (cycles)")
    ax.set_ylabel("Stress Range $S$ (MPa)")
    ax.set_xlim(1e3, 1e9)
    ax.set_ylim(10, 1000)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "sn_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_dong_concept(out: Path) -> None:
    """Dong/Battelle structural stress concept diagram."""
    plt = _setup_matplotlib()
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Dong Mesh-Insensitive Structural Stress Method", fontweight="bold")

    # Plate cross-section
    plate_w, plate_h = 8, 2
    ax.fill([0, plate_w * 4, plate_w * 4, 0],
            [0, 0, plate_h, plate_h],
            color="#dfe6e9", edgecolor=GRAY, linewidth=1.5)

    # Weld at left end
    ax.fill([0, -2, 0], [0, 0, 3], color="#bdc3c7", edgecolor=DARK, lw=1.5, hatch="//")
    ax.text(-1.5, 1.5, "Weld", fontsize=8, ha="center", color="#555")

    # Mesh lines (vertical)
    mesh_x = np.linspace(0, plate_w * 4, 9)
    for mx in mesh_x:
        ax.plot([mx, mx], [0, plate_h], color="#bdc3c7", linewidth=0.5)
    # Horizontal mesh lines
    for my in [0, plate_h / 2, plate_h]:
        ax.plot([0, plate_w * 4], [my, my], color="#bdc3c7", linewidth=0.5)

    # Node dots at weld toe line (x=0)
    node_y = [0, plate_h / 2, plate_h]
    for ny in node_y:
        ax.plot(0, ny, "ko", markersize=6, zorder=5)

    # Force arrows at the weld toe nodes
    force_scale = 3
    forces = [1.5, 0.8, 0.5]
    for ny, f in zip(node_y, forces):
        ax.annotate("", xy=(-f * force_scale, ny), xytext=(0, ny),
                     arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(-force_scale * 2, plate_h + 0.5, "Nodal\nforces $F_i$", fontsize=9,
            ha="center", color=RED, fontweight="bold")

    # Decomposition box on the right
    box_x = plate_w * 4 + 3
    # Membrane component
    ax.fill([box_x, box_x + 6, box_x + 6, box_x],
            [0, 0, plate_h, plate_h],
            color=BLUE, alpha=0.15, edgecolor=BLUE)
    ax.annotate("", xy=(box_x + 3, 0), xytext=(box_x + 3, plate_h),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    ax.text(box_x + 3, plate_h / 2 + 0.3, "$\\sigma_m$", fontsize=12,
            ha="center", color=BLUE, fontweight="bold")

    # Plus sign
    ax.text(box_x + 7, plate_h / 2, "+", fontsize=16, ha="center", va="center", color=DARK)

    # Bending component
    box_x2 = box_x + 8
    ax.fill([box_x2, box_x2 + 6, box_x2 + 3, box_x2 + 3],
            [0, 0, plate_h, plate_h],
            color=ORANGE, alpha=0.15, edgecolor=ORANGE)
    ax.text(box_x2 + 3, plate_h / 2 + 0.3, "$\\sigma_b$", fontsize=12,
            ha="center", color=ORANGE, fontweight="bold")

    # Arrow from forces to decomposition
    ax.annotate("", xy=(box_x, plate_h / 2), xytext=(plate_w * 4 + 1, plate_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, ls="--"))

    # Key message
    ax.text(0.5, -0.08,
            "Key: Uses nodal force equilibrium at the weld toe line, not stress values "
            "from individual elements.  Result is mesh-insensitive.",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", alpha=0.9, edgecolor="#f5cba7"),
            style="italic")

    ax.set_xlim(-8, box_x2 + 8)
    ax.set_ylim(-2, plate_h + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out / "dong_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_sed_concept(out: Path) -> None:
    """Strain Energy Density (SED) control volume diagram."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Strain Energy Density (SED) Method\n(Lazzarin)", fontweight="bold")

    # Notch (V-shape)
    notch_depth = 3
    notch_angle = 30  # degrees half-angle
    rad = np.radians(notch_angle)
    plate_top = 5
    plate_bot = -5

    # Upper plate surface
    ax.fill([0, -notch_depth, -15, -15, 0],
            [notch_depth * np.tan(rad), 0, 0, plate_top, plate_top],
            color="#dfe6e9", edgecolor=GRAY, linewidth=1.5)
    # Lower plate surface
    ax.fill([0, -notch_depth, -15, -15, 0],
            [-notch_depth * np.tan(rad), 0, 0, plate_bot, plate_bot],
            color="#dfe6e9", edgecolor=GRAY, linewidth=1.5)

    # Control volume circle
    R0 = 2.5
    theta = np.linspace(0, 2 * np.pi, 100)
    cx, cy = -notch_depth, 0
    ax.plot(cx + R0 * np.cos(theta), cy + R0 * np.sin(theta),
            color=BLUE, linewidth=2.5, linestyle="--", label=f"Control volume ($R_0$)")

    # SED gradient (concentric rings)
    for r_frac in [0.8, 0.6, 0.4, 0.2]:
        circle = plt.Circle((cx, cy), R0 * r_frac, color=RED,
                             alpha=0.08 + 0.12 * (1 - r_frac), zorder=2)
        ax.add_patch(circle)

    # R0 dimension
    ax.annotate("", xy=(cx + R0, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    ax.text(cx + R0 / 2, cy + 0.4, f"$R_0$", fontsize=12, ha="center",
            color=BLUE, fontweight="bold")

    # Notch tip label
    ax.plot(cx, cy, "ko", markersize=5, zorder=5)
    ax.text(cx + 0.3, cy - 0.6, "Notch tip", fontsize=8, color="#555")

    # Equation
    ax.text(0.98, 0.02,
            "$\\bar{W} = \\frac{1}{V_c} \\int_{V_c} W \\, dV$\n\n"
            "$R_0$ = control radius\n"
            "$V_c$ = control volume",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", alpha=0.95, edgecolor="#bdc3c7"))

    ax.set_xlim(-16, 3)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out / "sed_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example Output Gallery (using feaweld API with synthetic data)
# ============================================================================

def generate_example_through_thickness(out: Path) -> None:
    """Example through-thickness plot using feaweld API."""
    plt = _setup_matplotlib()
    from feaweld.postprocess.linearization import LinearizationResult
    from feaweld.visualization.plots_2d import plot_through_thickness

    z = np.linspace(0, 10, 50)
    total = np.zeros((50, 6))
    total[:, 0] = 120 + 40 * (z / 10) + 8 * np.sin(3 * np.pi * z / 10)
    total[:, 1] = 60 + 15 * (z / 10)

    result = LinearizationResult(
        membrane=np.array([140, 67.5, 0, 0, 0, 0], dtype=float),
        bending=np.array([20, 7.5, 0, 0, 0, 0], dtype=float),
        peak=np.array([5, 2, 0, 0, 0, 0], dtype=float),
        z_coords=z,
        total_stress=total,
        membrane_scalar=140.0,
        bending_scalar=20.0,
        peak_scalar=5.0,
        membrane_plus_bending_scalar=160.0,
    )

    fig = plot_through_thickness(result, show=False)
    fig.savefig(out / "example_through_thickness.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_hotspot(out: Path) -> None:
    """Example hot-spot extrapolation plot."""
    plt = _setup_matplotlib()
    from feaweld.postprocess.hotspot import HotSpotResult, HotSpotType
    from feaweld.visualization.plots_2d import plot_hotspot_extrapolation

    result = HotSpotResult(
        hot_spot_stress=245.0,
        reference_stresses=[195.0, 165.0, 142.0],
        reference_distances=[4.0, 10.0, 12.0],
        extrapolation_type=HotSpotType.TYPE_A,
        weld_toe_location=np.array([0.0, 0.0, 0.0]),
    )
    fig = plot_hotspot_extrapolation(result, show=False)
    fig.savefig(out / "example_hotspot.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_sn_curve(out: Path) -> None:
    """Example S-N curve plot with operating point."""
    plt = _setup_matplotlib()
    from feaweld.core.types import SNCurve, SNSegment, SNStandard
    from feaweld.visualization.plots_2d import plot_sn_curve

    curve = SNCurve(
        name="IIW FAT 90",
        standard=SNStandard.IIW,
        segments=[
            SNSegment(m=3.0, C=90.0 ** 3 * 2e6, stress_threshold=52.0),
            SNSegment(m=5.0, C=52.0 ** 5 * (90.0 ** 3 * 2e6 / 52.0 ** 3), stress_threshold=0.0),
        ],
        cutoff_cycles=1e7,
    )
    fig = plot_sn_curve(curve, stress_range=120.0, show=False)
    fig.savefig(out / "example_sn_curve.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_dong(out: Path) -> None:
    """Example Dong decomposition plot."""
    plt = _setup_matplotlib()
    from feaweld.postprocess.dong import DongResult
    from feaweld.visualization.plots_2d import plot_dong_decomposition

    result = DongResult(
        membrane_stress=np.array([95, 110, 125, 115, 105, 98]),
        bending_stress=np.array([28, 35, 42, 38, 30, 25]),
        structural_stress=np.array([123, 145, 167, 153, 135, 123]),
        bending_ratio=np.array([0.23, 0.24, 0.25, 0.25, 0.22, 0.20]),
    )
    fig = plot_dong_decomposition(result, show=False)
    fig.savefig(out / "example_dong.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_asme_check(out: Path) -> None:
    """Example ASME stress check plot."""
    plt = _setup_matplotlib()
    from feaweld.postprocess.nominal import StressCategorization
    from feaweld.visualization.plots_2d import plot_asme_check

    cat = StressCategorization(
        membrane=130.0, bending=45.0, peak=18.0,
        total=193.0, stress_intensity=190.0,
    )
    fig = plot_asme_check(cat, S_m=160.0, S_y=275.0, show=False)
    fig.savefig(out / "example_asme_check.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_weld_groups_gallery(out: Path) -> None:
    """All 9 weld group shapes in a 3x3 grid."""
    plt = _setup_matplotlib()
    from feaweld.core.types import WeldGroupShape
    from feaweld.visualization.plots_2d import plot_weld_group_geometry

    shapes = [
        WeldGroupShape.LINE, WeldGroupShape.PARALLEL, WeldGroupShape.C_SHAPE,
        WeldGroupShape.L_SHAPE, WeldGroupShape.BOX, WeldGroupShape.CIRCULAR,
        WeldGroupShape.I_SHAPE, WeldGroupShape.T_SHAPE, WeldGroupShape.U_SHAPE,
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle("Weld Group Shapes (Blodgett Method)", fontsize=16, fontweight="bold")

    for i, shape in enumerate(shapes):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        plot_weld_group_geometry(shape, d=100.0, b=60.0, show=False, ax=ax)

    fig.tight_layout()
    fig.savefig(out / "weld_groups_gallery.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("pipeline_overview", generate_pipeline_diagram),
        ("hotspot_concept", generate_hotspot_concept),
        ("linearization_concept", generate_linearization_concept),
        ("goldak_concept", generate_goldak_concept),
        ("joint_types", generate_joint_types),
        ("sn_concept", generate_sn_concept),
        ("dong_concept", generate_dong_concept),
        ("sed_concept", generate_sed_concept),
        ("weld_groups_gallery", generate_weld_groups_gallery),
        ("example_through_thickness", generate_example_through_thickness),
        ("example_hotspot", generate_example_hotspot),
        ("example_sn_curve", generate_example_sn_curve),
        ("example_dong", generate_example_dong),
        ("example_asme_check", generate_example_asme_check),
    ]

    for name, gen_fn in generators:
        print(f"  Generating {name}...", end=" ", flush=True)
        try:
            gen_fn(OUTPUT_DIR)
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nDone. {len(generators)} images written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
