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


def generate_rainflow_spectrum_concept(out: Path) -> None:
    """Spectrum fatigue chain: load history -> rainflow cycles -> Miner damage."""
    plt = _setup_matplotlib()
    from feaweld.core.types import SNCurve, SNSegment, SNStandard
    from feaweld.fatigue.rainflow import rainflow_count

    fig, (ax_hist, ax_cyc, ax_dmg) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Spectrum Fatigue — Rainflow Counting and Miner Summation",
                 fontweight="bold")

    # Synthetic variable-amplitude history (fixed seed, incommensurate periods)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 60, 600)
    signal = (
        90
        + 55 * np.sin(2 * np.pi * t / 31)
        + 35 * np.sin(2 * np.pi * t / 7.3 + rng.uniform(0, 2 * np.pi))
        + 20 * np.sin(2 * np.pi * t / 3.1 + rng.uniform(0, 2 * np.pi))
    )

    # --- Panel 1: history with turning points --------------------------------
    ax_hist.set_title("Load History", fontsize=11)
    d = np.diff(signal)
    tp = np.where(d[:-1] * d[1:] < 0)[0] + 1
    ax_hist.plot(t, signal, color=DARK, linewidth=1.2, zorder=2)
    ax_hist.plot(t[tp], signal[tp], "o", color=RED, markersize=4,
                 markeredgecolor="white", markeredgewidth=0.4, linestyle="none",
                 zorder=3, label=f"Turning points ({len(tp)})")
    ax_hist.set_xlabel("Time (s)")
    ax_hist.set_ylabel("Stress (MPa)")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, linestyle=":", alpha=0.3)
    ax_hist.text(0.5, -0.22, "Variable-amplitude history reduced to peaks and valleys.",
                 transform=ax_hist.transAxes, fontsize=8, ha="center",
                 color=GRAY, style="italic")

    # --- Panel 2: extracted rainflow cycles ----------------------------------
    ax_cyc.set_title("Extracted Cycles (range–mean)", fontsize=11)
    cycles = rainflow_count(signal)
    ranges = np.array([c[0] for c in cycles])
    means = np.array([c[1] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    full = counts >= 1.0
    ax_cyc.scatter(means[full], ranges[full], s=55, c=BLUE,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="Full cycles (n = 1)")
    ax_cyc.scatter(means[~full], ranges[~full], s=55, c=ORANGE, marker="^",
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="Half cycles (n = ½)")
    ax_cyc.text(0.03, 0.97,
                f"{len(cycles)} cycles extracted\n($\\Sigma n$ = {counts.sum():.1f})",
                transform=ax_cyc.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaf2f8",
                          alpha=0.95, edgecolor="#bdc3c7"))
    ax_cyc.set_xlabel("Mean stress (MPa)")
    ax_cyc.set_ylabel("Stress range $\\Delta\\sigma$ (MPa)")
    ax_cyc.legend(loc="lower right", fontsize=8)
    ax_cyc.grid(True, linestyle=":", alpha=0.3)
    ax_cyc.text(0.5, -0.22, "ASTM E1049 four-point rainflow pairs closed hysteresis loops.",
                transform=ax_cyc.transAxes, fontsize=8, ha="center",
                color=GRAY, style="italic")

    # --- Panel 3: spectrum histogram + per-bin Miner damage ------------------
    ax_dmg.set_title("Spectrum + Miner Damage", fontsize=11)
    curve = SNCurve(
        name="IIW FAT 90",
        standard=SNStandard.IIW,
        segments=[
            SNSegment(m=3.0, C=90.0 ** 3 * 2e6, stress_threshold=52.0),
            SNSegment(m=5.0, C=52.0 ** 5 * (90.0 ** 3 * 2e6 / 52.0 ** 3), stress_threshold=0.0),
        ],
        cutoff_cycles=1e7,
    )
    bins = np.linspace(0, ranges.max() * 1.05, 10)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    n_bin = np.zeros(len(centers))
    d_bin = np.zeros(len(centers))
    idx = np.clip(np.digitize(ranges, bins) - 1, 0, len(centers) - 1)
    for b, sr, cnt in zip(idx, ranges, counts):
        n_bin[b] += cnt
        N_i = curve.life(sr)
        if np.isfinite(N_i):
            d_bin[b] += cnt / N_i

    ax_dmg.bar(centers - 0.21 * width, n_bin, width=0.38 * width, color=BLUE,
               alpha=0.85, label="Cycles $n_i$")
    ax_dmg.set_xlabel("Stress range $\\Delta\\sigma$ (MPa)")
    ax_dmg.set_ylabel("Cycles per repetition $n_i$", color=BLUE)
    ax_dmg.tick_params(axis="y", labelcolor=BLUE)

    ax_dmg2 = ax_dmg.twinx()
    ax_dmg2.bar(centers + 0.21 * width, d_bin, width=0.38 * width, color=RED,
                alpha=0.85, label="Damage $n_i/N_i$ (FAT 90)")
    ax_dmg2.set_ylabel("Miner damage $n_i / N_i$", color=RED)
    ax_dmg2.tick_params(axis="y", labelcolor=RED)
    ax_dmg2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    D_total = d_bin.sum()
    ax_dmg.text(0.03, 0.97,
                f"$D = \\Sigma\\, n_i/N_i$ = {D_total:.1e}\n"
                f"Life $\\approx 1/D$ = {1.0 / D_total:.1e} repeats",
                transform=ax_dmg.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaf2f8",
                          alpha=0.95, edgecolor="#bdc3c7"))
    handles1, labels1 = ax_dmg.get_legend_handles_labels()
    handles2, labels2 = ax_dmg2.get_legend_handles_labels()
    ax_dmg.legend(handles1 + handles2, labels1 + labels2,
                  loc="upper right", fontsize=8)
    ax_dmg.grid(True, linestyle=":", alpha=0.3)
    ax_dmg.text(0.5, -0.22, "Per-bin damage against IIW FAT 90 — large ranges dominate.",
                transform=ax_dmg.transAxes, fontsize=8, ha="center",
                color=GRAY, style="italic")

    fig.tight_layout()
    fig.savefig(out / "rainflow_spectrum_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_haigh_concept(out: Path) -> None:
    """Haigh diagram: Goodman / Gerber / Soderberg mean-stress corrections."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Haigh Diagram — Mean-Stress Corrections", fontweight="bold")

    sigma_u, sigma_y, sigma_e = 500.0, 355.0, 180.0
    sm = np.linspace(0, sigma_u, 300)

    goodman = sigma_e * (1 - sm / sigma_u)
    gerber = sigma_e * (1 - (sm / sigma_u) ** 2)
    soderberg = np.where(sm <= sigma_y, sigma_e * (1 - sm / sigma_y), np.nan)

    ax.plot(sm, goodman, color=BLUE, linewidth=2.5, label="Goodman (to $\\sigma_u$)")
    ax.plot(sm, gerber, color=GREEN, linewidth=2, linestyle="--",
            label="Gerber (parabola)")
    ax.plot(sm, soderberg, color=ORANGE, linewidth=2, linestyle="-.",
            label="Soderberg (to $\\sigma_y$)")

    # Safe region below the Goodman line
    ax.fill_between(sm, 0, goodman, color=GREEN, alpha=0.08, zorder=0)
    ax.text(60, 55, "Safe region\n(below Goodman)", fontsize=9, color="#4a7a5a",
            style="italic")

    # Operating point and its equivalent fully-reversed amplitude
    sm_op, sa_op = 200.0, 100.0
    sa_r = sa_op / (1 - sm_op / sigma_u)
    ax.plot(sm, sa_r * (1 - sm / sigma_u), color=RED, linestyle=":",
            linewidth=1.2, alpha=0.7)
    ax.plot(sm_op, sa_op, "o", color=RED, markersize=10, zorder=5)
    ax.plot(0, sa_r, "*", color=RED, markersize=16, zorder=5)
    ax.annotate("", xy=(8, sa_r), xytext=(sm_op, sa_op),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate(f"Operating point\n($\\sigma_m$ = {sm_op:.0f}, $\\sigma_a$ = {sa_op:.0f})",
                xy=(sm_op, sa_op), xytext=(sm_op + 45, sa_op + 45),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(105, 190,
            f"Equivalent fully-reversed amplitude\n"
            f"$\\sigma_{{ar}} = \\sigma_a / (1 - \\sigma_m/\\sigma_u)$ = {sa_r:.0f} MPa",
            fontsize=9, color=RED, fontweight="bold")

    # Material landmarks on the axes
    ax.plot([sigma_y, sigma_u], [0, 0], "|", color=DARK, markersize=10, zorder=4)
    ax.text(sigma_y, -14, "$\\sigma_y$", fontsize=10, ha="center", color=ORANGE,
            fontweight="bold")
    ax.text(sigma_u, -14, "$\\sigma_u$", fontsize=10, ha="center", color=BLUE,
            fontweight="bold")
    ax.text(-12, sigma_e, "$\\sigma_e$", fontsize=10, ha="right", va="center",
            color=DARK, fontweight="bold")

    ax.text(0.98, 0.55,
            "Goodman: $\\frac{\\sigma_a}{\\sigma_{ar}} + \\frac{\\sigma_m}{\\sigma_u} = 1$\n\n"
            "Gerber: $\\frac{\\sigma_a}{\\sigma_{ar}} + \\left(\\frac{\\sigma_m}{\\sigma_u}\\right)^2 = 1$\n\n"
            "Soderberg: $\\frac{\\sigma_a}{\\sigma_{ar}} + \\frac{\\sigma_m}{\\sigma_y} = 1$",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", alpha=0.95,
                      edgecolor="#bdc3c7"))

    ax.set_xlabel("Mean stress $\\sigma_m$ (MPa)")
    ax.set_ylabel("Stress amplitude $\\sigma_a$ (MPa)")
    ax.set_xlim(-25, sigma_u + 25)
    ax.set_ylim(-25, sa_r + 60)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "haigh_mean_stress_concept.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_residual_stress_integration(out: Path) -> None:
    """Residual stress in the fatigue chain: profile -> mean-stress shift."""
    plt = _setup_matplotlib()
    from feaweld.data.residual_stress import evaluate_residual_stress

    sigma_y = 355.0
    fig, (ax_prof, ax_trace) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Residual Stress in the Fatigue Assessment", fontweight="bold")

    # --- Panel 1: through-thickness profiles (bundled dataset) ---------------
    ax_prof.set_title("Through-Thickness Profile (butt weld)", fontsize=11)
    zz = np.linspace(0, 1, 200)
    aw = evaluate_residual_stress("BS7910_Level2_butt", zz, sigma_y) / sigma_y
    pwht = evaluate_residual_stress("PWHT_butt", zz, sigma_y) / sigma_y

    ax_prof.plot(aw, zz, color=RED, linewidth=2.5, label="As-welded (BS 7910 L2)")
    ax_prof.plot(pwht, zz, color=BLUE, linewidth=2, linestyle="--", label="After PWHT")
    ax_prof.fill_betweenx(zz, 0, aw, color=RED, alpha=0.07)
    ax_prof.axvline(0, color=GRAY, linewidth=0.8)
    ax_prof.text(-0.06, 0.97, "compression", fontsize=7, ha="right", color=GRAY)
    ax_prof.text(0.06, 0.97, "tension", fontsize=7, ha="left", color=GRAY)

    # Surface values feed the fatigue mean-stress shift
    ax_prof.plot(aw[0], 0, "o", color=RED, markersize=9, zorder=5)
    ax_prof.plot(pwht[0], 0, "o", color=BLUE, markersize=9, zorder=5)
    ax_prof.annotate(f"$\\sigma_{{res}}(0)$ = {aw[0]:.2f} $\\sigma_y$",
                     xy=(aw[0], 0), xytext=(0.45, 0.14), fontsize=9, color=RED,
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax_prof.annotate(f"{pwht[0]:.2f} $\\sigma_y$",
                     xy=(pwht[0], 0), xytext=(0.02, 0.24), fontsize=9, color=BLUE,
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax_prof.text(1.02, 0.02, "weld toe surface", fontsize=8, color="#555",
                 va="top", ha="right", transform=ax_prof.get_yaxis_transform())
    ax_prof.text(1.02, 0.98, "back face", fontsize=8, color="#555",
                 va="bottom", ha="right", transform=ax_prof.get_yaxis_transform())

    ax_prof.set_xlabel("$\\sigma_{res} / \\sigma_y$")
    ax_prof.set_ylabel("$z / t$")
    ax_prof.set_xlim(-0.48, 1.15)
    ax_prof.set_ylim(1.03, -0.03)
    ax_prof.legend(loc="center left", fontsize=8)
    ax_prof.grid(True, linestyle=":", alpha=0.3)

    # --- Panel 2: mean-stress shift of the applied cycle ---------------------
    ax_trace.set_title("Mean-Stress Shift of the Applied Cycle", fontsize=11)
    res_aw = evaluate_residual_stress("BS7910_Level2_butt", 0.0, sigma_y)
    res_pwht = evaluate_residual_stress("PWHT_butt", 0.0, sigma_y)
    delta = 120.0
    t_c = np.linspace(0, 4, 400)
    applied = delta / 2 * (1 - np.cos(2 * np.pi * t_c))
    # As-welded: sigma_res = sigma_y, so the cycle shakes down to peak at yield
    trace_aw = sigma_y - delta + applied
    trace_pwht = res_pwht + applied

    ax_trace.plot(t_c, applied, color=GRAY, linewidth=1.5, linestyle=":",
                  label=f"Applied ($\\Delta\\sigma$ = {delta:.0f}, R = 0)")
    ax_trace.plot(t_c, trace_pwht, color=BLUE, linewidth=2,
                  label=f"PWHT: + {res_pwht:.0f} MPa")
    ax_trace.plot(t_c, trace_aw, color=RED, linewidth=2,
                  label=f"As-welded: + $\\sigma_{{res}}$ = {res_aw:.0f} MPa")
    ax_trace.axhline(sigma_y, color=DARK, linestyle="--", linewidth=1, alpha=0.6)
    ax_trace.text(0.1, sigma_y + 10, f"$\\sigma_y$ = {sigma_y:.0f} MPa",
                  fontsize=8, color=DARK)
    ax_trace.text(2.6, sigma_y + 17,
                  "shakedown: peak limited to $\\sigma_y$",
                  fontsize=7, color=GRAY, style="italic", ha="center")

    # Shifted ranges: same delta-sigma, higher effective R-ratio
    r_aw = (sigma_y - delta) / sigma_y
    r_pwht = res_pwht / (res_pwht + delta)
    for lo, color, r_eff in [(sigma_y - delta, RED, r_aw), (res_pwht, BLUE, r_pwht)]:
        ax_trace.annotate("", xy=(4.2, lo + delta), xytext=(4.2, lo),
                          arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
        ax_trace.text(4.35, lo + delta / 2,
                      f"$\\Delta\\sigma$ = {delta:.0f}\n$R_{{eff}}$ = {r_eff:.2f}",
                      fontsize=8, color=color, va="center")

    ax_trace.set_xlabel("Time (cycles)")
    ax_trace.set_ylabel("Stress (MPa)")
    ax_trace.set_xlim(0, 6.0)
    ax_trace.set_ylim(-25, 425)
    ax_trace.legend(loc="lower right", fontsize=8)
    ax_trace.grid(True, linestyle=":", alpha=0.3)

    fig.text(0.5, -0.02,
             "Residual stress leaves $\\Delta\\sigma$ unchanged but raises the effective "
             "mean / R-ratio — captured by superposition or a Goodman/Gerber correction.",
             ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", alpha=0.9,
                       edgecolor="#f5cba7"))

    fig.tight_layout()
    fig.savefig(out / "residual_stress_integration_concept.svg", format="svg",
                bbox_inches="tight")
    plt.close(fig)


def generate_hotspot_3d_stations(out: Path) -> None:
    """3-D hot-spot assessment: extrapolation stations along the weld toe line."""
    plt = _setup_matplotlib()

    fig, (ax_iso, ax_prof) = plt.subplots(
        2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1.5, 1]})
    fig.suptitle("3-D Hot-Spot Stress — Stations Along the Weld Toe Line",
                 fontweight="bold")

    # --- Top: pseudo-perspective plate with toe line and stations ------------
    ax_iso.set_title("Extrapolation stations on the toe line", fontsize=11)

    def persp(x: float, z: float) -> tuple[float, float]:
        # Skew (x = distance from toe, z = along weld) into screen coordinates
        return x + 0.45 * z, 0.30 * z

    L = 100.0
    t_pl = 10.0
    plate = [persp(0, 0), persp(35, 0), persp(35, L), persp(0, L)]
    ax_iso.fill(*zip(*plate), color="#dfe6e9", edgecolor=GRAY, linewidth=1.5)
    weld = [persp(-7, 0), persp(0, 0), persp(0, L), persp(-7, L)]
    ax_iso.fill(*zip(*weld), color="#bdc3c7", edgecolor=GRAY, linewidth=1.2, hatch="//")

    (x0, y0), (x1, y1) = persp(0, 0), persp(0, L)
    ax_iso.plot([x0, x1], [y0, y1], color=DARK, linewidth=3, zorder=4)
    toe_angle = np.degrees(np.arctan2(0.30, 0.45))
    ax_iso.text(*persp(-11.5, 48), "Weld toe line", fontsize=9, color=DARK,
                fontweight="bold", ha="center", rotation=toe_angle)

    z_stations = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    z_gov = 50.0
    for z_s in z_stations:
        for x_ref in (0.4 * t_pl, 1.0 * t_pl):
            ax_iso.plot(*persp(x_ref, z_s), "o", color=BLUE, markersize=5, zorder=5)
        ax_iso.annotate("", xy=persp(0.3, z_s), xytext=persp(13, z_s),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
    ax_iso.text(*persp(4, 4), "0.4t", fontsize=8, color=BLUE, ha="center", va="top")
    ax_iso.text(*persp(10.5, 2), "1.0t", fontsize=8, color=BLUE, ha="center", va="top")
    ax_iso.text(*persp(17, 32), "linear extrapolation\nto the toe", fontsize=8,
                color=BLUE, style="italic", ha="center", rotation=toe_angle)

    ax_iso.plot(*persp(0, z_gov), "*", color=RED, markersize=17, zorder=6)
    ax_iso.annotate("Governing station\n(max $\\sigma_{hs}$)",
                    xy=persp(0, z_gov), xytext=(3, 26), fontsize=9, color=RED,
                    fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax_iso.annotate("", xy=persp(39, 36), xytext=persp(39, 10),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))
    ax_iso.text(*persp(43.5, 22), "$z$ (along weld)", fontsize=8, color=GRAY,
                ha="center", va="center", rotation=toe_angle)

    ax_iso.text(0.99, 0.02,
                "At each station: extrapolate $\\sigma$ at 0.4t and 1.0t\n"
                "to the toe (IIW Type A); assess the maximum.",
                transform=ax_iso.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", alpha=0.95,
                          edgecolor="#bdc3c7"))

    ax_iso.set_xlim(-15, 84)
    ax_iso.set_ylim(-4, 33)
    ax_iso.set_aspect("equal")
    ax_iso.axis("off")

    # --- Bottom: hot-spot stress profile along the toe line ------------------
    ax_prof.set_title("Hot-spot stress along the toe line", fontsize=11)
    z = np.linspace(0, L, 300)
    sigma_hs = 160.0 + 55.0 * np.exp(-(((z - z_gov) / 22.0) ** 2))
    sigma_st = 160.0 + 55.0 * np.exp(-(((z_stations - z_gov) / 22.0) ** 2))

    ax_prof.plot(z, sigma_hs, color=DARK, linewidth=2, label="$\\sigma_{hs}(z)$")
    ax_prof.plot(z_stations, sigma_st, "o", color=BLUE, markersize=8, zorder=5,
                 label="Stations")
    s_max = sigma_st.max()
    ax_prof.plot(z_gov, s_max, "*", color=RED, markersize=17, zorder=6)
    ax_prof.axhline(s_max, color=RED, linestyle=":", alpha=0.4, linewidth=0.8)
    ax_prof.annotate(f"max $\\sigma_{{hs}}$ = {s_max:.0f} MPa $\\rightarrow$ fatigue check",
                     xy=(z_gov, s_max), xytext=(z_gov + 14, s_max - 9),
                     fontsize=9, color=RED, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax_prof.set_xlabel("Position along weld toe $z$ (mm)")
    ax_prof.set_ylabel("$\\sigma_{hs}$ (MPa)")
    ax_prof.set_xlim(0, L)
    ax_prof.set_ylim(145, 235)
    ax_prof.legend(loc="upper left", fontsize=9)
    ax_prof.grid(True, linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "hotspot_3d_stations_concept.svg", format="svg",
                bbox_inches="tight")
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


def generate_sn_standards_comparison(out: Path) -> None:
    """Overlay of weld-detail design S-N curves from six standards."""
    plt = _setup_matplotlib()
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    from feaweld.fatigue.sn_curves import (
        aws_curve, bs7608_curve, dnv_curve, ec3_curve, iiw_fat,
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_title("Design S-N Curves — Standards Comparison", fontweight="bold")

    n_min, n_max = 1e4, 3e8
    ec3 = ec3_curve(90)
    # These ~category-90 details share nearly identical m = 3 branches, so
    # stagger linewidth / z-order and phase-shift equal-period dashes to keep
    # every curve visible where they coincide.
    entries = [
        (iiw_fat(90), "IIW FAT 90", BLUE, "-", 3.4, 2.0),
        (ec3, "EC3 detail category 90", RED, "-", 1.2, 3.6),
        (bs7608_curve("D"), "BS 7608 class D", GREEN, (0, (3, 6)), 2.2, 2.6),
        (bs7608_curve("F"), "BS 7608 class F", "#16a085", "-", 2.0, 2.4),
        (aws_curve("C"), "AWS D1.1 category C", "#9b59b6", (3, (3, 6)), 2.2, 3.0),
        (dnv_curve("D"), "DNV-RP-C203 curve D", ORANGE, (6, (3, 6)), 2.2, 2.2),
    ]

    for curve, label, color, ls, lw, z in entries:
        # Sample each segment from its upper bound down to its threshold and
        # evaluate N = life(S); the finite-life polyline ends at the cutoff.
        N_parts: list[np.ndarray] = []
        S_parts: list[np.ndarray] = []
        for idx, seg in enumerate(curve.segments):
            if idx == 0:
                s_hi = (seg.C / n_min) ** (1.0 / seg.m)
            else:
                s_hi = curve.segments[idx - 1].stress_threshold
            s_lo = seg.stress_threshold
            if s_lo <= 0.0:
                s_lo = (seg.C / curve.cutoff_cycles) ** (1.0 / seg.m)
            s = np.logspace(np.log10(s_hi), np.log10(s_lo), 80)
            N_parts.append(np.array([curve.life(v) for v in s]))
            S_parts.append(s)
        N = np.concatenate(N_parts)
        S = np.concatenate(S_parts)
        keep = np.isfinite(N) & (N <= n_max)
        ax.loglog(N[keep], S[keep], color=color, linewidth=lw, linestyle=ls,
                  zorder=z, label=label)

        # Knee points (slope transitions)
        for seg_above in curve.segments[:-1]:
            s_knee = seg_above.stress_threshold
            ax.plot(curve.life(s_knee), s_knee, "D", color=color,
                    markersize=4, zorder=5)

        # Below the last threshold life is infinite: draw the flat CAFL tail.
        s_cafl = curve.segments[-1].stress_threshold
        if s_cafl > 0.0:
            n_cafl = curve.life(s_cafl)
            if np.isfinite(n_cafl) and n_cafl < n_max:
                ax.loglog([n_cafl, n_max], [s_cafl, s_cafl], ":",
                          color=color, linewidth=0.65 * lw, alpha=0.8, zorder=z)
                ax.plot(n_cafl, s_cafl, "o", color=color, markersize=4, zorder=5)

    # Reference life used to define detail categories / FAT classes
    ax.axvline(2e6, color=GRAY, linestyle=":", alpha=0.5, linewidth=1)
    ax.text(2e6 * 1.25, 640, "$N = 2\\times10^6$\n(reference life)",
            fontsize=8, color=GRAY, ha="left", va="top")

    # EC3 has its knee earlier than the IIW/DNV/BS 1e7 convention
    s_knee_ec3 = ec3.segments[0].stress_threshold
    ax.annotate("EC3 knee\n($N_D = 5\\times10^6$)",
                xy=(ec3.life(s_knee_ec3), s_knee_ec3), xytext=(4e5, 41),
                fontsize=8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    ax.text(1.5e5, 255, "$m$ = 3 finite-life branches nearly coincide",
            fontsize=8, color=GRAY, style="italic", rotation=-30,
            ha="center", va="bottom")

    ax.text(0.02, 0.03,
            "Dotted tails: constant-amplitude fatigue limit\n"
            "(infinite design life below the threshold)",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaf2f8",
                      alpha=0.95, edgecolor="#bdc3c7"))

    ax.set_xlabel("Cycles to failure $N$")
    ax.set_ylabel("Stress range $\\Delta\\sigma$ (MPa)")
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(22, 800)
    ax.yaxis.set_major_locator(FixedLocator([30, 50, 70, 100, 150, 200, 300, 500]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "sn_standards_comparison.svg", format="svg",
                bbox_inches="tight")
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
# Architecture overview (module map)
# ============================================================================

def generate_architecture_overview(out: Path) -> None:
    """Box-and-arrow module map of the feaweld package.

    Three bands: a foundation row (core, data), the linear analysis pipeline
    (geometry -> mesh -> solver -> postprocess -> fatigue -> pipeline/report),
    and an extensions row hooking into the pipeline.
    """
    plt = _setup_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(LIGHT_BG)

    def box(x, y, w, h, label, color, *, text_color="white", fontsize=10):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.92, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=text_color, zorder=3)

    # Band labels (left gutter).
    ax.text(0.15, 9.3, "Extensions", fontsize=11, fontweight="bold",
            color=GRAY, rotation=90, va="center", ha="center")
    ax.text(0.15, 5.0, "Analysis Pipeline", fontsize=11, fontweight="bold",
            color=DARK, rotation=90, va="center", ha="center")
    ax.text(0.15, 1.4, "Foundation", fontsize=11, fontweight="bold",
            color=GRAY, rotation=90, va="center", ha="center")

    # --- Middle band: the linear analysis pipeline --------------------------
    pipeline = [
        ("geometry", BLUE),
        ("mesh", "#16a085"),
        ("solver", "#8e44ad"),
        ("postprocess", RED),
        ("fatigue", ORANGE),
        ("pipeline\n/ report", GREEN),
    ]
    xs = np.linspace(2.0, 14.0, len(pipeline))
    y_mid = 5.0
    bw, bh = 1.9, 1.2
    for (label, color), x in zip(pipeline, xs):
        box(x, y_mid, bw, bh, label, color)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - bw / 2, y_mid),
                    xytext=(xs[i] + bw / 2, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))

    # --- Bottom band: foundation modules used everywhere --------------------
    y_found = 1.4
    box(4.5, y_found, 6.4, 1.1, "core  (types, materials, loads)",
        DARK, fontsize=11)
    box(11.5, y_found, 6.4, 1.1, "data  (registry + cache)", DARK, fontsize=11)
    # A few dashed "builds-on" arrows from foundation up into the pipeline.
    for x in (xs[0], xs[2], xs[4]):
        ax.annotate("", xy=(x, y_mid - bh / 2), xytext=(x, y_found + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2,
                                    ls="--", alpha=0.6))

    # --- Top band: extensions, each hooking into a pipeline stage -----------
    y_ext = 9.2
    ew, eh = 2.2, 0.9
    extensions = [
        ("probabilistic", 3),   # -> postprocess
        ("ml", 4),              # -> fatigue
        ("multiscale", 2),      # -> solver
        ("digital_twin", 5),    # -> pipeline/report
        ("singularity", 3),     # -> postprocess
        ("visualization", 5),   # -> pipeline/report
    ]
    ext_xs = np.linspace(2.0, 14.0, len(extensions))
    for (label, hook), x in zip(extensions, ext_xs):
        box(x, y_ext, ew, eh, label, ORANGE if label in
            ("probabilistic", "ml", "multiscale") else BLUE, fontsize=9)
        ax.annotate("", xy=(xs[hook], y_mid + bh / 2), xytext=(x, y_ext - eh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0,
                                    ls=":", alpha=0.65))

    ax.set_title("feaweld Architecture — Module Map", fontsize=16,
                 fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(out / "architecture_overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example Gallery — probabilistic, material, and multiscale plot families
# ============================================================================

def generate_example_sobol(out: Path) -> None:
    """Sobol sensitivity indices from a real analysis on a toy weld model."""
    plt = _setup_matplotlib()
    from feaweld.probabilistic.monte_carlo import RandomVariable
    from feaweld.probabilistic.sensitivity import sobol_indices
    from feaweld.visualization.probabilistic_plots import plot_sobol_indices

    variables = [
        RandomVariable("leg_size", "normal", {"mean": 8.0, "std": 1.2}),
        RandomVariable("load", "normal", {"mean": 10000.0, "std": 2000.0}),
        RandomVariable("thickness", "normal", {"mean": 10.0, "std": 1.0}),
    ]

    def weld_stress(v: dict) -> float:
        throat = 0.707 * v["leg_size"]
        area = throat * 100.0  # 100 mm weld length
        membrane = v["load"] / area
        return membrane * (1.0 + 0.04 * (12.0 - v["thickness"]))

    indices = sobol_indices(variables, weld_stress, n_base=128, seed=7)
    fig = plot_sobol_indices(
        indices, kind="bar", show=False,
        title="Sobol Sensitivity — Fillet Weld Stress",
    )
    fig.savefig(out / "example_sobol.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_mc_distribution(out: Path) -> None:
    """Monte Carlo safety-factor distribution with empirical CDF."""
    plt = _setup_matplotlib()
    from feaweld.visualization.probabilistic_plots import plot_mc_histogram

    rng = np.random.default_rng(2024)
    # Safety factor: lognormal-ish, centred near 1.6.
    samples = np.exp(rng.normal(np.log(1.6), 0.22, size=5000))
    fig = plot_mc_histogram(
        samples, show_cdf=True, xlabel="Safety factor",
        title="Monte Carlo — Fatigue Safety Factor", show=False,
    )
    fig.savefig(out / "example_mc_distribution.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_reliability(out: Path) -> None:
    """FORM reliability design point for a resistance-stress toy problem."""
    plt = _setup_matplotlib()
    from feaweld.probabilistic.monte_carlo import RandomVariable
    from feaweld.probabilistic.sensitivity import reliability_index_form
    from feaweld.visualization.probabilistic_plots import plot_form_reliability

    variables = [
        RandomVariable("resistance", "normal", {"mean": 350.0, "std": 25.0}),
        RandomVariable("stress", "normal", {"mean": 200.0, "std": 30.0}),
    ]

    def limit_state(v: dict) -> float:
        return v["resistance"] - v["stress"]  # failure when R < S

    form = reliability_index_form(variables, limit_state)
    fig = plot_form_reliability(
        form, show=False,
        title="FORM Reliability — Resistance vs Stress",
    )
    fig.savefig(out / "example_reliability.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_cct(out: Path) -> None:
    """CCT diagram for a bundled structural-steel grade with a cooling marker."""
    plt = _setup_matplotlib()
    from feaweld.data.cct import get_cct_diagram, list_cct_grades
    from feaweld.visualization.material_plots import plot_cct_diagram

    grade = "S355" if "S355" in list_cct_grades() else "A36"
    diagram = get_cct_diagram(grade)
    fig = plot_cct_diagram(diagram, grade=grade, cooling_rate=30.0, show=False)
    fig.savefig(out / "example_cct.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_residual_stress(out: Path) -> None:
    """Overlay of BS 7910 and API 579 through-thickness residual-stress profiles."""
    plt = _setup_matplotlib()
    from feaweld.visualization.material_plots import plot_residual_stress_profile

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_residual_stress_profile(
        "BS7910_Level2_butt", yield_strength=355.0, ax=ax, show=False,
    )
    plot_residual_stress_profile(
        "API579_Level2_butt", yield_strength=355.0, ax=ax, show=False,
        title="Residual Stress Profiles — Butt Weld (BS 7910 vs API 579)",
    )
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out / "example_residual_stress.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_creep_relaxation(out: Path) -> None:
    """Norton-Bailey creep-strain curves at several stress levels."""
    plt = _setup_matplotlib()
    from feaweld.visualization.material_plots import plot_creep_curve

    fig = plot_creep_curve(
        A=1e-20, n=5.0, m=0.0,
        stress_levels=[100.0, 200.0, 300.0],
        t_end=2 * 3600.0,  # 2 hours in seconds
        title="Norton-Bailey Creep — 2 h Hold",
        show=False,
    )
    fig.savefig(out / "example_creep_relaxation.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_example_multiscale(out: Path) -> None:
    """Hall-Petch curve with weld-zone grain-size markers."""
    plt = _setup_matplotlib()
    from feaweld.multiscale.micro import HALL_PETCH_LOW_CARBON_STEEL
    from feaweld.visualization.material_plots import plot_hall_petch

    # A single alloy: the zones differ only in grain size, not in the
    # Hall-Petch constants, so markers sit on one curve.
    fig = plot_hall_petch(
        HALL_PETCH_LOW_CARBON_STEEL,
        grain_size_range=(2.0, 200.0),
        markers={"FG-HAZ": 15.0, "Base": 30.0, "CG-HAZ": 100.0},
        title="Hall-Petch — Grain Size Across Weld Zones",
        show=False,
    )
    fig.savefig(out / "example_multiscale.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Example Gallery — 3-D PyVista screenshots (synthetic fillet-weld block)
# ============================================================================

def _build_fillet_mesh():
    """Synthetic 3-D fillet-weld block (HEX8) with a weld-toe stress hot spot.

    Builds a conforming structured hexahedral mesh over an L-shaped
    cross-section (base plate + upstanding web joined by a 45-degree fillet)
    extruded along the weld line, plus a StressField whose sigma_yy peaks at
    the re-entrant weld toe and a cantilever-style displacement field.

    Returns
    -------
    tuple
        ``(FEMesh, StressField, displacement)`` where *displacement* is an
        ``(n_nodes, 3)`` array.
    """
    from feaweld.core.types import ElementType, FEMesh, StressField

    # Cross-section (x, y) and weld-line (z) extents, mm.
    W, H, Lz = 30.0, 30.0, 36.0
    tb, tw, leg = 8.0, 8.0, 8.0  # base thickness, web thickness, fillet leg
    nx, ny, nz = 16, 16, 13
    dx, dy, dz = W / (nx - 1), H / (ny - 1), Lz / (nz - 1)

    def inside(xc: float, yc: float) -> bool:
        if yc <= tb:            # base plate
            return True
        if xc <= tw:            # web
            return True
        # 45-degree fillet bridging web and base plate
        return (xc - tw) + (yc - tb) <= leg

    node_id: dict[tuple[int, int, int], int] = {}
    coords: list[tuple[float, float, float]] = []

    def get_node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        nid = node_id.get(key)
        if nid is None:
            nid = len(coords)
            node_id[key] = nid
            coords.append((i * dx, j * dy, k * dz))
        return nid

    elements: list[list[int]] = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                if not inside((i + 0.5) * dx, (j + 0.5) * dy):
                    continue
                # VTK hexahedron node order: bottom quad then top quad.
                elements.append([
                    get_node(i, j, k), get_node(i + 1, j, k),
                    get_node(i + 1, j + 1, k), get_node(i, j + 1, k),
                    get_node(i, j, k + 1), get_node(i + 1, j, k + 1),
                    get_node(i + 1, j + 1, k + 1), get_node(i, j + 1, k + 1),
                ])

    nodes = np.asarray(coords, dtype=np.float64)
    elems = np.asarray(elements, dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elems, element_type=ElementType.HEX8)

    # Stress field: sigma_yy peaks at the re-entrant weld toe (tw, tb).
    d_toe = np.sqrt((nodes[:, 0] - tw) ** 2 + (nodes[:, 1] - tb) ** 2)
    nominal, peak, lam = 60.0, 260.0, 6.0
    syy = nominal + peak * np.exp(-d_toe / lam)
    values = np.zeros((mesh.n_nodes, 6), dtype=np.float64)
    values[:, 1] = syy                     # sigma_yy
    values[:, 0] = 0.15 * syy              # small sigma_xx
    values[:, 3] = 0.20 * (syy - nominal)  # tau_xy concentrated near the toe
    stress = StressField(values=values, location="nodes")

    # Cantilever-style displacement: web tip deflects in +x.
    web_h = np.clip(nodes[:, 1] - tb, 0.0, None)
    disp = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
    disp[:, 0] = 0.02 * web_h ** 2 / max(H - tb, 1.0)
    return mesh, stress, disp


def generate_3d_stress_screenshot(out: Path) -> None:
    """3-D von-Mises stress contour on the synthetic fillet-weld block."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("(pyvista unavailable, skipping)", end=" ")
        return

    from feaweld.visualization.export import export_png
    from feaweld.visualization.stress_plots import plot_stress_field

    mesh, stress, _disp = _build_fillet_mesh()
    plotter = plot_stress_field(mesh, stress, component="von_mises", show=False)
    plotter.camera_position = "iso"
    export_png(plotter, str(out / "example_3d_stress.png"), resolution=(1200, 800))
    plotter.close()


def generate_3d_deformed_screenshot(out: Path) -> None:
    """3-D deformed shape of the fillet-weld block, coloured by von-Mises stress."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("(pyvista unavailable, skipping)", end=" ")
        return

    from feaweld.visualization.export import export_png
    from feaweld.visualization.stress_plots import plot_deformed

    mesh, stress, disp = _build_fillet_mesh()
    plotter = plot_deformed(mesh, displacement=disp, scale=12.0, stress=stress,
                            show=False)
    plotter.camera_position = "iso"
    export_png(plotter, str(out / "example_3d_deformed.png"), resolution=(1200, 800))
    plotter.close()


def generate_3d_joint_screenshot(out: Path) -> None:
    """Extruded fillet T-joint from the real 3-D pipeline with toe lines.

    Unlike the synthetic block above, this runs the actual geometry + meshing
    stages (``FilletTJoint(dimension=3)`` -> Gmsh -> ``FEMesh``) and overlays
    the ``weld_toe_{i}`` node sets as red tubes plus hot-spot stations along
    the governing toe line.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("(pyvista unavailable, skipping)", end=" ")
        return
    try:
        import gmsh
    except ImportError:
        print("(gmsh unavailable, skipping)", end=" ")
        return

    from feaweld.geometry.joints import FilletTJoint
    from feaweld.mesh.generator import WeldMeshConfig, generate_mesh
    from feaweld.visualization.export import export_png
    from feaweld.visualization.stress_plots import _mesh_to_pyvista_grid
    from feaweld.visualization.theme import configure_plotter

    # Real 3-D pipeline: extruded joint + tet mesh with toe-line refinement.
    joint = FilletTJoint(
        base_width=60.0,
        base_thickness=10.0,
        web_height=30.0,
        web_thickness=8.0,
        weld_leg_size=6.0,
        length=40.0,
        dimension=3,
    )
    config = WeldMeshConfig(
        global_size=5.0,
        weld_toe_size=1.5,
        weld_region_size=2.5,
        refinement_distance=8.0,
        element_order=1,
        element_type_2d="tri",
    )
    # The script runs many generators in one process: make sure no stale Gmsh
    # session bleeds in, and none bleeds out even if meshing fails part-way
    # (generate_mesh itself finalizes only on success).
    if gmsh.is_initialized():
        gmsh.finalize()
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # keep script output clean
        mesh = generate_mesh(joint, config)
    finally:
        if gmsh.is_initialized():
            gmsh.finalize()

    grid = _mesh_to_pyvista_grid(mesh)

    plotter = pv.Plotter(off_screen=True)
    configure_plotter(plotter)

    # Plates in the light steel used by the 2-D schematics; the two weld
    # volumes (physical groups) in the darker weld gray.  Disjoint cell sets
    # avoid z-fighting between the overlays.
    weld_cells = np.concatenate(
        [mesh.physical_groups["weld_left"], mesh.physical_groups["weld_right"]]
    )
    plate_cells = np.setdiff1d(np.arange(mesh.n_elements), weld_cells)
    plotter.add_mesh(grid.extract_cells(plate_cells), color="#dfe6e9",
                     show_edges=True, edge_color="gray", edge_opacity=0.35)
    plotter.add_mesh(grid.extract_cells(weld_cells), color="#bdc3c7",
                     show_edges=True, edge_color=DARK, edge_opacity=0.4)

    # Weld-toe lines: the swept ``weld_toe_{i}`` node sets, ordered along z.
    for i in range(len(joint.get_weld_toe_points())):
        pts = mesh.nodes[mesh.node_sets[f"weld_toe_{i}"]]
        pts = pts[np.argsort(pts[:, 2])]
        plotter.add_mesh(pv.lines_from_points(pts).tube(radius=0.45), color=RED)

    # Hot-spot stations along the governing (right base-plate) toe line.
    (tx, ty, _), (_, _, tz1) = joint.get_weld_toe_lines()[3]
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        governing = frac == 0.5
        plotter.add_mesh(
            pv.Sphere(radius=1.6 if governing else 1.2,
                      center=(tx, ty, frac * tz1)),
            color=DARK if governing else BLUE,
        )

    plotter.add_text(
        "Extruded Fillet T-Joint (dimension: 3)\n"
        "weld-toe lines (red) with hot-spot stations",
        position="upper_left", font_size=11, color=DARK,
    )
    plotter.add_axes()

    # Oblique view from front-top-right so the z extrusion depth reads.
    center = np.asarray(grid.center)
    plotter.camera_position = [
        tuple(center + np.array([80.0, 55.0, 95.0])), tuple(center), (0, 1, 0),
    ]
    plotter.camera.zoom(1.2)
    export_png(plotter, str(out / "example_3d_joint.png"), resolution=(1200, 800))
    plotter.close()


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
        ("rainflow_spectrum_concept", generate_rainflow_spectrum_concept),
        ("haigh_mean_stress_concept", generate_haigh_concept),
        ("residual_stress_integration_concept", generate_residual_stress_integration),
        ("hotspot_3d_stations_concept", generate_hotspot_3d_stations),
        ("weld_groups_gallery", generate_weld_groups_gallery),
        ("example_through_thickness", generate_example_through_thickness),
        ("example_hotspot", generate_example_hotspot),
        ("example_sn_curve", generate_example_sn_curve),
        ("sn_standards_comparison", generate_sn_standards_comparison),
        ("example_dong", generate_example_dong),
        ("example_asme_check", generate_example_asme_check),
        ("architecture_overview", generate_architecture_overview),
        ("example_sobol", generate_example_sobol),
        ("example_mc_distribution", generate_example_mc_distribution),
        ("example_reliability", generate_example_reliability),
        ("example_cct", generate_example_cct),
        ("example_residual_stress", generate_example_residual_stress),
        ("example_creep_relaxation", generate_example_creep_relaxation),
        ("example_multiscale", generate_example_multiscale),
        ("example_3d_stress", generate_3d_stress_screenshot),
        ("example_3d_deformed", generate_3d_deformed_screenshot),
        ("example_3d_joint", generate_3d_joint_screenshot),
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
