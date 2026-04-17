"""Fatigue-specific 2D plots and animations.

Covers rainflow cycle histograms, damage-weighted S-N overlays, and the
damage-evolution animation. Consumes outputs of
:func:`feaweld.fatigue.rainflow.rainflow_count`,
:mod:`feaweld.fatigue.sn_curves`, and
:mod:`feaweld.fatigue.miner`.

All matplotlib imports are deferred via :func:`_require_matplotlib` so
the rest of feaweld stays importable without the optional ``viz`` extras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import NDArray


RainflowCycles = list[tuple[float, float, float]]
"""Output type of :func:`feaweld.fatigue.rainflow.rainflow_count`.

Each tuple is ``(stress_range, mean_stress, count)``.
"""


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for fatigue plots. "
            "Install with: pip install feaweld[viz]"
        ) from exc


def _unpack_cycles(cycles: RainflowCycles) -> tuple[NDArray, NDArray, NDArray]:
    """Return parallel arrays ``(ranges, means, counts)`` from rainflow output."""
    if not cycles:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    arr = np.asarray(cycles, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Rainflow histogram
# ---------------------------------------------------------------------------


def plot_rainflow_histogram(
    cycles: RainflowCycles,
    bins: int = 20,
    kind: Literal["range", "range_mean", "from_to"] = "range",
    *,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
):
    """Plot a rainflow cycle histogram.

    Parameters
    ----------
    cycles : list of (range, mean, count)
        Output of :func:`feaweld.fatigue.rainflow.rainflow_count`.
    bins : int
        Number of histogram bins along the range axis (and mean axis for
        the 2-D variants).
    kind : {"range", "range_mean", "from_to"}
        ``"range"`` — 1-D cycle-count bar chart vs. stress range.
        ``"range_mean"`` — 2-D heatmap with range on x-axis and mean
        on y-axis (the classic rainflow matrix).
        ``"from_to"`` — 2-D heatmap with from-stress on x-axis and
        to-stress on y-axis (the load-reversal form).
    title : str, optional
        Figure title.
    show : bool
        If ``True`` (default), call ``plt.show()``.
    ax : matplotlib Axes, optional
        Existing Axes to draw on. A new Figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure owning ``ax``.
    """
    from feaweld.visualization.theme import (
        apply_feaweld_style, get_cmap, FEAWELD_BLUE,
    )

    plt = _require_matplotlib()
    apply_feaweld_style()

    ranges, means, counts = _unpack_cycles(cycles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if len(ranges) == 0:
        ax.text(0.5, 0.5, "no cycles", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)
        return fig

    if kind == "range":
        # Weight each cycle by its count (half cycles count as 0.5)
        hist, edges = np.histogram(ranges, bins=bins, weights=counts)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        ax.bar(centers, hist, width=width * 0.9, color=FEAWELD_BLUE, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Stress range Δσ (MPa)")
        ax.set_ylabel("Cycle count")
        ax.grid(True, alpha=0.3)

    elif kind == "range_mean":
        hist, xedges, yedges = np.histogram2d(
            ranges, means, bins=bins, weights=counts,
        )
        im = ax.imshow(
            hist.T, origin="lower", aspect="auto",
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            cmap=get_cmap("damage"),
        )
        fig.colorbar(im, ax=ax, label="Cycle count")
        ax.set_xlabel("Stress range Δσ (MPa)")
        ax.set_ylabel("Mean stress σ_m (MPa)")

    elif kind == "from_to":
        frm = means - 0.5 * ranges
        to = means + 0.5 * ranges
        hist, xedges, yedges = np.histogram2d(
            frm, to, bins=bins, weights=counts,
        )
        im = ax.imshow(
            hist.T, origin="lower", aspect="auto",
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            cmap=get_cmap("damage"),
        )
        fig.colorbar(im, ax=ax, label="Cycle count")
        ax.set_xlabel("From-stress (MPa)")
        ax.set_ylabel("To-stress (MPa)")

    else:
        raise ValueError(
            f"Unknown kind {kind!r}. Use 'range', 'range_mean', or 'from_to'."
        )

    ax.set_title(title or f"Rainflow histogram ({kind})")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# S-N + damage-weighted cycle stack
# ---------------------------------------------------------------------------


def plot_sn_damage_stacked(
    cycles: RainflowCycles,
    sn_curve: Any,
    bins: int = 20,
    *,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
):
    """Overlay an S-N curve on a rainflow cycle-count histogram.

    Bars (left y-axis) are cycle counts per stress-range bin; the line
    (right y-axis) is the allowable life ``N(Δσ)`` from the supplied S-N
    curve. Bars rendered near or past the curve indicate damaging bins.

    Parameters
    ----------
    cycles : list of (range, mean, count)
        Rainflow output.
    sn_curve : object
        Object exposing ``sn_curve.life(stress_range: float) -> float``
        (most :mod:`feaweld.fatigue.sn_curves` classes satisfy this).
        A callable is also accepted.
    bins : int
        Number of cycle-count bins.
    title : str, optional
        Figure title.
    show : bool
        Show the figure after drawing.
    ax : matplotlib Axes, optional
        Reuse existing Axes; otherwise create a new Figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.theme import (
        apply_feaweld_style, FEAWELD_BLUE, FEAWELD_RED,
    )

    plt = _require_matplotlib()
    apply_feaweld_style()

    ranges, _means, counts = _unpack_cycles(cycles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if len(ranges) == 0:
        ax.text(0.5, 0.5, "no cycles", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    hist, edges = np.histogram(ranges, bins=bins, weights=counts)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centers, hist, width=width * 0.9, color=FEAWELD_BLUE, alpha=0.7,
           edgecolor="black", linewidth=0.4, label="Cycle count")
    ax.set_xlabel("Stress range Δσ (MPa)")
    ax.set_ylabel("Cycle count", color=FEAWELD_BLUE)
    ax.tick_params(axis="y", colors=FEAWELD_BLUE)
    ax.grid(True, alpha=0.3)

    # Allowable life from the S-N curve
    life_fn = sn_curve.life if hasattr(sn_curve, "life") else sn_curve
    stress_axis = np.linspace(max(centers.min(), 1.0), centers.max() * 1.1, 50)
    allowable = np.array([life_fn(float(s)) for s in stress_axis])

    ax2 = ax.twinx()
    ax2.plot(stress_axis, allowable, color=FEAWELD_RED, linewidth=2.0,
             label="Allowable N (S-N)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Allowable cycles N", color=FEAWELD_RED)
    ax2.tick_params(axis="y", colors=FEAWELD_RED)

    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper right")

    ax.set_title(title or "Rainflow cycles vs. S-N allowable")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Damage-evolution animation (Phase 9)
# ---------------------------------------------------------------------------


def animate_damage_evolution(
    load_blocks: Iterable[RainflowCycles],
    sn_curve: Any,
    output: str | Path,
    fps: int = 10,
    *,
    format: Literal["gif", "mp4"] | None = None,
    title: str | None = None,
) -> Path:
    """Animate Palmgren-Miner cumulative damage over a sequence of load blocks.

    At frame *k* the animation shows cumulative damage after applying
    blocks ``[0..k]``: a bar per block and a cumulative-damage line.
    The bar for the current block is highlighted.

    Parameters
    ----------
    load_blocks : iterable of rainflow cycles
        Each element is a rainflow-count list (``list[(range, mean, count)]``)
        for one block — typically one operating day, or one mission.
    sn_curve : object
        Object with ``sn_curve.life(stress_range) -> float``.
    output : str or Path
        Output file path. The extension (``.gif`` / ``.mp4``) selects the
        writer unless ``format`` is explicit.
    fps : int
        Frames per second.
    format : {"gif", "mp4"}, optional
        Override the writer. ``"mp4"`` falls back to GIF if ``ffmpeg``
        is not available (via ``imageio-ffmpeg``).
    title : str, optional
        Animation title.

    Returns
    -------
    Path
        Path to the written animation file.
    """
    from feaweld.visualization.theme import (
        apply_feaweld_style, FEAWELD_BLUE, FEAWELD_ORANGE, FEAWELD_RED,
    )

    plt = _require_matplotlib()
    from matplotlib.animation import FuncAnimation, PillowWriter
    apply_feaweld_style()

    output = Path(output)
    format = format or ("mp4" if output.suffix.lower() == ".mp4" else "gif")

    blocks = list(load_blocks)
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError("animate_damage_evolution needs at least one load block.")

    life_fn = sn_curve.life if hasattr(sn_curve, "life") else sn_curve

    # Per-block damage
    per_block_damage: list[float] = []
    for block in blocks:
        d_block = 0.0
        for rng, _mean, count in block:
            if rng <= 0.0:
                continue
            N = life_fn(float(rng))
            if N <= 0.0 or not np.isfinite(N):
                continue
            d_block += count / N
        per_block_damage.append(d_block)

    cumulative = np.cumsum(per_block_damage)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, n_blocks + 1)
    ax.set_xlim(0.5, n_blocks + 0.5)
    ymax = max(cumulative.max() * 1.15, 1.1)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Load block")
    ax.set_ylabel("Damage")
    ax.axhline(1.0, color=FEAWELD_RED, linestyle="--", linewidth=1.2, label="D = 1 (failure)")
    ax.grid(True, alpha=0.3)

    bars = ax.bar(x, np.zeros(n_blocks), color=FEAWELD_BLUE, alpha=0.8,
                  edgecolor="black", linewidth=0.4, label="Per-block damage")
    (line,) = ax.plot([], [], color=FEAWELD_ORANGE, linewidth=2.2,
                      marker="o", markersize=4, label="Cumulative D")
    text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10,
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_title(title or "Miner damage accumulation")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88))

    def update(frame: int):
        for i, bar in enumerate(bars):
            if i < frame:
                bar.set_height(per_block_damage[i])
                bar.set_color(FEAWELD_BLUE)
            elif i == frame:
                bar.set_height(per_block_damage[i])
                bar.set_color(FEAWELD_ORANGE)
            else:
                bar.set_height(0.0)
        line.set_data(x[: frame + 1], cumulative[: frame + 1])
        text.set_text(
            f"block {frame + 1}/{n_blocks}\n"
            f"cum. D = {cumulative[frame]:.3f}"
        )
        return (*bars, line, text)

    anim = FuncAnimation(fig, update, frames=n_blocks, interval=1000 / fps, blit=False)

    if format == "mp4":
        try:
            from matplotlib.animation import FFMpegWriter
            writer: Any = FFMpegWriter(fps=fps)
        except (ImportError, RuntimeError):
            output = output.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    anim.save(str(output), writer=writer, dpi=110)
    plt.close(fig)
    return output
