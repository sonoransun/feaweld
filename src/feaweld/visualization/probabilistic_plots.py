"""Matplotlib plots for probabilistic and reliability analyses.

Renders the outputs of `feaweld.probabilistic` — Sobol sensitivity
indices, Monte Carlo response distributions, and FORM reliability results.

All matplotlib imports are deferred via
`feaweld.visualization.plots_2d._require_matplotlib` so that the rest of
feaweld remains usable without the optional matplotlib dependency.  Every
public function follows the house signature style::

    (*, title=None, show=True, ax=None) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from itertools import cycle
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. Sobol sensitivity indices
# ---------------------------------------------------------------------------

def plot_sobol_indices(
    indices: Mapping[str, Mapping[str, float]],
    *,
    kind: str = "bar",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Horizontal bar chart of Sobol first-order and total sensitivity indices.

    Variables are sorted in descending order of their total index (tornado
    ordering), largest at the top.

    Parameters
    ----------
    indices : dict
        The return value of
        [feaweld.probabilistic.sensitivity.sobol_indices][], i.e.
        ``{"first_order": {name: S_i}, "total": {name: ST_i}}``.
    kind : {"bar", "tornado"}
        ``"bar"`` draws grouped bars comparing first-order (blue) against
        total (orange) indices.  ``"tornado"`` draws a single total-index bar
        per variable in the same sorted layout.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes
    from feaweld.visualization.theme import FEAWELD_BLUE, FEAWELD_ORANGE

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or "Sobol Sensitivity Indices")

    first_order = indices.get("first_order", {})
    total = indices.get("total", {})

    # Tornado ordering: descending by total index.
    names = sorted(total, key=lambda n: total[n], reverse=True)
    s_first = [float(first_order.get(n, 0.0)) for n in names]
    s_total = [float(total.get(n, 0.0)) for n in names]
    y = np.arange(len(names))

    max_st = max(s_total) if s_total else 1.0

    if kind == "tornado":
        ax.barh(y, s_total, color=FEAWELD_ORANGE, edgecolor="white",
                label="Total $S_{Ti}$")
        for yi, v in zip(y, s_total):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    elif kind == "bar":
        h = 0.4
        ax.barh(y - h / 2, s_first, height=h, color=FEAWELD_BLUE,
                edgecolor="white", label="First-order $S_i$")
        ax.barh(y + h / 2, s_total, height=h, color=FEAWELD_ORANGE,
                edgecolor="white", label="Total $S_{Ti}$")
        for yi, v in zip(y - h / 2, s_first):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7)
        for yi, v in zip(y + h / 2, s_total):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7)
    else:
        raise ValueError(f"Unknown kind '{kind}'. Choose from: 'bar', 'tornado'.")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # largest at the top
    ax.set_xlim(0, max(1.0, max_st) * 1.12)  # headroom for value labels
    ax.set_xlabel("Sensitivity index")
    if names:
        ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Monte Carlo response histogram
# ---------------------------------------------------------------------------

def plot_mc_histogram(
    results: NDArray | Sequence[float],
    *,
    percentiles: Sequence[float] | None = None,
    bins: int = 40,
    show_cdf: bool = True,
    xlabel: str = "Response",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Histogram of Monte Carlo responses with optional empirical CDF.

    Parameters
    ----------
    results : numpy.ndarray or sequence of float
        Sampled model responses.
    percentiles : sequence of float, optional
        Percentile levels (0-100) to mark with dashed vertical lines.
        Defaults to ``[5, 50, 95]`` (P5/P50/P95).
    bins : int
        Number of histogram bins.
    show_cdf : bool
        Overlay the empirical cumulative distribution on a twin y-axis.
    xlabel : str
        Label for the response axis.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes
    from feaweld.visualization.theme import (
        FEAWELD_BLUE, FEAWELD_DARK, FEAWELD_GREEN, FEAWELD_ORANGE, FEAWELD_RED,
    )

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or "Monte Carlo Distribution")

    data = np.asarray(results, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]

    ax.hist(data, bins=bins, color=FEAWELD_BLUE, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    ax2 = None
    if show_cdf and data.size:
        ax2 = ax.twinx()
        xs = np.sort(data)
        cdf = np.arange(1, xs.size + 1) / xs.size
        ax2.plot(xs, cdf, color=FEAWELD_DARK, linewidth=1.5, label="Empirical CDF")
        ax2.set_ylabel("Cumulative probability")
        ax2.set_ylim(0.0, 1.05)

    if percentiles is None:
        percentiles = [5, 50, 95]

    color_cycle = cycle([FEAWELD_GREEN, FEAWELD_ORANGE, FEAWELD_RED])
    for p, color in zip(percentiles, color_cycle):
        val = float(np.percentile(data, p)) if data.size else 0.0
        ax.axvline(val, color=color, linestyle="--", linewidth=1.2,
                   label=f"P{int(round(p))} = {val:.3g}")

    # Merge legends from the histogram axis and the CDF twin axis.
    lines_1, labels_1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
    else:
        lines_2, labels_2 = [], []
    if lines_1 or lines_2:
        ax.legend(lines_1 + lines_2, labels_1 + labels_2,
                  loc="upper right", fontsize="small")

    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. FORM reliability
# ---------------------------------------------------------------------------

def plot_form_reliability(
    form_result: Mapping[str, Any],
    *,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Bar chart of the FORM design point with a reliability annotation.

    Parameters
    ----------
    form_result : dict
        The return value of
        [feaweld.probabilistic.sensitivity.reliability_index_form][],
        i.e. ``{"beta": float, "probability_of_failure": float,
        "design_point": {name: value}}``.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes
    from feaweld.visualization.theme import FEAWELD_BLUE

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or "FORM Reliability")

    design_point = form_result.get("design_point", {})
    beta = float(form_result.get("beta", float("nan")))
    pf = float(form_result.get("probability_of_failure", float("nan")))

    names = list(design_point.keys())
    values = [float(design_point[n]) for n in names]
    y = np.arange(len(names))

    ax.barh(y, values, color=FEAWELD_BLUE, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Design-point value (most probable point)")

    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:.3g}", va="center", fontsize=8)

    # Reliability annotation box, styled like the equation boxes in plots_2d.
    ax.text(
        0.98, 0.02,
        f"$\\beta$ = {beta:.3f}\n$P_f$ = {pf:.3e}",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7"),
    )

    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
