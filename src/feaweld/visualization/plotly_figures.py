"""Plotly figure builders for interactive HTML reports.

Mirrors the most useful static :mod:`~feaweld.visualization.report_figures`
plots with Plotly equivalents. Each builder returns a
:class:`plotly.graph_objects.Figure`; embed them in HTML via
``fig.to_html(include_plotlyjs="cdn", full_html=False)``.

Plotly is an optional dependency — installed with ``pip install feaweld[viz]``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive reports. "
            "Install with: pip install feaweld[viz]"
        ) from exc


# ---------------------------------------------------------------------------
# Stress histogram
# ---------------------------------------------------------------------------


def stress_histogram_plotly(stress_values: NDArray[np.float64]) -> Any:
    """Interactive histogram of von Mises stress values.

    Parameters
    ----------
    stress_values : NDArray
        Nodal von Mises stresses.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()

    fig = go.Figure(
        data=[
            go.Histogram(
                x=stress_values,
                nbinsx=40,
                marker={"color": "#2980b9", "line": {"color": "black", "width": 0.4}},
                hovertemplate="σ_vm ∈ [%{x}]<br>count = %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Von Mises stress distribution",
        xaxis_title="σ_vm (MPa)",
        yaxis_title="Node count",
        bargap=0.04,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# S-N curve
# ---------------------------------------------------------------------------


def sn_curve_plotly(
    sn_curve: Any,
    operating_point: tuple[float, float] | None = None,
    stress_range: tuple[float, float] = (10.0, 1000.0),
    n_points: int = 100,
) -> Any:
    """Interactive log-log S-N curve.

    Parameters
    ----------
    sn_curve : object
        Object with a ``life(stress_range) -> float`` method.
    operating_point : (stress_range, cycles), optional
        Marker for the applied stress range and realized life.
    stress_range : (low, high)
        Stress axis bounds (MPa).
    n_points : int
        Number of points along the curve.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()

    life_fn = sn_curve.life if hasattr(sn_curve, "life") else sn_curve

    s = np.logspace(np.log10(stress_range[0]), np.log10(stress_range[1]), n_points)
    N = np.array([life_fn(float(si)) for si in s])
    mask = np.isfinite(N) & (N > 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=N[mask], y=s[mask], mode="lines",
        name=getattr(sn_curve, "name", "S-N"),
        line={"color": "#2980b9", "width": 2.5},
        hovertemplate="N = %{x:.2e}<br>σ = %{y:.1f} MPa<extra></extra>",
    ))

    if operating_point is not None:
        op_s, op_n = operating_point
        fig.add_trace(go.Scatter(
            x=[op_n], y=[op_s], mode="markers",
            name="Operating point",
            marker={"color": "#e74c3c", "size": 12, "symbol": "circle-open", "line": {"width": 2}},
            hovertemplate=f"σ = {op_s:.1f} MPa<br>N = {op_n:.2e}<extra></extra>",
        ))

    fig.update_layout(
        title="S-N fatigue curve",
        xaxis={"title": "Cycles to failure N", "type": "log"},
        yaxis={"title": "Stress range Δσ (MPa)", "type": "log"},
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Rainflow histogram
# ---------------------------------------------------------------------------


def rainflow_plotly(cycles: list[tuple[float, float, float]], bins: int = 20) -> Any:
    """Interactive rainflow-range histogram.

    Parameters
    ----------
    cycles : list of (range, mean, count)
        Output of :func:`feaweld.fatigue.rainflow.rainflow_count`.
    bins : int
        Number of histogram bins.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()

    if cycles:
        arr = np.asarray(cycles, dtype=float)
        ranges, means, counts = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        ranges = means = counts = np.array([], dtype=float)

    hist, edges = np.histogram(ranges, bins=bins, weights=counts)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure(
        data=[
            go.Bar(
                x=centers, y=hist,
                marker={"color": "#2980b9", "line": {"color": "black", "width": 0.4}},
                hovertemplate="Δσ ≈ %{x:.1f} MPa<br>count = %{y:.1f}<extra></extra>",
                name="cycles",
            )
        ]
    )
    fig.update_layout(
        title="Rainflow cycle histogram",
        xaxis_title="Stress range Δσ (MPa)",
        yaxis_title="Cycle count",
        bargap=0.05,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------


def convergence_plotly(result: Any) -> Any:
    """Interactive mesh convergence plot.

    Parameters
    ----------
    result : feaweld.singularity.convergence.ConvergenceResult

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()

    sizes = np.asarray(result.mesh_sizes, dtype=float)
    values = np.asarray(result.stress_values, dtype=float)
    extrap = float(result.extrapolated_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=values, mode="lines+markers",
        line={"color": "#2980b9", "width": 2},
        marker={"size": 10, "color": "#2980b9"},
        name="FEA result",
        hovertemplate="h = %{x:.3g}<br>value = %{y:.4g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[sizes.min(), sizes.max()],
        y=[extrap, extrap], mode="lines",
        line={"color": "#27ae60", "width": 1.5, "dash": "dash"},
        name=f"Richardson = {extrap:.3g}",
    ))

    fig.update_layout(
        title=(
            f"Mesh convergence (GCI = {result.gci * 100:.2f} %, "
            f"{'converged' if result.is_converged else 'not converged'})"
        ),
        xaxis={
            "title": "Element size h (mm)",
            "type": "log",
            "autorange": "reversed",
        },
        yaxis_title="Quantity of interest",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Report helper: build a dict of Plotly div strings
# ---------------------------------------------------------------------------


def generate_interactive_figures(result: Any) -> dict[str, str]:
    """Generate Plotly HTML fragments for an analysis WorkflowResult.

    Each entry in the returned dict is an HTML ``<div>`` string ready to
    drop into a Jinja / string-template HTML report. The Plotly runtime
    is referenced once via CDN.

    Parameters
    ----------
    result : feaweld.pipeline.workflow.WorkflowResult

    Returns
    -------
    dict[str, str]
        Map of figure key to self-contained HTML fragment.
    """
    figures: dict[str, str] = {}

    # Stress distribution
    if result.fea_results is not None and result.fea_results.stress is not None:
        fig = stress_histogram_plotly(result.fea_results.stress.von_mises)
        figures["stress_distribution"] = fig.to_html(
            include_plotlyjs=False, full_html=False,
        )

    # S-N curve (try to pull the curve name from postprocess config)
    try:
        from feaweld.fatigue.sn_curves import get_sn_curve
        curve_name = result.case.postprocess.sn_curve
        sn = get_sn_curve(curve_name)

        op = None
        if result.fatigue_results:
            for key, value in result.fatigue_results.items():
                if isinstance(value, dict) and "life" in value and "stress_range" in value:
                    op = (float(value["stress_range"]), float(value["life"]))
                    break

        fig = sn_curve_plotly(sn, operating_point=op)
        figures["sn_curve"] = fig.to_html(include_plotlyjs=False, full_html=False)
    except Exception:
        pass

    # Rainflow (only if present in postprocess results)
    if result.postprocess_results:
        rf = result.postprocess_results.get("rainflow")
        if isinstance(rf, (list, tuple)) and rf and isinstance(rf[0], tuple):
            fig = rainflow_plotly(list(rf))
            figures["rainflow"] = fig.to_html(include_plotlyjs=False, full_html=False)

    return figures
