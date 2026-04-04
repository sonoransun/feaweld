"""Comparison visualization for parametric studies.

Provides plots for comparing metrics, parameter sensitivity, stress
field differences, and composite dashboards across multiple analysis cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for comparison plots. "
            "Install with: pip install feaweld[viz]"
        ) from exc


def _prepare_axes(ax, show, figsize=(8, 5)):
    """Get or create axes, return (fig, ax, should_show)."""
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax, show


# ---------------------------------------------------------------------------
# Metric comparison bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    study_results: Any,
    metric: str = "max_von_mises",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Bar chart comparing one metric across all cases.

    Args:
        study_results: StudyResults from Study.run()
        metric: MetricSet field name (e.g., "max_von_mises", "fatigue_life")
        title: Plot title. Auto-generated if None.
        show: If True, display interactively.
        ax: Optional Matplotlib axes.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from feaweld.pipeline.comparison import MetricSet

    fig, ax, show = _prepare_axes(ax, show, figsize=(max(8, len(study_results.results) * 1.2), 5))

    names = []
    values = []
    for name, result in study_results.results.items():
        ms = MetricSet.from_workflow_result(result)
        val = getattr(ms, metric, None)
        if val is not None:
            names.append(name)
            values.append(val)

    if not values:
        ax.text(0.5, 0.5, f"No data for metric: {metric}",
                ha="center", va="center", transform=ax.transAxes)
        if show:
            plt.show()
        return fig

    # Color by magnitude: green (low) to red (high) for stress metrics
    # Reverse for safety_factor and fatigue_life (higher is better)
    reverse = metric in ("safety_factor", "fatigue_life")
    norm_vals = np.array(values, dtype=float)
    vmin, vmax = norm_vals.min(), norm_vals.max()
    if vmax > vmin:
        fracs = (norm_vals - vmin) / (vmax - vmin)
    else:
        fracs = np.full_like(norm_vals, 0.5)
    if reverse:
        fracs = 1.0 - fracs

    cmap = plt.cm.RdYlGn_r  # red=high, green=low
    colors = [cmap(f) for f in fracs]

    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, values):
        y = bar.get_height()
        label = f"{val:.1f}" if abs(val) < 1e5 else f"{val:.1e}"
        ax.text(bar.get_x() + bar.get_width() / 2, y, label,
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    display_name = metric.replace("_", " ").title()
    ax.set_ylabel(display_name)
    ax.set_title(title or f"Comparison: {display_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity(
    study_results: Any,
    param_path: str,
    metric: str = "max_von_mises",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Scatter + line plot of a metric vs. a swept parameter.

    Args:
        study_results: StudyResults from Study.run()
        param_path: Dot-path of the parameter (e.g., "load.axial_force")
        metric: MetricSet field name.
        show: If True, display interactively.
        ax: Optional Matplotlib axes.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from feaweld.pipeline.comparison import MetricSet
    from feaweld.pipeline.study import _get_nested_attr

    fig, ax, show = _prepare_axes(ax, show)

    param_vals = []
    metric_vals = []
    labels = []

    for name, case in study_results.cases.items():
        if name not in study_results.results:
            continue
        try:
            pv = _get_nested_attr(case, param_path)
            if not isinstance(pv, (int, float)):
                continue
        except AttributeError:
            continue

        ms = MetricSet.from_workflow_result(study_results.results[name])
        mv = getattr(ms, metric, None)
        if mv is not None:
            param_vals.append(float(pv))
            metric_vals.append(mv)
            labels.append(name)

    if not param_vals:
        ax.text(0.5, 0.5, "No sensitivity data available",
                ha="center", va="center", transform=ax.transAxes)
        if show:
            plt.show()
        return fig

    # Sort by parameter value
    order = np.argsort(param_vals)
    param_vals = [param_vals[i] for i in order]
    metric_vals = [metric_vals[i] for i in order]
    labels = [labels[i] for i in order]

    ax.plot(param_vals, metric_vals, "o-", color="#2980b9", markersize=8, linewidth=2)

    # Label each point
    for x, y, lbl in zip(param_vals, metric_vals, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 8),
                    fontsize=7, color="#555")

    short_param = param_path.rsplit(".", 1)[-1]
    display_metric = metric.replace("_", " ").title()
    ax.set_xlabel(short_param.replace("_", " ").title())
    ax.set_ylabel(display_metric)
    ax.set_title(title or f"Sensitivity: {display_metric} vs. {short_param}")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Stress field difference
# ---------------------------------------------------------------------------

def plot_stress_difference(
    mesh: Any,
    stress_a: Any,
    stress_b: Any,
    component: str = "von_mises",
    label_a: str = "A",
    label_b: str = "B",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Contour plot of stress difference (A - B).

    Uses a diverging colormap centered at zero.

    Args:
        mesh: FEMesh
        stress_a: StressField from case A
        stress_b: StressField from case B
        component: Stress component to difference.
        label_a: Name of case A.
        label_b: Name of case B.
        show: If True, display.
        ax: Optional axes.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from matplotlib.tri import Triangulation
    from feaweld.pipeline.comparison import compute_stress_field_difference

    fig, ax, show = _prepare_axes(ax, show)

    diff = compute_stress_field_difference(stress_a, stress_b)

    if component == "von_mises":
        # von Mises of A minus von Mises of B (not von Mises of difference)
        vals = stress_a.von_mises - stress_b.von_mises
    elif component == "tresca":
        vals = stress_a.tresca - stress_b.tresca
    elif component in ("xx", "yy", "zz", "xy", "yz", "xz"):
        idx = ["xx", "yy", "zz", "xy", "yz", "xz"].index(component)
        vals = diff.values[:, idx]
    else:
        vals = stress_a.von_mises - stress_b.von_mises

    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    vmax = max(abs(np.min(vals)), abs(np.max(vals)))
    if vmax < 1e-12:
        vmax = 1.0

    if mesh.elements.shape[1] >= 3:
        tri = Triangulation(x, y, mesh.elements[:, :3])
        tc = ax.tricontourf(tri, vals, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(tc, ax=ax, label=f"Delta {component} (MPa)")
    else:
        sc = ax.scatter(x, y, c=vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=5)
        fig.colorbar(sc, ax=ax, label=f"Delta {component} (MPa)")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title or f"Stress Difference: {label_a} - {label_b}")
    ax.set_aspect("equal")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Stress envelope (distribution overlay)
# ---------------------------------------------------------------------------

def plot_stress_envelope(
    study_results: Any,
    component: str = "von_mises",
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Overlay stress distributions from all cases on one axes.

    Each case is drawn as a semi-transparent histogram.

    Args:
        study_results: StudyResults
        component: Stress component.
        show: If True, display.
        ax: Optional axes.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax, show = _prepare_axes(ax, show)

    cmap = plt.cm.tab10
    plotted = 0

    for i, (name, result) in enumerate(study_results.results.items()):
        fea = result.fea_results
        if fea is None or fea.stress is None:
            continue

        if component == "von_mises":
            vals = fea.stress.von_mises
        elif component == "tresca":
            vals = fea.stress.tresca
        elif component in ("xx", "yy", "zz", "xy", "yz", "xz"):
            idx = ["xx", "yy", "zz", "xy", "yz", "xz"].index(component)
            vals = fea.stress.values[:, idx]
        else:
            vals = fea.stress.von_mises

        color = cmap(i % 10)
        ax.hist(vals, bins=30, alpha=0.4, color=color, edgecolor=color,
                linewidth=1.5, label=name, density=True)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center",
                transform=ax.transAxes)

    display = component.replace("_", " ").title()
    ax.set_xlabel(f"{display} Stress (MPa)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Stress Distribution Comparison: {display}")
    if plotted > 0:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Comparison dashboard
# ---------------------------------------------------------------------------

def comparison_dashboard(
    study_results: Any,
    baseline: str | None = None,
    show: bool = True,
) -> Any:
    """Multi-panel comparison dashboard (2x2).

    Panels:
      [1] Metric comparison bars (max von Mises)
      [2] Auto-detected parameter sensitivity (or stress envelope if no sweep)
      [3] Stress distribution overlay
      [4] Summary text

    Args:
        study_results: StudyResults
        baseline: Optional baseline case for delta reporting.
        show: If True, display.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from feaweld.pipeline.comparison import MetricSet

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Study Comparison: {study_results.study_name}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Panel 1: Metric bars
    plot_metric_comparison(study_results, "max_von_mises", show=False, ax=axes[0, 0])

    # Panel 2: Sensitivity or second metric
    _auto_sensitivity_panel(study_results, axes[0, 1])

    # Panel 3: Stress envelope
    plot_stress_envelope(study_results, show=False, ax=axes[1, 0])

    # Panel 4: Summary text
    _summary_panel(study_results, baseline, axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    return fig


def _auto_sensitivity_panel(study_results: Any, ax: Any) -> None:
    """Auto-detect a swept parameter and plot sensitivity."""
    from feaweld.pipeline.study import _get_nested_attr

    # Try to detect which parameter was swept by checking for variation
    test_paths = [
        "load.axial_force", "load.bending_moment", "load.shear_force",
        "load.pressure", "mesh.global_size", "mesh.weld_toe_size",
        "geometry.base_thickness", "geometry.weld_leg_size",
        "material.temperature",
    ]

    cases = list(study_results.cases.values())
    if len(cases) < 2:
        ax.text(0.5, 0.5, "Not enough cases for sensitivity",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Parameter Sensitivity")
        return

    for path in test_paths:
        try:
            vals = [_get_nested_attr(c, path) for c in cases]
            if isinstance(vals[0], (int, float)) and len(set(vals)) > 1:
                plot_parameter_sensitivity(study_results, path, "max_von_mises",
                                          show=False, ax=ax)
                return
        except (AttributeError, TypeError):
            continue

    # Fallback: plot fatigue life comparison
    plot_metric_comparison(study_results, "fatigue_life", show=False, ax=ax)


def _summary_panel(study_results: Any, baseline: str | None, ax: Any) -> None:
    """Summary text panel for comparison dashboard."""
    from feaweld.pipeline.comparison import MetricSet

    lines = []
    lines.append(f"Cases: {study_results.n_cases}")
    lines.append(f"Succeeded: {study_results.n_succeeded}")
    lines.append(f"Failed: {study_results.n_failed}")
    lines.append(f"Elapsed: {study_results.elapsed_seconds:.1f} s")
    lines.append("")

    # Find best/worst
    metrics = {
        name: MetricSet.from_workflow_result(r)
        for name, r in study_results.results.items()
    }

    vm_values = {n: m.max_von_mises for n, m in metrics.items() if m.max_von_mises is not None}
    if vm_values:
        worst = max(vm_values, key=vm_values.get)
        best = min(vm_values, key=vm_values.get)
        lines.append(f"Highest stress: {worst}")
        lines.append(f"  {vm_values[worst]:.1f} MPa")
        lines.append(f"Lowest stress: {best}")
        lines.append(f"  {vm_values[best]:.1f} MPa")
        lines.append(f"Range: {vm_values[worst] - vm_values[best]:.1f} MPa")

    life_values = {n: m.fatigue_life for n, m in metrics.items() if m.fatigue_life is not None}
    if life_values:
        lines.append("")
        longest = max(life_values, key=life_values.get)
        shortest = min(life_values, key=life_values.get)
        lines.append(f"Longest life: {longest}")
        lines.append(f"  {life_values[longest]:.0f} cycles")
        lines.append(f"Shortest life: {shortest}")
        lines.append(f"  {life_values[shortest]:.0f} cycles")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9))
    ax.set_title("Study Summary", fontsize=12, fontweight="bold")
    ax.axis("off")
