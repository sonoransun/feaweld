"""Composite multi-panel dashboard views for engineering assessment.

Combines multiple 2D plots into unified figures that give engineers
a single-view summary of analysis results.
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
            "matplotlib is required for dashboard views. "
            "Install with: pip install feaweld[viz]"
        ) from exc


def _apply_theme():
    from feaweld.visualization.theme import apply_feaweld_style
    apply_feaweld_style()


def engineering_dashboard(
    workflow_result: Any,
    show: bool = True,
) -> Any:
    """Multi-panel engineering summary dashboard.

    Layout (2x3 grid):
      [1] Stress heatmap (2D cross-section or von Mises distribution)
      [2] Through-thickness linearization (if available)
      [3] S-N curve with operating point
      [4] ASME check / Dong decomposition (whichever is available)
      [5] Weld group geometry (if Blodgett result available)
      [6] Summary text box with key metrics

    Args:
        workflow_result: WorkflowResult from pipeline.workflow
        show: If True, display the figure interactively.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    _apply_theme()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Engineering Assessment: {workflow_result.case.name}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- Panel 1: Stress distribution ---
    ax1 = axes[0, 0]
    _panel_stress_distribution(ax1, workflow_result)

    # --- Panel 2: Through-thickness linearization ---
    ax2 = axes[0, 1]
    _panel_linearization(ax2, workflow_result)

    # --- Panel 3: S-N curve ---
    ax3 = axes[0, 2]
    _panel_sn_curve(ax3, workflow_result)

    # --- Panel 4: Dong decomposition or ASME check ---
    ax4 = axes[1, 0]
    _panel_postprocess_method(ax4, workflow_result)

    # --- Panel 5: Weld group geometry ---
    ax5 = axes[1, 1]
    _panel_weld_group(ax5, workflow_result)

    # --- Panel 6: Summary text ---
    ax6 = axes[1, 2]
    _panel_summary_text(ax6, workflow_result)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    return fig


def fatigue_dashboard(
    workflow_result: Any,
    show: bool = True,
) -> Any:
    """Fatigue-focused dashboard (2x2 grid).

    Panels:
      [1] S-N curve with computed life
      [2] Dong decomposition or hotspot extrapolation
      [3] Stress distribution (histogram or cross-section)
      [4] Summary text with Miner damage, life, safety factor

    Args:
        workflow_result: WorkflowResult from pipeline.workflow
        show: If True, display the figure interactively.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Fatigue Assessment: {workflow_result.case.name}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Panel 1: S-N curve
    _panel_sn_curve(axes[0, 0], workflow_result)

    # Panel 2: Dong or hotspot
    _panel_postprocess_method(axes[0, 1], workflow_result)

    # Panel 3: Stress distribution
    _panel_stress_distribution(axes[1, 0], workflow_result)

    # Panel 4: Summary
    _panel_fatigue_summary(axes[1, 1], workflow_result)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    return fig


def comparison_view(
    results: list[tuple[str, Any, Any]],
    component: str = "von_mises",
    show: bool = True,
) -> Any:
    """Side-by-side stress contour comparison.

    Args:
        results: List of (label, FEMesh, StressField) tuples.
        component: Stress component to plot.
        show: If True, display the figure.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    from matplotlib.tri import Triangulation

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (label, mesh, stress) in enumerate(results):
        ax = axes[i]

        # Get stress component values
        if component == "von_mises":
            vals = stress.von_mises
        elif component == "tresca":
            vals = stress.tresca
        elif component in ("xx", "yy", "zz", "xy", "yz", "xz"):
            idx = ["xx", "yy", "zz", "xy", "yz", "xz"].index(component)
            vals = stress.values[:, idx]
        else:
            vals = stress.von_mises

        x = mesh.nodes[:, 0]
        y = mesh.nodes[:, 1]

        # For triangle meshes, use tricontourf
        if mesh.elements.shape[1] >= 3:
            tri = Triangulation(x, y, mesh.elements[:, :3])
            from feaweld.visualization.theme import get_cmap
            tc = ax.tricontourf(tri, vals, levels=20, cmap=get_cmap("stress"))
            fig.colorbar(tc, ax=ax, label=f"{component} (MPa)")
        else:
            from feaweld.visualization.theme import get_cmap
            sc = ax.scatter(x, y, c=vals, cmap=get_cmap("stress"), s=5)
            fig.colorbar(sc, ax=ax, label=f"{component} (MPa)")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def postprocess_summary(
    workflow_result: Any,
    show: bool = True,
) -> Any:
    """One subplot per post-processing method that was run.

    Args:
        workflow_result: WorkflowResult
        show: If True, display the figure.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()

    pp = workflow_result.postprocess_results or {}
    n_methods = max(len(pp), 1)
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 4 * n_methods))
    if n_methods == 1:
        axes = [axes]

    for i, (method_name, result_data) in enumerate(pp.items()):
        ax = axes[i]
        ax.set_title(f"Method: {method_name}", fontsize=12, fontweight="bold")

        if isinstance(result_data, dict):
            # Render as a simple bar chart of numeric values
            keys = []
            vals = []
            for k, v in result_data.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    keys.append(k)
                    vals.append(v)
            if keys:
                colors = ["#2980b9" if v >= 0 else "#e74c3c" for v in vals]
                ax.barh(keys, vals, color=colors)
                ax.set_xlabel("Value")
            else:
                ax.text(0.5, 0.5, f"{method_name}: see report for details",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f"{method_name}: data type not plottable",
                    ha="center", va="center", transform=ax.transAxes)

    if not pp:
        axes[0].text(0.5, 0.5, "No post-processing results available",
                     ha="center", va="center", transform=axes[0].transAxes)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Internal panel builders
# ---------------------------------------------------------------------------

def _panel_stress_distribution(ax: Any, wf: Any) -> None:
    """Panel: von Mises stress distribution (histogram or 2D cross-section)."""
    plt = _require_matplotlib()

    if wf.fea_results is not None and wf.fea_results.stress is not None:
        vm = wf.fea_results.stress.von_mises
        ax.hist(vm, bins=40, color="#2980b9", edgecolor="white", alpha=0.8)
        ax.axvline(np.max(vm), color="#e74c3c", linestyle="--", linewidth=2,
                   label=f"Max: {np.max(vm):.1f} MPa")
        ax.axvline(np.mean(vm), color="#f39c12", linestyle=":", linewidth=2,
                   label=f"Mean: {np.mean(vm):.1f} MPa")
        ax.set_xlabel("von Mises Stress (MPa)")
        ax.set_ylabel("Node Count")
        ax.set_title("Stress Distribution")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_title("Stress Distribution")


def _panel_linearization(ax: Any, wf: Any) -> None:
    """Panel: through-thickness stress linearization."""
    # Check if linearization results exist in postprocess_results
    lin_data = None
    pp = wf.postprocess_results or {}
    for key, val in pp.items():
        if "linearization" in key.lower() and hasattr(val, "z_coords"):
            lin_data = val
            break

    if lin_data is not None:
        from feaweld.visualization.plots_2d import plot_through_thickness
        plot_through_thickness(lin_data, show=False, ax=ax)
    else:
        # Show stress profile if mesh + stress available
        if wf.fea_results is not None and wf.fea_results.stress is not None:
            mesh = wf.fea_results.mesh
            vm = wf.fea_results.stress.von_mises
            y_coords = mesh.nodes[:, 1]
            from feaweld.visualization.theme import get_cmap
            ax.scatter(vm, y_coords, c=vm, cmap=get_cmap("stress"), s=3, alpha=0.5)
            ax.set_xlabel("von Mises Stress (MPa)")
            ax.set_ylabel("y position (mm)")
            ax.set_title("Stress vs. Position")
        else:
            ax.text(0.5, 0.5, "No linearization data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#888")
            ax.set_title("Stress Linearization")


def _panel_sn_curve(ax: Any, wf: Any) -> None:
    """Panel: S-N curve with operating point."""
    stress_range = None
    fatigue = wf.fatigue_results or {}
    for method_data in fatigue.values():
        if isinstance(method_data, dict) and "stress_range" in method_data:
            stress_range = method_data["stress_range"]
            break

    try:
        sn_spec = wf.case.postprocess.sn_curve
        from feaweld.fatigue.sn_curves import get_sn_curve
        if "_" in sn_spec:
            parts = sn_spec.split("_", 1)
            curve = get_sn_curve(parts[0].lower(), parts[1])
        else:
            curve = get_sn_curve("iiw", sn_spec)

        from feaweld.visualization.plots_2d import plot_sn_curve
        plot_sn_curve(curve, stress_range=stress_range, show=False, ax=ax)
    except Exception:
        ax.text(0.5, 0.5, "S-N curve not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_title("S-N Curve")


def _panel_postprocess_method(ax: Any, wf: Any) -> None:
    """Panel: Dong decomposition, hotspot, or ASME check."""
    pp = wf.postprocess_results or {}

    # Try Dong decomposition first
    for key, val in pp.items():
        if "dong" in key.lower() and isinstance(val, dict):
            dong_result = val.get("dong_result")
            if dong_result is not None and hasattr(dong_result, "membrane_stress"):
                from feaweld.visualization.plots_2d import plot_dong_decomposition
                plot_dong_decomposition(dong_result, show=False, ax=ax)
                return

    # Try hotspot
    for key, val in pp.items():
        if "hotspot" in key.lower() and isinstance(val, dict):
            results_list = val.get("results", [])
            if results_list:
                from feaweld.visualization.plots_2d import plot_hotspot_extrapolation
                plot_hotspot_extrapolation(results_list[0], show=False, ax=ax)
                return

    ax.text(0.5, 0.5, "No post-processing visualization", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#888")
    ax.set_title("Post-Processing")


def _panel_weld_group(ax: Any, wf: Any) -> None:
    """Panel: weld group geometry from Blodgett results."""
    pp = wf.postprocess_results or {}

    for key, val in pp.items():
        if "blodgett" in key.lower() and isinstance(val, dict):
            props = val.get("properties")
            if props is not None:
                from feaweld.visualization.plots_2d import plot_weld_group_geometry
                from feaweld.core.types import WeldGroupShape
                # Use the geometry config to determine shape
                shape = WeldGroupShape.LINE
                d = wf.case.geometry.base_width
                b = wf.case.geometry.web_height
                plot_weld_group_geometry(shape, d, b, props=props, show=False, ax=ax)
                return

    # Default: show joint dimensions as text
    geo = wf.case.geometry
    text = (
        f"Joint: {geo.joint_type.value}\n"
        f"Base: {geo.base_width} x {geo.base_thickness} mm\n"
        f"Web: {geo.web_height} x {geo.web_thickness} mm\n"
        f"Weld leg: {geo.weld_leg_size} mm"
    )
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#eaf2f8", alpha=0.8))
    ax.set_title("Joint Geometry")
    ax.axis("off")


def _panel_summary_text(ax: Any, wf: Any) -> None:
    """Panel: key engineering metrics summary."""
    lines = []
    status = "PASS" if wf.success else "ERRORS"
    status_color = "#27ae60" if wf.success else "#e74c3c"

    lines.append(f"Status: {status}")
    lines.append("")

    # Max stress
    if wf.fea_results is not None and wf.fea_results.stress is not None:
        vm = wf.fea_results.stress.von_mises
        lines.append(f"Max von Mises:  {np.max(vm):.1f} MPa")
        lines.append(f"Mean von Mises: {np.mean(vm):.1f} MPa")

    # Max displacement
    if wf.fea_results is not None and wf.fea_results.displacement is not None:
        max_d = np.max(np.linalg.norm(wf.fea_results.displacement, axis=1))
        lines.append(f"Max displacement: {max_d:.4f} mm")

    lines.append("")

    # Fatigue results
    fatigue = wf.fatigue_results or {}
    for method, data in fatigue.items():
        if isinstance(data, dict) and "life" in data:
            lines.append(f"Fatigue ({method}):")
            lines.append(f"  Life: {data['life']:.0f} cycles")
            if "stress_range" in data:
                lines.append(f"  Stress range: {data['stress_range']:.1f} MPa")

    # Safety factor (simple estimate)
    if wf.fea_results is not None and wf.fea_results.stress is not None:
        from feaweld.core.materials import load_material
        try:
            mat = load_material(wf.case.material.base_metal)
            sy = mat.sigma_y(wf.case.material.temperature)
            max_vm = np.max(wf.fea_results.stress.von_mises)
            sf = sy / max_vm if max_vm > 0 else float("inf")
            lines.append("")
            lines.append(f"Safety factor: {sf:.2f}")
            if sf < 1.0:
                lines.append("  ** YIELD EXCEEDED **")
        except Exception:
            pass

    # Errors
    if wf.errors:
        lines.append("")
        lines.append(f"Errors: {len(wf.errors)}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9))
    ax.set_title("Summary", fontsize=12, fontweight="bold")
    ax.axis("off")


def _panel_fatigue_summary(ax: Any, wf: Any) -> None:
    """Panel: fatigue-specific summary text."""
    lines = []
    fatigue = wf.fatigue_results or {}

    lines.append(f"S-N curve: {wf.case.postprocess.sn_curve}")
    lines.append("")

    for method, data in fatigue.items():
        if isinstance(data, dict):
            lines.append(f"Method: {method}")
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    if "life" in k.lower():
                        lines.append(f"  {k}: {v:.0f} cycles")
                    elif "damage" in k.lower():
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v:.2f}")
            lines.append("")

    if not fatigue:
        lines.append("No fatigue results computed.")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.9))
    ax.set_title("Fatigue Summary", fontsize=12, fontweight="bold")
    ax.axis("off")
