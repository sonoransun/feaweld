"""Bridge between visualization and HTML report generation.

Renders plots to base64-encoded PNG images for embedding directly
in HTML reports as self-contained ``<img>`` tags.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def figure_to_base64(fig: Any, dpi: int = 150) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG string.

    Args:
        fig: matplotlib.figure.Figure
        dpi: Resolution for rendering.

    Returns:
        Base64-encoded PNG string (no data-URI prefix).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()

    # Close the figure to free memory
    import matplotlib.pyplot as plt
    plt.close(fig)

    return data


def plotter_to_base64(
    plotter: Any,
    resolution: tuple[int, int] = (800, 600),
) -> str:
    """Render a PyVista plotter off-screen and return base64 PNG.

    Args:
        plotter: pyvista.Plotter (must have been created with off_screen=True)
        resolution: (width, height) in pixels.

    Returns:
        Base64-encoded PNG string.
    """
    img = plotter.screenshot(return_img=True, window_size=resolution)
    plotter.close()

    # Convert ndarray to PNG bytes via matplotlib's imsave
    buf = io.BytesIO()
    import matplotlib.pyplot as plt
    plt.imsave(buf, img, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    return data


def html_img_tag(base64_png: str, alt: str = "", width: str = "100%") -> str:
    """Return an HTML ``<img>`` tag with embedded base64 data.

    Args:
        base64_png: Base64-encoded PNG data (no prefix).
        alt: Alt text for accessibility.
        width: CSS width value.

    Returns:
        Complete ``<img>`` tag string.
    """
    return (
        f'<img src="data:image/png;base64,{base64_png}" '
        f'alt="{alt}" style="max-width:{width}; height:auto;"/>'
    )


def generate_report_figures(
    workflow_result: Any,
) -> dict[str, str]:
    """Generate all applicable figures for a workflow result.

    Inspects the available data in *workflow_result* and generates
    plots for each data type that is present. Missing data is silently
    skipped.

    Args:
        workflow_result: WorkflowResult from feaweld.pipeline.workflow

    Returns:
        Dict mapping descriptive names to base64-encoded PNG strings.
        Example keys: ``"stress_distribution"``, ``"sn_curve"``,
        ``"dong_decomposition"``, ``"engineering_dashboard"``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}  # matplotlib not installed — no figures

    figures: dict[str, str] = {}
    wf = workflow_result

    # --- Stress distribution histogram ---
    if wf.fea_results is not None and wf.fea_results.stress is not None:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            vm = wf.fea_results.stress.von_mises
            ax.hist(vm, bins=40, color="#2980b9", edgecolor="white", alpha=0.8)
            ax.axvline(np.max(vm), color="#e74c3c", linestyle="--", lw=2,
                       label=f"Max: {np.max(vm):.1f} MPa")
            ax.axvline(np.mean(vm), color="#f39c12", linestyle=":", lw=2,
                       label=f"Mean: {np.mean(vm):.1f} MPa")
            ax.set_xlabel("von Mises Stress (MPa)")
            ax.set_ylabel("Node Count")
            ax.set_title("Stress Distribution")
            ax.legend()
            figures["stress_distribution"] = figure_to_base64(fig)
        except Exception:
            pass

    # --- S-N curve ---
    try:
        sn_spec = wf.case.postprocess.sn_curve
        from feaweld.fatigue.sn_curves import get_sn_curve
        if "_" in sn_spec:
            parts = sn_spec.split("_", 1)
            curve = get_sn_curve(parts[0].lower(), parts[1])
        else:
            curve = get_sn_curve("iiw", sn_spec)

        # Find operating stress range from fatigue results
        stress_range = None
        for data in (wf.fatigue_results or {}).values():
            if isinstance(data, dict) and "stress_range" in data:
                stress_range = data["stress_range"]
                break

        from feaweld.visualization.plots_2d import plot_sn_curve
        fig = plot_sn_curve(curve, stress_range=stress_range, show=False)
        figures["sn_curve"] = figure_to_base64(fig)
    except Exception:
        pass

    # --- Dong decomposition ---
    pp = wf.postprocess_results or {}
    for key, val in pp.items():
        if "dong" in key.lower() and isinstance(val, dict):
            dong_result = val.get("dong_result")
            if dong_result is not None and hasattr(dong_result, "membrane_stress"):
                try:
                    from feaweld.visualization.plots_2d import plot_dong_decomposition
                    fig = plot_dong_decomposition(dong_result, show=False)
                    figures["dong_decomposition"] = figure_to_base64(fig)
                except Exception:
                    pass
                break

    # --- Hotspot extrapolation ---
    for key, val in pp.items():
        if "hotspot" in key.lower() and isinstance(val, dict):
            results_list = val.get("results", [])
            if results_list:
                try:
                    from feaweld.visualization.plots_2d import plot_hotspot_extrapolation
                    fig = plot_hotspot_extrapolation(results_list[0], show=False)
                    figures["hotspot_extrapolation"] = figure_to_base64(fig)
                except Exception:
                    pass
                break

    # --- Weld group geometry ---
    for key, val in pp.items():
        if "blodgett" in key.lower() and isinstance(val, dict):
            props = val.get("properties")
            if props is not None:
                try:
                    from feaweld.visualization.plots_2d import plot_weld_group_geometry
                    from feaweld.core.types import WeldGroupShape
                    d = wf.case.geometry.base_width
                    b = wf.case.geometry.web_height
                    fig = plot_weld_group_geometry(
                        WeldGroupShape.LINE, d, b, props=props, show=False,
                    )
                    figures["weld_group"] = figure_to_base64(fig)
                except Exception:
                    pass
                break

    # --- Engineering dashboard (composite) ---
    try:
        from feaweld.visualization.dashboard import engineering_dashboard
        fig = engineering_dashboard(wf, show=False)
        figures["engineering_dashboard"] = figure_to_base64(fig, dpi=120)
    except Exception:
        pass

    return figures
