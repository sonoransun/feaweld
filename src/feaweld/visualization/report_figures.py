"""Bridge between visualization and HTML report generation.

Renders plots to base64-encoded PNG images for embedding directly in
HTML reports as self-contained ``<img>`` tags. Which figures a report
gets is driven by the declarative `FIGURE_SPECS` registry: each
spec pairs an availability predicate (checked against the
``WorkflowResult``) with a builder, so figures appear exactly when
their data exists and a failing builder never breaks the report.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


def figure_to_base64(fig: Any, dpi: int = 150) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    dpi : int
        Resolution for rendering.

    Returns
    -------
    str
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

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter created with ``off_screen=True``.
    resolution : tuple[int, int]
        ``(width, height)`` in pixels.

    Returns
    -------
    str
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

    Parameters
    ----------
    base64_png : str
        Base64-encoded PNG data (no prefix).
    alt : str
        Alt text for accessibility.
    width : str
        CSS width value.

    Returns
    -------
    str
        Complete ``<img>`` tag string.
    """
    return (
        f'<img src="data:image/png;base64,{base64_png}" '
        f'alt="{alt}" style="max-width:{width}; height:auto;"/>'
    )


# ---------------------------------------------------------------------------
# Declarative figure registry
# ---------------------------------------------------------------------------

@dataclass
class FigureSpec:
    """One report figure: when it applies and how to build it.

    Attributes
    ----------
    key : str
        Stable identifier (dict key in the report context).
    caption : str
        Human-readable caption shown under the figure.
    kind : {"mpl", "pyvista"}
        Rendering backend; ``"pyvista"`` specs are skipped when pyvista
        is not installed (they also need matplotlib for PNG encoding).
    available : Callable
        Predicate on the ``WorkflowResult``; the figure is only built
        when it returns ``True``.
    build : Callable
        Returns a Matplotlib ``Figure`` (``kind="mpl"``) or an
        off-screen PyVista ``Plotter`` (``kind="pyvista"``).
    layout : {"grid", "full"}
        Placement in the report: two-column grid or full width.
    """

    key: str
    caption: str
    kind: Literal["mpl", "pyvista"]
    available: Callable[[Any], bool]
    build: Callable[[Any], Any]
    layout: Literal["grid", "full"] = "grid"


# --- predicates -------------------------------------------------------------

def _has_stress(wf: Any) -> bool:
    return wf.fea_results is not None and wf.fea_results.stress is not None


def _stress_is_nodal(wf: Any) -> bool:
    """True when a nodal stress field matches the mesh node count.

    PyVista point-data attachment and per-node indexing both require
    exactly ``mesh.n_nodes`` stress rows. Cell-located stress (one row
    per element) satisfies :func:`_has_stress` but crashes those
    builders, so figures that colour or index stress per node must be
    gated on this predicate rather than mere stress presence.

    Parameters
    ----------
    wf : WorkflowResult
        The workflow result to inspect.

    Returns
    -------
    bool
        ``True`` when stress and a mesh are present and the stress field
        has one row per mesh node.
    """
    if not _has_stress(wf) or wf.mesh is None:
        return False
    return wf.fea_results.stress.values.shape[0] == wf.mesh.n_nodes


def _has_planar_mesh(wf: Any) -> bool:
    if not _has_stress(wf) or wf.mesh is None:
        return False
    nodes = wf.mesh.nodes
    if nodes.shape[1] < 3:
        return True
    return float(np.ptp(nodes[:, 2])) < 1e-9 * max(float(np.ptp(nodes[:, 0])), 1.0)


def _has_displacement(wf: Any) -> bool:
    return wf.fea_results is not None and wf.fea_results.displacement is not None


def _has_temperature_field(wf: Any) -> bool:
    return wf.fea_results is not None and wf.fea_results.temperature is not None


def _has_temperature_history(wf: Any) -> bool:
    fea = wf.fea_results
    return (
        fea is not None
        and fea.temperature is not None
        and fea.temperature.ndim == 2
        and fea.time_steps is not None
        and len(fea.time_steps) == fea.temperature.shape[0]
    )


def _find_pp(wf: Any, inner_key: str) -> Any:
    """Return the first postprocess value dict containing *inner_key*."""
    for val in (wf.postprocess_results or {}).values():
        if isinstance(val, dict) and val.get(inner_key) is not None:
            return val
    return None


def _rainflow_cycles(wf: Any) -> list | None:
    raw = (wf.postprocess_results or {}).get("rainflow")
    if isinstance(raw, list) and raw and isinstance(raw[0], (tuple, list)):
        return raw
    return None


def _resolve_case_sn_curve(wf: Any) -> Any:
    from feaweld.fatigue.sn_curves import parse_sn_spec
    return parse_sn_spec(wf.case.postprocess.sn_curve)


def _prob(wf: Any) -> dict:
    return wf.probabilistic_results or {}


# --- builders ---------------------------------------------------------------

def _build_stress_histogram(wf: Any) -> Any:
    import matplotlib.pyplot as plt
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
    return fig


def _build_stress_contour_2d(wf: Any) -> Any:
    from feaweld.visualization.plots_2d import plot_stress_contour_2d
    return plot_stress_contour_2d(
        wf.mesh, wf.fea_results.stress, component="von_mises", show=False,
    )


def _build_stress_contour_3d(wf: Any) -> Any:
    from feaweld.visualization.stress_plots import plot_stress_field
    return plot_stress_field(
        wf.mesh, wf.fea_results.stress, component="von_mises", show=False,
    )


def _build_deformed(wf: Any) -> Any:
    from feaweld.visualization.stress_plots import plot_deformed
    # Colour by stress only when it is nodal; cell-located stress would
    # crash the point-data attachment, so fall back to an uncoloured
    # deformed shape rather than dropping the figure entirely.
    stress = wf.fea_results.stress if _stress_is_nodal(wf) else None
    return plot_deformed(
        wf.mesh, wf.fea_results.displacement,
        stress=stress, show=False,
    )


def _build_mesh_preview(wf: Any) -> Any:
    from feaweld.visualization.enhanced_3d import plot_mesh_preview
    highlight = None
    if wf.mesh.physical_groups and "weld" in wf.mesh.physical_groups:
        highlight = {"weld": "tomato"}
    return plot_mesh_preview(wf.mesh, highlight_sets=highlight, show=False)


def _build_temperature_field(wf: Any) -> Any:
    from feaweld.visualization.stress_plots import plot_temperature_field
    temp = wf.fea_results.temperature
    if temp.ndim == 2:
        temp = temp[-1]
    return plot_temperature_field(wf.mesh, temp, show=False)


def _build_temperature_history(wf: Any) -> Any:
    from feaweld.visualization.thermal_plots import plot_temperature_history
    return plot_temperature_history(
        wf.fea_results.time_steps, wf.fea_results.temperature, show=False,
    )


def _build_sn_curve(wf: Any) -> Any:
    from feaweld.visualization.plots_2d import plot_sn_curve
    curve = _resolve_case_sn_curve(wf)
    stress_range = None
    for data in (wf.fatigue_results or {}).values():
        if isinstance(data, dict) and data.get("stress_range") is not None:
            stress_range = data["stress_range"]
            break
    return plot_sn_curve(curve, stress_range=stress_range, show=False)


def _build_dong(wf: Any) -> Any:
    from feaweld.visualization.plots_2d import plot_dong_decomposition
    val = _find_pp(wf, "dong_result")
    return plot_dong_decomposition(val["dong_result"], show=False)


def _build_hotspot(wf: Any) -> Any:
    from feaweld.visualization.plots_2d import plot_hotspot_extrapolation
    for key, val in (wf.postprocess_results or {}).items():
        if "hotspot" in key and isinstance(val, dict) and val.get("results"):
            return plot_hotspot_extrapolation(val["results"][0], show=False)
    raise ValueError("no hotspot results")


def _build_weld_group(wf: Any) -> Any:
    from feaweld.core.types import WeldGroupShape
    from feaweld.visualization.plots_2d import plot_weld_group_geometry
    val = _find_pp(wf, "properties")
    return plot_weld_group_geometry(
        WeldGroupShape.LINE,
        wf.case.geometry.base_width,
        wf.case.geometry.web_height,
        props=val["properties"],
        show=False,
    )


def _build_through_thickness(wf: Any) -> Any:
    from feaweld.visualization.plots_2d import plot_through_thickness
    val = _find_pp(wf, "linearization")
    return plot_through_thickness(val["linearization"], show=False)


def _build_asme_check(wf: Any) -> Any:
    from feaweld.core.materials import load_material
    from feaweld.visualization.plots_2d import plot_asme_check
    val = _find_pp(wf, "categorization")
    material = load_material(wf.case.material.base_metal)
    T = wf.case.material.temperature
    S_y = material.sigma_y(T)
    try:
        S_m = min(material.sigma_u(T) / 2.4, S_y / 1.5)
    except (KeyError, ValueError):
        S_m = S_y / 1.5
    return plot_asme_check(val["categorization"], S_m=S_m, S_y=S_y, show=False)


def _build_rainflow(wf: Any) -> Any:
    from feaweld.visualization.fatigue_plots import plot_rainflow_histogram
    return plot_rainflow_histogram(_rainflow_cycles(wf), kind="range", show=False)


def _life_field(wf: Any) -> Any:
    curve = _resolve_case_sn_curve(wf)
    vm = wf.fea_results.stress.von_mises
    life = np.array([curve.life(max(float(s), 1e-6)) for s in vm])
    # Stress below the S-N cutoff yields infinite life; clip both ends so
    # the downstream log10 colour map (and the derived damage field) stay
    # finite. Matches the cap applied in fatigue_maps.plot_fatigue_life.
    return np.clip(
        np.nan_to_num(life, nan=1e12, posinf=1e12, neginf=1.0), 1.0, 1e12,
    )


def _build_fatigue_life_map(wf: Any) -> Any:
    from feaweld.visualization.fatigue_maps import plot_fatigue_life
    return plot_fatigue_life(wf.mesh, _life_field(wf), show=False)


def _build_damage_map(wf: Any) -> Any:
    from feaweld.visualization.fatigue_maps import plot_damage
    cycles = _rainflow_cycles(wf)
    n_total = float(sum(c[-1] for c in cycles))
    life = _life_field(wf)
    damage = n_total / np.maximum(life, 1e-12)
    return plot_damage(wf.mesh, damage, show=False)


def _build_mc_histogram(wf: Any) -> Any:
    from feaweld.visualization.probabilistic_plots import plot_mc_histogram
    prob = _prob(wf)
    return plot_mc_histogram(
        np.asarray(prob["results"]),
        percentiles=prob.get("percentiles"),
        xlabel=str(prob.get("response", "Response")).replace("_", " "),
        show=False,
    )


def _build_sobol(wf: Any) -> Any:
    from feaweld.visualization.probabilistic_plots import plot_sobol_indices
    return plot_sobol_indices(_prob(wf)["sobol"], show=False)


def _build_form(wf: Any) -> Any:
    from feaweld.visualization.probabilistic_plots import plot_form_reliability
    return plot_form_reliability(_prob(wf)["form"], show=False)


def _build_dashboard(wf: Any) -> Any:
    from feaweld.visualization.dashboard import engineering_dashboard
    return engineering_dashboard(wf, show=False)


FIGURE_SPECS: list[FigureSpec] = [
    FigureSpec(
        key="mesh_preview", caption="Finite Element Mesh", kind="pyvista",
        available=lambda wf: wf.mesh is not None,
        build=_build_mesh_preview,
    ),
    FigureSpec(
        key="stress_contour_3d", caption="Von Mises Stress Contour",
        kind="pyvista", layout="full",
        available=_stress_is_nodal,
        build=_build_stress_contour_3d,
    ),
    FigureSpec(
        key="deformed_shape", caption="Deformed Shape (scaled)", kind="pyvista",
        available=lambda wf: _has_displacement(wf) and wf.mesh is not None,
        build=_build_deformed,
    ),
    FigureSpec(
        key="stress_contour_2d", caption="Von Mises Stress Contour (section)",
        kind="mpl",
        available=lambda wf: _has_planar_mesh(wf) and _stress_is_nodal(wf),
        build=_build_stress_contour_2d,
    ),
    FigureSpec(
        key="stress_distribution", caption="Von Mises Stress Distribution",
        kind="mpl",
        available=_has_stress,
        build=_build_stress_histogram,
    ),
    FigureSpec(
        key="temperature_field", caption="Temperature Field", kind="pyvista",
        available=lambda wf: _has_temperature_field(wf) and wf.mesh is not None,
        build=_build_temperature_field,
    ),
    FigureSpec(
        key="temperature_history", caption="Peak-Node Temperature History",
        kind="mpl",
        available=_has_temperature_history,
        build=_build_temperature_history,
    ),
    FigureSpec(
        key="sn_curve", caption="S-N Fatigue Curve", kind="mpl",
        available=lambda wf: bool(wf.case.postprocess.sn_curve),
        build=_build_sn_curve,
    ),
    FigureSpec(
        key="dong_decomposition", caption="Dong Structural Stress Decomposition",
        kind="mpl",
        available=lambda wf: _find_pp(wf, "dong_result") is not None,
        build=_build_dong,
    ),
    FigureSpec(
        key="hotspot_extrapolation", caption="Hot-Spot Stress Extrapolation",
        kind="mpl",
        available=lambda wf: any(
            "hotspot" in k and isinstance(v, dict) and v.get("results")
            for k, v in (wf.postprocess_results or {}).items()
        ),
        build=_build_hotspot,
    ),
    FigureSpec(
        key="through_thickness", caption="Through-Thickness Stress Linearization",
        kind="mpl",
        available=lambda wf: _find_pp(wf, "linearization") is not None,
        build=_build_through_thickness,
    ),
    FigureSpec(
        key="asme_check", caption="ASME VIII Div 2 Stress Check", kind="mpl",
        available=lambda wf: _find_pp(wf, "categorization") is not None,
        build=_build_asme_check,
    ),
    FigureSpec(
        key="weld_group", caption="Weld Group Geometry", kind="mpl",
        available=lambda wf: _find_pp(wf, "properties") is not None,
        build=_build_weld_group,
    ),
    FigureSpec(
        key="rainflow", caption="Rainflow Cycle Histogram", kind="mpl",
        available=lambda wf: _rainflow_cycles(wf) is not None,
        build=_build_rainflow,
    ),
    FigureSpec(
        key="fatigue_life_map", caption="Fatigue Life Map", kind="pyvista",
        available=lambda wf: _stress_is_nodal(wf)
        and wf.case.postprocess.fatigue_assessment,
        build=_build_fatigue_life_map,
    ),
    FigureSpec(
        key="damage_map", caption="Miner Damage Map", kind="pyvista",
        available=lambda wf: _stress_is_nodal(wf)
        and _rainflow_cycles(wf) is not None,
        build=_build_damage_map,
    ),
    FigureSpec(
        key="mc_histogram", caption="Monte Carlo Response Distribution",
        kind="mpl",
        available=lambda wf: isinstance(_prob(wf).get("results"), np.ndarray),
        build=_build_mc_histogram,
    ),
    FigureSpec(
        key="sobol_indices", caption="Sobol Sensitivity Indices", kind="mpl",
        available=lambda wf: isinstance(_prob(wf).get("sobol"), dict)
        and "first_order" in _prob(wf)["sobol"],
        build=_build_sobol,
    ),
    FigureSpec(
        key="form_reliability", caption="FORM Reliability (design point)",
        kind="mpl",
        available=lambda wf: isinstance(_prob(wf).get("form"), dict)
        and "beta" in _prob(wf)["form"],
        build=_build_form,
    ),
    FigureSpec(
        key="engineering_dashboard", caption="Engineering Assessment Dashboard",
        kind="mpl", layout="full",
        available=lambda wf: True,
        build=_build_dashboard,
    ),
]


def generate_report_figures(
    workflow_result: Any,
) -> dict[str, dict[str, str]]:
    """Generate all applicable figures for a workflow result.

    Walks `FIGURE_SPECS`, building every figure whose data is
    present. A failing predicate or builder skips that figure only —
    the report never breaks on a visualization error.

    Parameters
    ----------
    workflow_result : WorkflowResult
        Result from [feaweld.pipeline.workflow.run_analysis][].

    Returns
    -------
    dict
        Mapping of figure key to ``{"b64", "caption", "layout"}``.
        Empty when matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return {}  # matplotlib not installed — no figures

    try:
        import pyvista  # noqa: F401
        has_pyvista = True
    except ImportError:
        has_pyvista = False

    figures: dict[str, dict[str, str]] = {}
    for spec in FIGURE_SPECS:
        if spec.kind == "pyvista" and not has_pyvista:
            continue
        try:
            if not spec.available(workflow_result):
                continue
            artifact = spec.build(workflow_result)
            if spec.kind == "pyvista":
                b64 = plotter_to_base64(artifact)
            else:
                dpi = 120 if spec.layout == "full" else 150
                b64 = figure_to_base64(artifact, dpi=dpi)
            figures[spec.key] = {
                "b64": b64,
                "caption": spec.caption,
                "layout": spec.layout,
            }
        except Exception:
            continue

    return figures
