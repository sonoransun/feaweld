"""Visualization tools for weld FEA results.

The package's public plotting API is re-exported lazily (PEP 562):
``from feaweld.visualization import plot_stress_field`` works without
importing matplotlib or pyvista until the function is actually used.
Heavy dependencies stay optional — importing this package costs nothing.
"""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, str] = {
    # theme
    "FEAWELD_BLUE": "theme",
    "FEAWELD_RED": "theme",
    "FEAWELD_ORANGE": "theme",
    "FEAWELD_GREEN": "theme",
    "FEAWELD_DARK": "theme",
    "FEAWELD_GRAY": "theme",
    "FEAWELD_LIGHT_BG": "theme",
    "get_cmap": "theme",
    "apply_feaweld_style": "theme",
    "configure_plotter": "theme",
    # plots_2d
    "plot_through_thickness": "plots_2d",
    "plot_hotspot_extrapolation": "plots_2d",
    "plot_dong_decomposition": "plots_2d",
    "plot_sn_curve": "plots_2d",
    "plot_stress_along_path": "plots_2d",
    "plot_weld_group_geometry": "plots_2d",
    "plot_asme_check": "plots_2d",
    "plot_cross_section_stress": "plots_2d",
    "plot_stress_contour_2d": "plots_2d",
    "plot_mesh_convergence": "plots_2d",
    # stress_plots (PyVista)
    "stress_field_to_pyvista": "stress_plots",
    "resolve_component": "stress_plots",
    "resolve_grid": "stress_plots",
    "plot_stress_field": "stress_plots",
    "plot_deformed": "stress_plots",
    "plot_temperature_field": "stress_plots",
    # enhanced_3d (PyVista)
    "plot_stress_with_clipping": "enhanced_3d",
    "plot_stress_threshold": "enhanced_3d",
    "plot_iso_surface": "enhanced_3d",
    "plot_force_vectors": "enhanced_3d",
    "plot_weld_region_highlight": "enhanced_3d",
    "plot_sed_control_volume": "enhanced_3d",
    "plot_mesh_preview": "enhanced_3d",
    "plot_annotated_stress": "enhanced_3d",
    # fatigue
    "plot_rainflow_histogram": "fatigue_plots",
    "plot_sn_damage_stacked": "fatigue_plots",
    "animate_damage_evolution": "fatigue_plots",
    "plot_fatigue_life": "fatigue_maps",
    "plot_damage": "fatigue_maps",
    # thermal
    "render_goldak_source": "thermal_plots",
    "plot_temperature_history": "thermal_plots",
    # probabilistic
    "plot_sobol_indices": "probabilistic_plots",
    "plot_mc_histogram": "probabilistic_plots",
    "plot_form_reliability": "probabilistic_plots",
    # material / metallurgy
    "plot_cct_diagram": "material_plots",
    "plot_residual_stress_profile": "material_plots",
    "plot_creep_curve": "material_plots",
    "plot_hall_petch": "material_plots",
    # study comparison
    "detect_swept_parameters": "comparison",
    "plot_metric_comparison": "comparison",
    "plot_parameter_sensitivity": "comparison",
    "plot_stress_difference": "comparison",
    "plot_stress_envelope": "comparison",
    "comparison_dashboard": "comparison",
    # composite dashboards
    "engineering_dashboard": "dashboard",
    "fatigue_dashboard": "dashboard",
    "comparison_view": "dashboard",
    "postprocess_summary": "dashboard",
    # annotations
    "CriticalPoint": "annotations",
    "find_critical_points": "annotations",
    "singularity_warning_markers": "annotations",
    "safety_factor_field": "annotations",
    "annotate_2d": "annotations",
    "annotate_3d": "annotations",
    "format_engineering_value": "annotations",
    # export
    "export_vtk": "export",
    "export_png": "export",
    "export_gltf": "export",
    # report bridge
    "FigureSpec": "report_figures",
    "FIGURE_SPECS": "report_figures",
    "figure_to_base64": "report_figures",
    "plotter_to_base64": "report_figures",
    "html_img_tag": "report_figures",
    "generate_report_figures": "report_figures",
    # interactive (Plotly)
    "stress_histogram_plotly": "plotly_figures",
    "sn_curve_plotly": "plotly_figures",
    "rainflow_plotly": "plotly_figures",
    "convergence_plotly": "plotly_figures",
    "generate_interactive_figures": "plotly_figures",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'feaweld.visualization' has no attribute {name!r}"
        ) from None
    import importlib
    module = importlib.import_module(f"feaweld.visualization.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
