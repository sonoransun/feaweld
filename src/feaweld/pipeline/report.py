"""HTML report generation for analysis results.

Reports are rendered from Jinja2 templates in
``feaweld/pipeline/templates/`` with all figures embedded as base64
PNGs, so a report is a single self-contained HTML file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from feaweld.pipeline.workflow import WorkflowResult

_env = None


def _get_env():
    """Return the shared Jinja2 environment (lazily created)."""
    global _env
    if _env is None:
        from jinja2 import Environment, PackageLoader, select_autoescape
        _env = Environment(
            loader=PackageLoader("feaweld.pipeline", "templates"),
            autoescape=select_autoescape(["html", "j2"]),
        )
    return _env


def generate_report(
    result: WorkflowResult,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> str:
    """Generate an HTML report from workflow results.

    Parameters
    ----------
    result : WorkflowResult
        Completed workflow result from [run_analysis][feaweld.pipeline.workflow.run_analysis].
    output_dir : str or Path, optional
        Directory to write the report file. Defaults to
        ``result.case.output_dir``.
    interactive : bool
        If ``True``, embed Plotly figures (hover / zoom / legend
        toggle) alongside the static base64 PNGs. The Plotly runtime
        is loaded once from a CDN. Default ``False``.

    Returns
    -------
    str
        Absolute path to the generated HTML report.
    """
    from feaweld import __version__

    out_dir = Path(output_dir or result.case.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case = result.case
    ctx: dict[str, Any] = {
        "title": case.name,
        "body_class": "report",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "case": {
            "name": case.name,
            "description": case.description,
            "joint_type": case.geometry.joint_type.value,
            "base_metal": case.material.base_metal,
            "weld_metal": case.material.weld_metal,
            "solver_type": case.solver.solver_type.value,
            "backend": case.solver.backend,
            "success": result.success,
        },
        "mesh": None,
        "fea_rows": [],
        "postprocess_rows": _postprocess_rows(result),
        "fatigue_rows": _fatigue_rows(result),
        "probabilistic_rows": _probabilistic_rows(result),
        "figures": [],
        "interactive_figures": [],
        "plotly_cdn": False,
        "warnings": list(getattr(result, "warnings", []) or []),
        "errors": list(result.errors),
    }

    if result.mesh is not None:
        mesh = result.mesh
        ctx["mesh"] = {
            "n_nodes": mesh.n_nodes,
            "n_elements": mesh.n_elements,
            "element_type": mesh.element_type.value,
            "ndim": mesh.ndim,
            "physical_groups": ", ".join(mesh.physical_groups.keys()),
        }

    if result.fea_results is not None:
        ctx["fea_rows"] = _fea_rows(result)

    # Static figures (gracefully skipped if matplotlib not installed)
    try:
        from feaweld.visualization.report_figures import generate_report_figures
        figure_map = generate_report_figures(result)
        ctx["figures"] = [
            {"key": key, **entry} for key, entry in figure_map.items()
        ]
    except ImportError:
        pass  # matplotlib not available — skip figures
    except Exception:
        pass  # other errors — don't break report generation

    # Interactive Plotly figures (optional; opt-in via interactive=True)
    if interactive:
        try:
            from feaweld.visualization.plotly_figures import (
                generate_interactive_figures,
            )
            interactive_figs = generate_interactive_figures(result)
            if interactive_figs:
                ctx["plotly_cdn"] = True
                ctx["interactive_figures"] = [
                    {
                        "fragment": fragment,
                        "caption": key.replace("_", " ").title()
                        + " (interactive)",
                    }
                    for key, fragment in interactive_figs.items()
                ]
        except ImportError:
            pass  # plotly not available — skip
        except Exception:
            pass

    template = _get_env().get_template("report.html.j2")
    report_html = template.render(**ctx)

    report_path = out_dir / f"{case.name}_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    result.report_path = str(report_path)
    return str(report_path)


def _fea_rows(result: WorkflowResult) -> list[tuple[str, str]]:
    """Summary rows for the FEA results table."""
    fea = result.fea_results
    rows: list[tuple[str, str]] = [
        ("Analysis Type", "Transient" if fea.is_transient else "Static"),
    ]

    if fea.stress is not None:
        vm = fea.stress.von_mises
        rows.append(("Max von Mises Stress", f"{np.max(vm):.2f} MPa"))
        rows.append(("Min von Mises Stress", f"{np.min(vm):.2f} MPa"))
        rows.append(("Mean von Mises Stress", f"{np.mean(vm):.2f} MPa"))

    if fea.displacement is not None:
        max_disp = np.max(np.linalg.norm(fea.displacement, axis=1))
        rows.append(("Max Displacement", f"{max_disp:.4f} mm"))

    if fea.temperature is not None:
        temp = fea.temperature
        rows.append(("Max Temperature", f"{np.max(temp):.1f} C"))
        rows.append(("Min Temperature", f"{np.min(temp):.1f} C"))

    if "as_welded_max_von_mises" in fea.metadata:
        rows.append((
            "As-Welded Max von Mises (before PWHT)",
            f"{fea.metadata['as_welded_max_von_mises']:.2f} MPa",
        ))

    return rows


def _postprocess_rows(result: WorkflowResult) -> list[tuple[str, str, str]]:
    rows = []
    for method, pp in result.postprocess_results.items():
        if isinstance(pp, dict):
            for key, value in pp.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    rows.append((method, key, f"{value:.4f}"))
    return rows


def _fatigue_rows(result: WorkflowResult) -> list[tuple[str, str, str]]:
    rows = []
    for key, value in result.fatigue_results.items():
        if isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    rows.append((key, k2, f"{v2:.2f}"))
                elif isinstance(v2, str):
                    rows.append((key, k2, v2))
        elif isinstance(value, (str, int, float)):
            rows.append((key, "", str(value)))
    return rows


def _probabilistic_rows(result: WorkflowResult) -> list[tuple[str, str]]:
    rows = []
    for key, value in result.probabilistic_results.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append((key, f"{value:.4f}"))
        elif isinstance(value, bool):
            rows.append((key, "yes" if value else "no"))
        elif key == "percentiles" and isinstance(value, dict):
            for p, v in sorted(value.items()):
                rows.append((f"P{p}", f"{v:.4f}"))
    return rows
