"""Metric extraction, delta computation, and comparison report generation.

Provides tools to extract comparable scalar metrics from multiple
WorkflowResult objects, compute differences, and generate HTML
comparison reports.
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import StressField
from feaweld.pipeline.workflow import WorkflowResult


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

@dataclass
class MetricSet:
    """Scalar engineering metrics extracted from one analysis result."""
    max_von_mises: float | None = None
    mean_von_mises: float | None = None
    max_displacement: float | None = None
    max_tresca: float | None = None
    fatigue_life: float | None = None
    safety_factor: float | None = None
    hotspot_stress: float | None = None
    max_principal_1: float | None = None
    n_nodes: int | None = None
    n_elements: int | None = None

    @classmethod
    def from_workflow_result(
        cls,
        result: WorkflowResult,
        allowable_stress: float | None = None,
    ) -> MetricSet:
        """Extract all available metrics from a WorkflowResult."""
        m = cls()

        if result.mesh is not None:
            m.n_nodes = result.mesh.n_nodes
            m.n_elements = result.mesh.n_elements

        fea = result.fea_results
        if fea is not None and fea.stress is not None:
            vm = fea.stress.von_mises
            m.max_von_mises = float(np.max(vm))
            m.mean_von_mises = float(np.mean(vm))
            m.max_tresca = float(np.max(fea.stress.tresca))
            principals = fea.stress.principal
            m.max_principal_1 = float(np.max(principals[:, 2]))

            if allowable_stress is not None and m.max_von_mises > 0:
                m.safety_factor = allowable_stress / m.max_von_mises

        if fea is not None and fea.displacement is not None:
            m.max_displacement = float(
                np.max(np.linalg.norm(fea.displacement, axis=1))
            )

        # Extract fatigue life from fatigue_results
        for method_data in (result.fatigue_results or {}).values():
            if isinstance(method_data, dict) and "life" in method_data:
                life = method_data["life"]
                if m.fatigue_life is None or life < m.fatigue_life:
                    m.fatigue_life = life

        # Extract hotspot stress
        pp = result.postprocess_results or {}
        for key, val in pp.items():
            if "hotspot" in key.lower() and isinstance(val, dict):
                hs = val.get("max_stress")
                if hs is not None:
                    m.hotspot_stress = hs

        return m

    def to_dict(self) -> dict[str, float | None]:
        """Return all metrics as a flat dict."""
        return {
            "max_von_mises": self.max_von_mises,
            "mean_von_mises": self.mean_von_mises,
            "max_displacement": self.max_displacement,
            "max_tresca": self.max_tresca,
            "fatigue_life": self.fatigue_life,
            "safety_factor": self.safety_factor,
            "hotspot_stress": self.hotspot_stress,
            "max_principal_1": self.max_principal_1,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
        }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

@dataclass
class ComparisonTable:
    """Tabular comparison of metrics across multiple cases."""
    case_names: list[str]
    metrics: dict[str, MetricSet]  # keyed by case name

    def to_rows(self) -> list[dict[str, Any]]:
        """Return list of flat dicts, one per case."""
        rows = []
        for name in self.case_names:
            row = {"case": name}
            row.update(self.metrics[name].to_dict())
            rows.append(row)
        return rows

    def to_text_table(self) -> str:
        """Formatted ASCII table."""
        rows = self.to_rows()
        if not rows:
            return "(no data)"

        # Determine columns
        columns = list(rows[0].keys())
        # Column widths
        widths = {col: max(len(col), 8) for col in columns}
        for row in rows:
            for col in columns:
                val = row[col]
                formatted = _fmt_value(val)
                widths[col] = max(widths[col], len(formatted))

        # Header
        header = " | ".join(col.rjust(widths[col]) for col in columns)
        separator = "-+-".join("-" * widths[col] for col in columns)
        lines = [header, separator]

        # Data rows
        for row in rows:
            line = " | ".join(
                _fmt_value(row[col]).rjust(widths[col]) for col in columns
            )
            lines.append(line)

        return "\n".join(lines)

    def delta_from_baseline(self, baseline: str) -> list[dict[str, Any]]:
        """Compute deltas relative to a baseline case.

        Returns rows with absolute delta and percentage change for each metric.
        """
        if baseline not in self.metrics:
            raise ValueError(f"Baseline '{baseline}' not found in cases")

        base = self.metrics[baseline].to_dict()
        deltas = []

        for name in self.case_names:
            if name == baseline:
                continue
            row = {"case": name}
            current = self.metrics[name].to_dict()
            for key in base:
                bv = base[key]
                cv = current[key]
                if bv is not None and cv is not None and isinstance(bv, (int, float)):
                    abs_delta = cv - bv
                    pct_delta = (abs_delta / abs(bv) * 100) if abs(bv) > 1e-12 else 0.0
                    row[f"{key}_delta"] = abs_delta
                    row[f"{key}_pct"] = pct_delta
                else:
                    row[f"{key}_delta"] = None
                    row[f"{key}_pct"] = None
            deltas.append(row)

        return deltas


def _fmt_value(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) >= 1e5 or (0 < abs(val) < 0.01):
            return f"{val:.2e}"
        return f"{val:.2f}"
    return str(val)


# ---------------------------------------------------------------------------
# Stress field difference
# ---------------------------------------------------------------------------

def compute_stress_field_difference(
    stress_a: StressField,
    stress_b: StressField,
) -> StressField:
    """Compute element-wise stress difference (A - B).

    Requires both fields to have the same number of points.

    Args:
        stress_a: First stress field.
        stress_b: Second stress field.

    Returns:
        StressField with values = stress_a.values - stress_b.values

    Raises:
        ValueError: If the stress fields have different shapes.
    """
    if stress_a.values.shape != stress_b.values.shape:
        raise ValueError(
            f"Incompatible stress field shapes: "
            f"{stress_a.values.shape} vs {stress_b.values.shape}. "
            f"Fields must be defined on the same mesh."
        )
    return StressField(values=stress_a.values - stress_b.values)


# ---------------------------------------------------------------------------
# Comparison report generation
# ---------------------------------------------------------------------------

def generate_comparison_report(
    study_results: Any,  # StudyResults
    output_dir: str | Path,
    baseline: str | None = None,
) -> str:
    """Generate an HTML comparison report for a parametric study.

    Args:
        study_results: StudyResults from Study.run()
        output_dir: Directory to write the report
        baseline: Name of baseline case for delta computation.

    Returns:
        Path to the generated HTML file.
    """
    from feaweld import __version__

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    case_names = list(study_results.results.keys())
    metrics = {
        name: MetricSet.from_workflow_result(result)
        for name, result in study_results.results.items()
    }
    table = ComparisonTable(case_names=case_names, metrics=metrics)

    content_parts = []

    # Study overview
    content_parts.append(_section_overview(study_results))

    # Metric comparison table
    content_parts.append(_section_metric_table(table))

    # Delta table if baseline specified
    if baseline and baseline in metrics:
        content_parts.append(_section_delta_table(table, baseline))

    # Embedded figures
    try:
        figures = _generate_comparison_figures(study_results, table, baseline)
        if figures:
            content_parts.append(_section_figures(figures))
    except ImportError:
        pass

    # Errors
    if study_results.errors:
        error_html = "".join(
            f"<li><b>{html_mod.escape(k)}:</b> {html_mod.escape(v)}</li>"
            for k, v in study_results.errors.items()
        )
        content_parts.append(f"""
        <div class="error-box">
            <h2>Failed Cases ({len(study_results.errors)})</h2>
            <ul>{error_html}</ul>
        </div>
        """)

    content = "\n".join(content_parts)

    report_html = _COMPARISON_TEMPLATE.replace("{{ title }}", html_mod.escape(study_results.study_name))
    report_html = report_html.replace("{{ content }}", content)
    report_html = report_html.replace("{{ version }}", __version__)
    report_html = report_html.replace("{{ timestamp }}", datetime.now().isoformat())

    report_path = out_dir / f"{study_results.study_name}_comparison.html"
    report_path.write_text(report_html)
    return str(report_path)


# ---------------------------------------------------------------------------
# HTML template and section builders
# ---------------------------------------------------------------------------

_COMPARISON_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>feaweld Comparison Report - {{ title }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }
        h1 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }
        h2 { color: #2471a3; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
        th { background-color: #2980b9; color: white; text-align: center; }
        td:first-child { text-align: left; font-weight: bold; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .positive { color: #e74c3c; }
        .negative { color: #27ae60; }
        .summary-box { background: #eaf2f8; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .error-box { background: #fadbd8; padding: 15px; border-radius: 5px; }
        .figure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }
        .figure-item { text-align: center; }
        .figure-item img { border: 1px solid #ddd; border-radius: 4px; max-width: 100%; }
        .figure-caption { font-style: italic; color: #555; margin-top: 5px; }
        footer { margin-top: 40px; color: #888; font-size: 0.9em; border-top: 1px solid #ddd; padding-top: 10px; }
    </style>
</head>
<body>
    <h1>Parametric Study Comparison: {{ title }}</h1>
    {{ content }}
    <footer>Generated by feaweld v{{ version }} on {{ timestamp }}</footer>
</body>
</html>"""


def _section_overview(sr: Any) -> str:
    return f"""
    <div class="summary-box">
        <h2>Study Overview</h2>
        <strong>Name:</strong> {html_mod.escape(sr.study_name)}<br>
        <strong>Total cases:</strong> {sr.n_cases}<br>
        <strong>Succeeded:</strong> {sr.n_succeeded}<br>
        <strong>Failed:</strong> {sr.n_failed}<br>
        <strong>Elapsed:</strong> {sr.elapsed_seconds:.1f} s
    </div>
    """


def _section_metric_table(table: ComparisonTable) -> str:
    rows = table.to_rows()
    if not rows:
        return "<p>No results to compare.</p>"

    # Metric display names
    display = {
        "case": "Case",
        "max_von_mises": "Max VM (MPa)",
        "mean_von_mises": "Mean VM (MPa)",
        "max_displacement": "Max Disp (mm)",
        "max_tresca": "Max Tresca (MPa)",
        "fatigue_life": "Fatigue Life (N)",
        "safety_factor": "Safety Factor",
        "hotspot_stress": "Hotspot (MPa)",
        "max_principal_1": "Max P1 (MPa)",
        "n_nodes": "Nodes",
        "n_elements": "Elements",
    }

    cols = [c for c in rows[0].keys() if rows[0][c] is not None or c == "case"]
    header = "".join(f"<th>{display.get(c, c)}</th>" for c in cols)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{_fmt_value(row[c])}</td>" for c in cols)
        body += f"<tr>{cells}</tr>"

    return f"""
    <div class="section">
        <h2>Metric Comparison</h2>
        <table><tr>{header}</tr>{body}</table>
    </div>
    """


def _section_delta_table(table: ComparisonTable, baseline: str) -> str:
    deltas = table.delta_from_baseline(baseline)
    if not deltas:
        return ""

    # Find metric keys
    metric_keys = [k.replace("_delta", "") for k in deltas[0] if k.endswith("_delta")]
    display = {
        "max_von_mises": "Max VM",
        "mean_von_mises": "Mean VM",
        "max_displacement": "Max Disp",
        "fatigue_life": "Fatigue Life",
        "safety_factor": "Safety Factor",
    }

    header = "<th>Case</th>"
    for mk in metric_keys:
        if mk in display:
            header += f"<th>{display[mk]} delta</th><th>%</th>"

    body = ""
    for row in deltas:
        cells = f"<td>{html_mod.escape(row['case'])}</td>"
        for mk in metric_keys:
            if mk not in display:
                continue
            d = row.get(f"{mk}_delta")
            p = row.get(f"{mk}_pct")
            if d is not None:
                cls = "positive" if d > 0 else "negative"
                cells += f'<td class="{cls}">{_fmt_value(d)}</td>'
                cells += f'<td class="{cls}">{p:+.1f}%</td>'
            else:
                cells += "<td>-</td><td>-</td>"
        body += f"<tr>{cells}</tr>"

    return f"""
    <div class="section">
        <h2>Deltas vs. Baseline ({html_mod.escape(baseline)})</h2>
        <table><tr>{header}</tr>{body}</table>
    </div>
    """


def _section_figures(figures: dict[str, str]) -> str:
    from feaweld.visualization.report_figures import html_img_tag

    captions = {
        "metric_comparison": "Key Metric Comparison",
        "parameter_sensitivity": "Parameter Sensitivity",
        "stress_overlay": "Stress Distribution Overlay",
        "comparison_dashboard": "Comparison Dashboard",
    }

    items = ""
    for key, b64 in figures.items():
        cap = captions.get(key, key.replace("_", " ").title())
        img = html_img_tag(b64, alt=cap)
        items += f'<div class="figure-item">{img}<div class="figure-caption">{cap}</div></div>\n'

    return f"""
    <div class="section">
        <h2>Visualizations</h2>
        <div class="figure-grid">{items}</div>
    </div>
    """


def _generate_comparison_figures(
    study_results: Any,
    table: ComparisonTable,
    baseline: str | None,
) -> dict[str, str]:
    """Generate base64 figures for the comparison report."""
    import matplotlib
    matplotlib.use("Agg")

    from feaweld.visualization.report_figures import figure_to_base64

    figures: dict[str, str] = {}

    try:
        from feaweld.visualization.comparison import plot_metric_comparison
        fig = plot_metric_comparison(study_results, "max_von_mises", show=False)
        figures["metric_comparison"] = figure_to_base64(fig)
    except Exception:
        pass

    try:
        from feaweld.visualization.comparison import plot_stress_envelope
        fig = plot_stress_envelope(study_results, show=False)
        figures["stress_overlay"] = figure_to_base64(fig)
    except Exception:
        pass

    try:
        from feaweld.visualization.comparison import comparison_dashboard
        fig = comparison_dashboard(study_results, baseline=baseline, show=False)
        figures["comparison_dashboard"] = figure_to_base64(fig)
    except Exception:
        pass

    return figures
