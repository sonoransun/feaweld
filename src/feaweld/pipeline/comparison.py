"""Metric extraction, delta computation, and comparison report generation.

Provides tools to extract comparable scalar metrics from multiple
WorkflowResult objects, compute differences, and generate HTML
comparison reports.
"""

from __future__ import annotations

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

    Parameters
    ----------
    stress_a : StressField
        First stress field.
    stress_b : StressField
        Second stress field.

    Returns
    -------
    StressField
        StressField with values = stress_a.values - stress_b.values

    Raises
    ------
    ValueError
        If the stress fields have different shapes.
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

    Parameters
    ----------
    study_results : Any
        StudyResults from Study.run()
    output_dir : str | Path
        Directory to write the report
    baseline : str | None
        Name of baseline case for delta computation.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    from feaweld import __version__
    from feaweld.pipeline.report import _get_env

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    case_names = list(study_results.results.keys())
    metrics = {
        name: MetricSet.from_workflow_result(result)
        for name, result in study_results.results.items()
    }
    table = ComparisonTable(case_names=case_names, metrics=metrics)

    ctx: dict[str, Any] = {
        "title": study_results.study_name,
        "body_class": "comparison",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "overview": {
            "study_name": study_results.study_name,
            "n_cases": study_results.n_cases,
            "n_succeeded": study_results.n_succeeded,
            "n_failed": study_results.n_failed,
            "elapsed": f"{study_results.elapsed_seconds:.1f}",
        },
        "metric_table": _metric_table_ctx(table),
        "delta_table": (
            _delta_table_ctx(table, baseline)
            if baseline and baseline in metrics else None
        ),
        "figures": [],
        "failed_cases": [
            (name, message) for name, message in study_results.errors.items()
        ],
    }

    # Embedded figures
    try:
        figure_map = _generate_comparison_figures(study_results, table, baseline)
        ctx["figures"] = [
            {"key": key, "layout": "grid", **entry}
            for key, entry in figure_map.items()
        ]
    except ImportError:
        pass

    template = _get_env().get_template("comparison.html.j2")
    report_html = template.render(**ctx)

    report_path = out_dir / f"{study_results.study_name}_comparison.html"
    report_path.write_text(report_html, encoding="utf-8")
    return str(report_path)


# ---------------------------------------------------------------------------
# Template context builders
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = {
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


def _metric_table_ctx(table: ComparisonTable) -> dict[str, Any] | None:
    rows = table.to_rows()
    if not rows:
        return None

    # Keep a column when *any* row has a value for it, so a failed first case
    # (all-None metrics) does not wipe columns the other cases populate.
    # rows[0].keys() preserves 'case' first followed by MetricSet field order.
    cols = [
        c for c in rows[0].keys()
        if c == "case" or any(row.get(c) is not None for row in rows)
    ]
    return {
        "headers": [_METRIC_DISPLAY.get(c, c) for c in cols],
        "rows": [[_fmt_value(row.get(c)) for c in cols] for row in rows],
    }


def _delta_table_ctx(table: ComparisonTable, baseline: str) -> dict[str, Any] | None:
    deltas = table.delta_from_baseline(baseline)
    if not deltas:
        return None

    metric_keys = [
        k.replace("_delta", "") for k in deltas[0] if k.endswith("_delta")
    ]
    display = {
        "max_von_mises": "Max VM",
        "mean_von_mises": "Mean VM",
        "max_displacement": "Max Disp",
        "fatigue_life": "Fatigue Life",
        "safety_factor": "Safety Factor",
    }
    shown_keys = [mk for mk in metric_keys if mk in display]

    headers = ["Case"]
    for mk in shown_keys:
        headers.extend([f"{display[mk]} delta", "%"])

    rows = []
    for row in deltas:
        cells: list[dict[str, str | None]] = [{"text": row["case"], "cls": None}]
        for mk in shown_keys:
            d = row.get(f"{mk}_delta")
            p = row.get(f"{mk}_pct")
            if d is not None:
                cls = "positive" if d > 0 else "negative"
                cells.append({"text": _fmt_value(d), "cls": cls})
                cells.append({"text": f"{p:+.1f}%", "cls": cls})
            else:
                cells.append({"text": "-", "cls": None})
                cells.append({"text": "-", "cls": None})
        rows.append(cells)

    return {"baseline": baseline, "headers": headers, "rows": rows}


def _generate_comparison_figures(
    study_results: Any,
    table: ComparisonTable,
    baseline: str | None,
) -> dict[str, dict[str, str]]:
    """Generate base64 figures for the comparison report."""
    import matplotlib
    matplotlib.use("Agg")

    from feaweld.visualization.report_figures import figure_to_base64

    figures: dict[str, dict[str, str]] = {}

    def _add(key: str, caption: str, build) -> None:
        try:
            figures[key] = {"b64": figure_to_base64(build()), "caption": caption}
        except Exception:
            pass

    from feaweld.visualization import comparison as vc

    _add(
        "metric_comparison", "Key Metric Comparison",
        lambda: vc.plot_metric_comparison(study_results, "max_von_mises", show=False),
    )

    # One sensitivity figure per detected swept parameter (up to 3)
    try:
        swept = vc.detect_swept_parameters(study_results)
    except Exception:
        swept = []
    for param in swept[:3]:
        _add(
            f"sensitivity_{param.replace('.', '_')}",
            f"Sensitivity: {param}",
            lambda p=param: vc.plot_parameter_sensitivity(
                study_results, p, "max_von_mises", show=False,
            ),
        )

    _add(
        "stress_overlay", "Stress Distribution Overlay",
        lambda: vc.plot_stress_envelope(study_results, show=False),
    )

    # Stress difference vs baseline: only when meshes match exactly
    if baseline and baseline in study_results.results:
        base_result = study_results.results[baseline]
        if (
            base_result.fea_results is not None
            and base_result.fea_results.stress is not None
            and base_result.mesh is not None
        ):
            base_stress = base_result.fea_results.stress
            best_name, best_delta = None, -1.0
            for name, result in study_results.results.items():
                if name == baseline or result.fea_results is None:
                    continue
                stress = result.fea_results.stress
                if (
                    stress is None
                    or result.mesh is None
                    or result.mesh.n_nodes != base_result.mesh.n_nodes
                    or stress.values.shape != base_stress.values.shape
                ):
                    continue
                delta = abs(
                    float(np.max(stress.von_mises))
                    - float(np.max(base_stress.von_mises))
                )
                if delta > best_delta:
                    best_name, best_delta = name, delta
            if best_name is not None:
                comp_stress = study_results.results[best_name].fea_results.stress
                _add(
                    "stress_difference",
                    f"Stress Difference: {best_name} - {baseline}",
                    lambda: vc.plot_stress_difference(
                        base_result.mesh, comp_stress, base_stress,
                        label_a=best_name, label_b=baseline, show=False,
                    ),
                )

    _add(
        "comparison_dashboard", "Comparison Dashboard",
        lambda: vc.comparison_dashboard(study_results, baseline=baseline, show=False),
    )

    return figures
