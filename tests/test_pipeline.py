"""Tests for pipeline workflow and report generation."""

import numpy as np
import pytest
from pathlib import Path

from feaweld.pipeline.workflow import (
    AnalysisCase,
    MaterialConfig,
    GeometryConfig,
    MeshConfig,
    SolverConfig,
    LoadConfig,
    PostProcessConfig,
    load_case,
    save_case,
)
from feaweld.pipeline.report import generate_report


def test_analysis_case_default():
    """Test AnalysisCase creates with valid defaults."""
    case = AnalysisCase()
    assert case.name == "default"
    assert case.material.base_metal == "A36"
    assert case.geometry.joint_type.value == "fillet_t"
    assert case.solver.solver_type.value == "linear_elastic"


def test_analysis_case_custom():
    """Test AnalysisCase with custom config."""
    case = AnalysisCase(
        name="test_case",
        material=MaterialConfig(base_metal="304SS"),
        geometry=GeometryConfig(base_thickness=25.0, weld_leg_size=10.0),
        load=LoadConfig(axial_force=50000.0),
    )
    assert case.material.base_metal == "304SS"
    assert case.geometry.base_thickness == 25.0
    assert case.load.axial_force == 50000.0


def test_analysis_case_uppercase_enums():
    """YAML may spell enums in any case, or by member name (e.g. SED)."""
    from feaweld.core.types import JointType, SolverType, StressMethod

    case = AnalysisCase(
        geometry={"joint_type": "FILLET_T"},
        solver={"solver_type": "Linear_Elastic"},
        postprocess={"stress_methods": ["HOTSPOT_LINEAR", "SED"]},
    )
    assert case.geometry.joint_type is JointType.FILLET_T
    assert case.solver.solver_type is SolverType.LINEAR_ELASTIC
    assert case.postprocess.stress_methods[1] is StressMethod.SED

    with pytest.raises(ValueError):
        JointType("not_a_joint")


def test_save_and_load_case(tmp_path):
    """Test round-trip save/load of analysis case."""
    case = AnalysisCase(name="round_trip_test")
    yaml_path = tmp_path / "test_case.yaml"

    save_case(case, yaml_path)
    assert yaml_path.exists()

    loaded = load_case(yaml_path)
    assert loaded.name == "round_trip_test"
    assert loaded.material.base_metal == case.material.base_metal


def test_report_template_valid():
    """Test that the Jinja2 report templates load and render."""
    from feaweld.pipeline.report import _get_env

    env = _get_env()
    for name in ("base.html.j2", "report.html.j2", "comparison.html.j2"):
        assert env.get_template(name) is not None

    html = env.get_template("report.html.j2").render(
        title="smoke", version="0.0", timestamp="now",
        case=None, mesh=None, fea_rows=[], postprocess_rows=[],
        fatigue_rows=[], probabilistic_rows=[], figures=[],
        interactive_figures=[], plotly_cdn=False, warnings=[], errors=[],
    )
    assert "feaweld Analysis Report" in html
    assert "0.0" in html


def test_generate_report_creates_file(tmp_path, uniform_stress_results):
    """Test report generation creates an HTML file."""
    from feaweld.pipeline.workflow import WorkflowResult

    case = AnalysisCase(name="test_report", output_dir=str(tmp_path))
    result = WorkflowResult(
        case=case,
        mesh=uniform_stress_results.mesh,
        fea_results=uniform_stress_results,
    )

    report_path = generate_report(result, tmp_path)
    assert Path(report_path).exists()
    assert report_path.endswith(".html")

    content = Path(report_path).read_text()
    assert "feaweld Analysis Report" in content
    assert "test_report" in content
