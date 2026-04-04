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
from feaweld.pipeline.report import generate_report, REPORT_TEMPLATE


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
    """Test that report template contains required placeholders."""
    assert "{{ title }}" in REPORT_TEMPLATE
    assert "{{ content }}" in REPORT_TEMPLATE
    assert "{{ version }}" in REPORT_TEMPLATE


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
