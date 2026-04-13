"""Error path tests for the analysis workflow.

These tests verify that run_analysis collects errors gracefully rather
than crashing, so callers always get a WorkflowResult object back.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from feaweld.pipeline.workflow import (
    AnalysisCase,
    GeometryConfig,
    LoadConfig,
    MaterialConfig,
    MeshConfig,
    PostProcessConfig,
    SolverConfig,
    WorkflowResult,
    run_analysis,
)
from feaweld.core.types import StressMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(**overrides) -> AnalysisCase:
    """Create a minimal AnalysisCase with optional overrides."""
    defaults = {
        "name": "error_test",
        "material": MaterialConfig(base_metal="A36", weld_metal="E70XX", haz="A36"),
        "geometry": GeometryConfig(joint_type="fillet_t", base_thickness=20.0),
        "mesh": MeshConfig(global_size=2.0, weld_toe_size=0.2),
        "solver": SolverConfig(solver_type="linear_elastic", backend="auto"),
        "load": LoadConfig(axial_force=10000.0),
    }
    defaults.update(overrides)
    return AnalysisCase(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMaterialErrors:

    def test_material_not_found(self):
        """A non-existent material name should produce a collected error, not a crash."""
        case = _make_case(
            material=MaterialConfig(
                base_metal="NONEXISTENT_ALLOY_XYZ_999",
                weld_metal="E70XX",
                haz="A36",
            )
        )
        result = run_analysis(case)

        assert isinstance(result, WorkflowResult)
        assert not result.success
        assert len(result.errors) > 0
        # The error message should mention the missing material or workflow error
        error_text = " ".join(result.errors).lower()
        assert "error" in error_text or "not found" in error_text or "workflow" in error_text


class TestGeometryErrors:

    def test_workflow_catches_geometry_error(self):
        """If _build_geometry raises, the error is collected in WorkflowResult."""
        case = _make_case()

        with patch(
            "feaweld.pipeline.workflow._build_geometry",
            side_effect=ValueError("Geometry construction failed"),
        ):
            result = run_analysis(case)

        assert isinstance(result, WorkflowResult)
        assert not result.success
        assert any("Geometry construction failed" in e or "Workflow error" in e for e in result.errors)


class TestSolverErrors:

    def test_workflow_catches_solver_error(self):
        """If get_backend raises, the error is collected in WorkflowResult."""
        case = _make_case()

        with patch(
            "feaweld.solver.backend.get_backend",
            side_effect=ImportError("No solver backend available"),
        ):
            result = run_analysis(case)

        assert isinstance(result, WorkflowResult)
        assert not result.success
        assert any("No solver backend" in e or "Workflow error" in e for e in result.errors)


class TestPostprocessErrors:

    def test_workflow_catches_postprocess_error(self):
        """If _run_postprocess raises for a method, it is recorded per-method."""
        case = _make_case(
            postprocess=PostProcessConfig(
                stress_methods=[StressMethod.HOTSPOT_LINEAR, StressMethod.STRUCTURAL_DONG],
            )
        )

        # We need to mock everything up to and including the solver
        # so that the pipeline reaches the post-processing stage.
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 4
        mock_mesh.n_elements = 2
        mock_mesh.node_sets = {"weld_toe": np.array([1, 2])}

        mock_stress = MagicMock()
        mock_stress.values = np.zeros((4, 6))

        mock_fea = MagicMock()
        mock_fea.stress = mock_stress

        mock_backend = MagicMock()
        mock_backend.solve_static.return_value = mock_fea

        with patch("feaweld.pipeline.workflow._run_postprocess",
                   side_effect=RuntimeError("Post-processing boom")), \
             patch("feaweld.pipeline.workflow._build_geometry"), \
             patch("feaweld.mesh.generator.generate_mesh", return_value=mock_mesh), \
             patch("feaweld.solver.backend.get_backend", return_value=mock_backend):
            result = run_analysis(case)

        assert isinstance(result, WorkflowResult)
        # Should have two post-process errors (one per method)
        pp_errors = [e for e in result.errors if "Post-processing" in e]
        assert len(pp_errors) == 2


class TestPartialResults:

    def test_workflow_partial_results(self):
        """WorkflowResult should have mesh even when solver fails."""
        case = _make_case()

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 10
        mock_mesh.n_elements = 5

        with patch("feaweld.pipeline.workflow._build_geometry"), \
             patch("feaweld.mesh.generator.generate_mesh", return_value=mock_mesh), \
             patch(
                 "feaweld.solver.backend.get_backend",
                 side_effect=RuntimeError("Solver exploded"),
             ):
            result = run_analysis(case)

        assert isinstance(result, WorkflowResult)
        # Mesh was assigned before the solver error
        assert result.mesh is mock_mesh
        # Solver error is collected
        assert not result.success
        # FEA results should be None since solver failed
        assert result.fea_results is None


class TestSuccessProperty:

    def test_success_true_when_no_errors(self):
        result = WorkflowResult(case=_make_case())
        assert result.success is True
        assert result.errors == []

    def test_success_false_with_errors(self):
        result = WorkflowResult(case=_make_case())
        result.errors.append("Something went wrong")
        assert result.success is False

    def test_success_false_with_multiple_errors(self):
        result = WorkflowResult(case=_make_case())
        result.errors.extend(["Error 1", "Error 2", "Error 3"])
        assert result.success is False
        assert len(result.errors) == 3
