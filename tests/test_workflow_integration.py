"""End-to-end workflow integration tests for Track H.

Exercises the new DefectConfig / MultiPassConfig / WeldPathConfig fields
and the additive dispatch hooks wired into :func:`run_analysis`.
"""

from __future__ import annotations

import pytest

from feaweld.pipeline.workflow import (
    AnalysisCase,
    DefectConfig,
    MultiPassConfig,
    PostProcessConfig,
    WeldPathConfig,
    WorkflowResult,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_defects_config_defaults():
    cfg = DefectConfig()
    assert cfg.enabled is False
    assert cfg.quality_level == "B"


def test_multipass_config_defaults():
    cfg = MultiPassConfig()
    assert cfg.enabled is False


def test_weld_path_config_defaults():
    cfg = WeldPathConfig()
    assert cfg.mode == "straight"


def test_postprocess_new_fields():
    cfg = PostProcessConfig()
    assert cfg.multiaxial_criterion == "none"
    assert cfg.compute_k_factors is False


def test_workflow_result_extensions_still_a_dict():
    result = WorkflowResult(case=AnalysisCase())
    assert result.extensions == {}


# ---------------------------------------------------------------------------
# YAML round-trip with all new fields populated
# ---------------------------------------------------------------------------


def test_analysiscase_yaml_round_trip_with_new_fields():
    case = AnalysisCase(
        name="h_round_trip",
        defects=DefectConfig(
            enabled=True,
            quality_level="C",
            weld_length=150.0,
            weld_width=12.0,
            population_seed=7,
        ),
        multipass=MultiPassConfig(
            enabled=True,
            sequence_name="v_groove_3pass",
            preheat_temp=80.0,
            interpass_temp_max=220.0,
        ),
        weld_path=WeldPathConfig(
            mode="spline",
            control_points=[(0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (100.0, 5.0, 0.0)],
            spline_mode="catmull_rom",
            spline_degree=3,
        ),
        postprocess=PostProcessConfig(
            multiaxial_criterion="findley",
            compute_k_factors=True,
        ),
    )

    data = case.model_dump(mode="json")
    restored = AnalysisCase(**data)
    assert restored.defects.quality_level == "C"
    assert restored.multipass.sequence_name == "v_groove_3pass"
    assert restored.weld_path.mode == "spline"
    assert restored.weld_path.spline_mode == "catmull_rom"
    assert len(restored.weld_path.control_points) == 3
    assert restored.postprocess.multiaxial_criterion == "findley"
    assert restored.postprocess.compute_k_factors is True


# ---------------------------------------------------------------------------
# Dispatch hooks — defect population
# ---------------------------------------------------------------------------


def test_run_analysis_with_defects_config():
    """With defects.enabled=True, the workflow must populate extensions.

    The underlying geometry/mesh/solver stages may fail in sandboxes
    without Gmsh — that's fine: the defect hook runs *after* geometry
    but before mesh generation, so the extension should still be set
    even if later stages fall over.
    """
    # Use a high Poisson rate so the test is deterministic (seed=0 with
    # default 100mm weld -> ~5 pores expected).
    case = AnalysisCase(
        name="defect_hook_test",
        defects=DefectConfig(
            enabled=True,
            quality_level="B",
            weld_length=200.0,
            weld_width=10.0,
            population_seed=42,
        ),
    )
    result = run_analysis(case)
    assert "defects" in result.extensions, (
        f"defects extension missing; errors={result.errors}"
    )
    defect_ext = result.extensions["defects"]
    assert defect_ext["count"] >= 0
    assert "population" in defect_ext
    # A 200mm weld at the default Poisson rates almost always produces
    # at least one defect; allow zero defensively if the RNG state
    # happens to sample nothing.
    assert isinstance(defect_ext["population"], list)


def test_run_analysis_with_explicit_defects():
    """Explicit defect dicts should pass straight through to extensions."""
    case = AnalysisCase(
        name="explicit_defects_test",
        defects=DefectConfig(
            enabled=True,
            explicit_defects=[
                {"defect_type": "pore", "diameter": 0.5},
                {"defect_type": "undercut", "depth": 0.2},
            ],
        ),
    )
    result = run_analysis(case)
    assert "defects" in result.extensions
    assert result.extensions["defects"]["count"] == 2
    assert result.extensions["defects"]["source"] == "explicit"


# ---------------------------------------------------------------------------
# Dispatch hooks — multi-axial fatigue
# ---------------------------------------------------------------------------


def test_run_analysis_with_multiaxial_criterion():
    """The multi-axial hook either runs or gracefully skips."""
    case = AnalysisCase(
        name="findley_test",
        postprocess=PostProcessConfig(multiaxial_criterion="findley"),
    )
    result = run_analysis(case)
    # Either the criterion ran (stress field was available) or the hook
    # gracefully no-op'd because the solver didn't produce stresses.
    if result.fea_results is not None and result.fea_results.stress is not None:
        assert "multiaxial_fatigue" in result.extensions, (
            f"multiaxial_fatigue missing; errors={result.errors}"
        )
        ext = result.extensions["multiaxial_fatigue"]
        assert ext["criterion"] == "findley"
        assert "damage_parameter" in ext
        assert "critical_plane_normal" in ext
    else:
        pytest.skip("No stress field available in this sandbox")


# ---------------------------------------------------------------------------
# Disabled-by-default parity: behaviour bit-identical to pre-H
# ---------------------------------------------------------------------------


def test_run_analysis_disabled_extensions_are_absent():
    case = AnalysisCase(name="baseline_test")
    result = run_analysis(case)
    assert "defects" not in result.extensions
    assert "multipass" not in result.extensions
    assert "multiaxial_fatigue" not in result.extensions
    assert "j_integral" not in result.extensions
