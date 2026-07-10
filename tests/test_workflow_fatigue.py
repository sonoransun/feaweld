"""Tests for the spectrum-fatigue chain in ``feaweld.pipeline.workflow``.

Covers the ``fatigue:`` config block (validators, YAML round-trip, legacy
fallback), the spectrum assessment path through ``run_analysis`` (R-ratio,
blocks, history, history file), per-method curve handling (FAT225 notch,
Dong power law), residual stress integration, and the PWHT/residual
interplay — all via the FakeBackend, no FE backend or Gmsh required.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from feaweld.core.materials import Material, load_material
from feaweld.core.types import StressField, StressMethod
from feaweld.fatigue.sn_curves import parse_sn_spec
from feaweld.pipeline.workflow import (
    AnalysisCase,
    FatigueConfig,
    GeometryConfig,
    PostProcessConfig,
    ResidualStressConfig,
    SpectrumBlock,
    ThermalConfig,
    _run_fatigue_assessment,
    load_case,
    run_analysis,
    save_case,
)


def _case(**over) -> AnalysisCase:
    kw = dict(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        postprocess=PostProcessConfig(singularity_check=False),
    )
    kw.update(over)
    return AnalysisCase(**kw)


def _run_full(case, mesh, fake_backend, monkeypatch):
    """Run ``run_analysis`` with a fixed mesh and the FakeBackend."""
    monkeypatch.setattr("feaweld.mesh.generator.generate_mesh",
                        lambda joint, cfg: mesh)
    monkeypatch.setattr("feaweld.solver.backend.get_backend",
                        lambda preference="auto": fake_backend)
    return run_analysis(case)


def _fast_pwht() -> ThermalConfig:
    return ThermalConfig(pwht_enabled=True, pwht_time_hours=0.05,
                         pwht_heating_rate=3600.0, pwht_cooling_rate=3600.0)


# ---------------------------------------------------------------------------
# FatigueConfig / SpectrumBlock / ResidualStressConfig validation
# ---------------------------------------------------------------------------

class TestFatigueConfig:
    def test_default_case_has_inactive_fatigue_block(self):
        case = AnalysisCase(**{})
        assert case.fatigue.r_ratio is None
        assert case.fatigue.blocks == []
        assert case.fatigue.history == []
        assert case.fatigue.history_file is None
        assert case.fatigue.mean_stress_correction == "none"
        assert case.fatigue.thickness_correction is True
        assert case.fatigue.environment == "air"
        assert case.fatigue.residual_stress.configured is False

    @pytest.mark.parametrize("kw", [
        dict(r_ratio=0.1, history=[0.0, 1.0, 0.0]),
        dict(r_ratio=0.1, history_file="h.csv"),
        dict(history=[0.0, 1.0], history_file="h.csv"),
        dict(r_ratio=0.1, blocks=[SpectrumBlock(range_factor=1.0)]),
    ])
    def test_two_cyclic_styles_rejected(self, kw):
        with pytest.raises(ValueError, match="one cyclic-loading style"):
            FatigueConfig(**kw)

    def test_block_mixed_key_styles_rejected(self):
        with pytest.raises(ValueError, match="mixes factor keys"):
            SpectrumBlock(range_factor=1.0, mean_stress=50.0)
        with pytest.raises(ValueError, match="mixes factor keys"):
            SpectrumBlock(stress_range=100.0, mean_factor=0.5)

    def test_block_needs_a_range(self):
        with pytest.raises(ValueError, match="range_factor"):
            SpectrumBlock(cycles=10.0)

    def test_residual_multiple_sources_rejected(self):
        with pytest.raises(ValueError, match="at most one"):
            ResidualStressConfig(profile="BS7910_Level1", value=100.0)
        with pytest.raises(ValueError, match="at most one"):
            ResidualStressConfig(value=100.0, as_welded=True)

    def test_yaml_round_trip(self, tmp_path):
        case = AnalysisCase(fatigue=FatigueConfig(
            blocks=[
                SpectrumBlock(range_factor=1.0, cycles=1000.0),
                SpectrumBlock(range_factor=0.5, mean_factor=0.25, cycles=1e4),
            ],
            mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(value=80.0),
        ))
        path = tmp_path / "case.yaml"
        save_case(case, path)
        loaded = load_case(str(path))
        assert loaded.fatigue.mean_stress_correction == "goodman"
        assert len(loaded.fatigue.blocks) == 2
        assert loaded.fatigue.blocks[0].range_factor == 1.0
        assert loaded.fatigue.blocks[1].cycles == 1e4
        assert loaded.fatigue.residual_stress.value == 80.0


# ---------------------------------------------------------------------------
# Legacy fallback: no cyclic definition
# ---------------------------------------------------------------------------

def test_legacy_output_with_default_fatigue_block():
    """A default FatigueConfig reproduces the legacy result exactly."""
    case = AnalysisCase()
    pp = {
        "m1": {"max_stress": 100.0},
        "m2": {"max_stress": 50.0, "fatigue_life": 1e6, "sn_curve_used": "X"},
    }
    out = _run_fatigue_assessment(pp, case)

    curve = parse_sn_spec(case.postprocess.sn_curve)
    assert out == {
        "sn_curve": "IIW_FAT90",
        "m1": {"stress_range": 100.0, "life": curve.life(100.0)},
        "m2": {"stress_range": 50.0, "life": 1e6, "sn_curve": "X"},
    }
    # No spectrum-only keys leak into the legacy path.
    assert "loading" not in out
    assert "corrections" not in out


# ---------------------------------------------------------------------------
# Spectrum assessment through the full pipeline
# ---------------------------------------------------------------------------

class TestSpectrumPipeline:
    def test_r_ratio_constant_amplitude(self, fake_backend, monkeypatch,
                                        grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(r_ratio=0.1, cycles=1e5))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        entry = result.fatigue_results["hotspot_linear"]
        for key in ("stress_range", "life", "sn_curve", "damage",
                    "life_repeats", "equivalent_stress_range", "utilization"):
            assert key in entry

        ref = result.postprocess_results["hotspot_linear"]["max_stress"]
        expected_range = (1.0 - 0.1) * ref
        curve = parse_sn_spec("IIW_FAT90")
        expected_damage = 1e5 / curve.life(expected_range)
        assert entry["equivalent_stress_range"] == pytest.approx(expected_range)
        assert entry["damage"] == pytest.approx(expected_damage)
        assert entry["utilization"] == pytest.approx(expected_damage)
        assert entry["life_repeats"] == pytest.approx(1.0 / expected_damage)
        assert entry["life"] == pytest.approx(1e5 / expected_damage)

        loading = result.fatigue_results["loading"]
        assert loading["type"] == "constant_amplitude"
        assert loading["r_ratio"] == 0.1
        assert loading["cycles"] == 1e5

        corrections = result.fatigue_results["corrections"]
        assert corrections["mean_stress"] == "none"
        assert corrections["k_total"] == pytest.approx(1.0)  # t = 20 < 25 mm
        assert corrections["residual_mean"] == 0.0

        # Governing cycles stashed for the rainflow/damage-map figures.
        rain = result.postprocess_results["rainflow"]
        assert isinstance(rain, list) and len(rain) == 1
        rng, mean, count = rain[0]
        assert rng == pytest.approx(expected_range)
        assert mean == pytest.approx(ref * (1.0 + 0.1) / 2.0)
        assert count == pytest.approx(1e5)

    def test_blocks_spectrum(self, fake_backend, monkeypatch, grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(blocks=[
            SpectrumBlock(range_factor=1.0, cycles=1000.0),
            SpectrumBlock(range_factor=0.5, cycles=1e5),
        ]))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        entry = result.fatigue_results["hotspot_linear"]
        ref = result.postprocess_results["hotspot_linear"]["max_stress"]
        curve = parse_sn_spec("IIW_FAT90")
        expected_damage = (1000.0 / curve.life(ref)
                           + 1e5 / curve.life(0.5 * ref))
        assert entry["damage"] == pytest.approx(expected_damage)
        # No top-level design cycle count => no utilization.
        assert "utilization" not in entry
        assert result.fatigue_results["loading"]["type"] == "blocks"
        assert len(result.postprocess_results["rainflow"]) == 2

    def test_history_inline(self, fake_backend, monkeypatch, grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            history=[0.0, 1.0, -0.2, 0.8, 0.1, 1.0, 0.0],
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        loading = result.fatigue_results["loading"]
        assert loading["type"] == "history"
        assert loading["n_rainflow_cycles"] >= 1
        entry = result.fatigue_results["hotspot_linear"]
        assert entry["damage"] > 0.0
        assert len(result.postprocess_results["rainflow"]) == \
            loading["n_rainflow_cycles"]

    def test_history_file_resolved_against_yaml_dir(self, fake_backend,
                                                    monkeypatch,
                                                    grid_plate_mesh, tmp_path):
        signal = np.concatenate([
            np.linspace(0.0, 1.0, 5), np.linspace(1.0, -0.5, 5),
            np.linspace(-0.5, 0.9, 5),
        ])
        np.savetxt(tmp_path / "history.csv", signal, delimiter=",")
        case_dict = {
            "geometry": {"base_width": 40.0, "base_thickness": 20.0},
            "postprocess": {"singularity_check": False},
            "fatigue": {"history_file": "history.csv"},
        }
        path = tmp_path / "case.yaml"
        with open(path, "w") as f:
            yaml.dump(case_dict, f)

        case = load_case(path)
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors
        assert result.fatigue_results["loading"]["type"] == "history"
        assert result.fatigue_results["hotspot_linear"]["damage"] > 0.0

    def test_notch_assessed_against_fat225(self, fake_backend, monkeypatch,
                                           grid_plate_mesh):
        from feaweld.postprocess.notch_stress import FAT225_CURVE

        case = _case(
            postprocess=PostProcessConfig(
                singularity_check=False,
                stress_methods=[StressMethod.NOTCH_STRESS],
            ),
            fatigue=FatigueConfig(r_ratio=0.0, cycles=1e4),
        )
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        entry = result.fatigue_results["notch_stress"]
        assert entry["sn_curve"] == "IIW FAT225 (effective notch)"
        ref = result.postprocess_results["notch_stress"]["max_stress"]
        # R = 0 => one cycle at the full notch range on FAT225.
        assert entry["damage"] == pytest.approx(1e4 / FAT225_CURVE.life(ref))

    def test_dong_spectrum_uses_power_law(self, fake_backend, monkeypatch,
                                          grid_plate_mesh):
        from feaweld.postprocess.dong import MASTER_SN_H

        case = _case(
            postprocess=PostProcessConfig(
                singularity_check=False,
                stress_methods=[StressMethod.STRUCTURAL_DONG],
            ),
            fatigue=FatigueConfig(r_ratio=0.5, cycles=2000.0),
        )
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        entry = result.fatigue_results["structural_dong"]
        assert entry["corrections_applied"] is False

        life_ref = result.postprocess_results["structural_dong"]["fatigue_life"]
        # One cycle of range factor (1 - R) = 0.5, count 2000:
        # repeats = N_ref / (n * f^m), life = repeats * n.
        expected_repeats = life_ref / (2000.0 * 0.5 ** MASTER_SN_H)
        assert entry["life_repeats"] == pytest.approx(expected_repeats)
        assert entry["life"] == pytest.approx(expected_repeats * 2000.0)
        assert entry["damage"] == pytest.approx(1.0 / expected_repeats)

    def test_knockdowns_enter_strength_factor(self, fake_backend, monkeypatch,
                                              grid_plate_mesh):
        from feaweld.fatigue.knockdown import (
            environment_factor, thickness_correction,
        )

        case = _case(
            geometry=GeometryConfig(base_width=40.0, base_thickness=40.0),
            fatigue=FatigueConfig(r_ratio=0.0, cycles=1e4,
                                  environment="seawater",
                                  thickness_exponent=0.25),
        )
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        corrections = result.fatigue_results["corrections"]
        k_t = thickness_correction(40.0, exponent=0.25)
        k_env = environment_factor("seawater")
        assert corrections["k_thickness"] == pytest.approx(k_t)
        assert corrections["k_environment"] == pytest.approx(k_env)
        assert corrections["k_surface"] == pytest.approx(1.0)
        assert corrections["k_total"] == pytest.approx(k_t * k_env)

        entry = result.fatigue_results["hotspot_linear"]
        ref = result.postprocess_results["hotspot_linear"]["max_stress"]
        curve = parse_sn_spec("IIW_FAT90")
        assert entry["damage"] == pytest.approx(
            1e4 / curve.life(ref / (k_t * k_env))
        )


# ---------------------------------------------------------------------------
# Residual stress integration
# ---------------------------------------------------------------------------

class TestResidualStress:
    def test_value_enters_corrections(self, fake_backend, monkeypatch,
                                      grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(value=120.0),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        meta = result.fea_results.metadata["residual_stress"]
        assert meta["source"] == "value"
        assert meta["surface_value"] == pytest.approx(120.0)
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(120.0)

    def test_profile_surface_value(self, fake_backend, monkeypatch,
                                   grid_plate_mesh):
        from feaweld.data.residual_stress import surface_residual_stress

        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(profile="BS7910_Level2_butt"),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        sigma_y = load_material("A36").sigma_y(20.0)
        expected = surface_residual_stress("BS7910_Level2_butt", sigma_y)
        meta = result.fea_results.metadata["residual_stress"]
        assert meta["source"] == "profile:BS7910_Level2_butt"
        assert meta["surface_value"] == pytest.approx(expected)
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(expected)

    def test_as_welded_is_yield_magnitude(self, fake_backend, monkeypatch,
                                          grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4,
            residual_stress=ResidualStressConfig(as_welded=True),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        sigma_y = load_material("A36").sigma_y(20.0)
        meta = result.fea_results.metadata["residual_stress"]
        assert meta["source"] == "as_welded"
        assert meta["surface_value"] == pytest.approx(sigma_y)
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(sigma_y)

    def test_superimpose_changes_saved_field_keeps_mean(self, fake_backend,
                                                        monkeypatch,
                                                        grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(value=150.0,
                                                 superimpose=True),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        # The saved field carries the residual (superimposed after the
        # assessment) and the pre-superposition peak is recorded.
        meta = result.fea_results.metadata["residual_stress"]
        prior = meta["load_only_max_von_mises"]
        now = float(np.max(result.fea_results.stress.von_mises))
        assert now != pytest.approx(prior)
        # The residual still enters the assessment as a mean stress.
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(150.0)

    def test_superimpose_never_enters_stress_range(self, fake_backend_cls,
                                                   monkeypatch,
                                                   grid_plate_mesh):
        """Regression: a static residual must never inflate the fatigue
        ranges — superimpose true/false give identical assessed ranges and
        lives (the pre-fix pipeline reported 340.9 vs 191.6 MPa here)."""
        def run(superimpose):
            case = _case(fatigue=FatigueConfig(
                r_ratio=0.0, cycles=1e4,
                residual_stress=ResidualStressConfig(
                    value=150.0, superimpose=superimpose),
            ))
            return _run_full(case, grid_plate_mesh, fake_backend_cls(),
                             monkeypatch)

        base = run(False)
        sup = run(True)
        assert base.success, base.errors
        assert sup.success, sup.errors

        e_base = base.fatigue_results["hotspot_linear"]
        e_sup = sup.fatigue_results["hotspot_linear"]
        assert e_sup["equivalent_stress_range"] == \
            pytest.approx(e_base["equivalent_stress_range"])
        assert e_sup["stress_range"] == pytest.approx(e_base["stress_range"])
        assert e_sup["life"] == pytest.approx(e_base["life"])
        # Post-processing saw the load-only field in both runs.
        assert sup.postprocess_results["hotspot_linear"]["max_stress"] == \
            pytest.approx(
                base.postprocess_results["hotspot_linear"]["max_stress"]
            )
        # ... but the saved output field carries the residual.
        assert float(np.max(sup.fea_results.stress.von_mises)) > \
            float(np.max(base.fea_results.stress.von_mises))

    def test_superimpose_profile_anchored_at_plate_surface(self, fake_backend,
                                                           monkeypatch,
                                                           grid_plate_mesh):
        from feaweld.data.residual_stress import surface_residual_stress

        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(
                profile="BS7910_Level2_butt", superimpose=True),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        # The through-thickness field is anchored at the plate surface
        # (y = base_thickness), recorded in the metadata.
        meta = result.fea_results.metadata["residual_stress"]
        assert meta["field_surface_coordinate"] == pytest.approx(20.0)
        sigma_y = load_material("A36").sigma_y(20.0)
        expected = surface_residual_stress("BS7910_Level2_butt", sigma_y)
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(expected)

    def test_warning_goodman_plus_as_welded(self, fake_backend, monkeypatch,
                                            grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(as_welded=True),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert any("double-counts" in w for w in result.warnings)

    def test_warning_inert_residual(self, fake_backend, monkeypatch,
                                    grid_plate_mesh):
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4,
            residual_stress=ResidualStressConfig(value=50.0),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert any("inert" in w for w in result.warnings)

    def test_warning_superimpose_without_correction(self, fake_backend,
                                                    monkeypatch,
                                                    grid_plate_mesh):
        """superimpose + correction 'none': the residual reaches only the
        saved field, never the assessment — warn about that too."""
        case = _case(fatigue=FatigueConfig(
            r_ratio=0.0, cycles=1e4,
            residual_stress=ResidualStressConfig(value=50.0,
                                                 superimpose=True),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert any("does not enter the fatigue assessment" in w
                   for w in result.warnings)

    def test_warning_inert_residual_legacy_path(self, fake_backend,
                                                monkeypatch, grid_plate_mesh):
        """Regression: with no cyclic definition the legacy path used to
        drop the residual (and the correction) silently."""
        case = _case(fatigue=FatigueConfig(
            mean_stress_correction="goodman",
            residual_stress=ResidualStressConfig(value=120.0),
        ))
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors
        assert "loading" not in result.fatigue_results  # legacy path
        assert any("inert" in w for w in result.warnings)

    def test_no_inert_warning_legacy_path_without_residual(self, fake_backend,
                                                           monkeypatch,
                                                           grid_plate_mesh):
        result = _run_full(_case(), grid_plate_mesh, fake_backend,
                           monkeypatch)
        assert result.success, result.errors
        assert not any("inert" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# PWHT / residual interplay
# ---------------------------------------------------------------------------

class TestPwhtResidualInterplay:
    def test_pwht_relaxes_residual_not_load_field(self, fake_backend,
                                                  monkeypatch,
                                                  grid_plate_mesh):
        case = _case(
            fatigue=FatigueConfig(
                r_ratio=0.0, cycles=1e4, mean_stress_correction="goodman",
                residual_stress=ResidualStressConfig(value=200.0),
            ),
            thermal=_fast_pwht(),
        )
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors

        # No legacy "relaxing the load field" warning, no legacy marker.
        assert not any("load-stress field" in w for w in result.warnings)
        meta = result.fea_results.metadata
        assert "as_welded_max_von_mises" not in meta

        factor = meta["pwht_residual_relaxation"]
        assert 0.0 < factor <= 1.0

        # The elastic load field is untouched by PWHT.
        fresh_vm = StressField(
            values=fake_backend.canned_stress(grid_plate_mesh),
        ).von_mises
        np.testing.assert_allclose(result.fea_results.stress.von_mises,
                                   fresh_vm)

        # The fatigue assessment consumes the relaxed residual mean.
        assert result.fatigue_results["corrections"]["residual_mean"] == \
            pytest.approx(200.0 * factor)
        assert meta["residual_stress"]["surface_value"] == \
            pytest.approx(200.0 * factor)

    def test_pwht_without_residual_keeps_legacy_path(self, fake_backend,
                                                     monkeypatch,
                                                     grid_plate_mesh):
        case = _case(
            postprocess=PostProcessConfig(singularity_check=False,
                                          fatigue_assessment=False),
            thermal=_fast_pwht(),
        )
        result = _run_full(case, grid_plate_mesh, fake_backend, monkeypatch)
        assert result.success, result.errors
        assert any("load-stress field" in w for w in result.warnings)
        assert "as_welded_max_von_mises" in result.fea_results.metadata
        assert "pwht_residual_relaxation" not in result.fea_results.metadata


def test_pwht_relaxation_factor_direct():
    from feaweld.core.loads import PWHTSchedule
    from feaweld.solver.creep import pwht_relaxation_factor

    mat = Material(
        name="CreepSteel",
        density=7850.0,
        elastic_modulus={20.0: 200000.0, 700.0: 120000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        creep_A=1e-10, creep_n=3.0, creep_m=0.0,
    )
    schedule = PWHTSchedule(heating_rate=200.0, holding_temperature=620.0,
                            holding_time=2.0, cooling_rate=200.0)

    factor = pwht_relaxation_factor(mat, schedule, 300.0, dt=120.0)
    assert 0.0 < factor < 1.0

    # Zero initial stress or zero creep pre-factor: nothing to relax.
    assert pwht_relaxation_factor(mat, schedule, 0.0) == 1.0
    no_creep = Material(
        name="NoCreep", density=7850.0,
        elastic_modulus={20.0: 200000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        creep_A=0.0,
    )
    assert pwht_relaxation_factor(no_creep, schedule, 300.0) == 1.0
