"""Tests for multi-dimensional EnKF, SIF interpolation, and observation operators."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.digital_twin.assimilation import (
    CrackEnKF,
    MultiStateCrackEnKF,
    MultiStateParisModel,
    ParisLawModel,
    paris_law_sif,
)
from feaweld.digital_twin.sif_interpolator import (
    SIFInterpolator,
    SIFTable,
    combined_sif,
    residual_stress_sif,
)
from feaweld.digital_twin.observation_operators import (
    MultiObservation,
    ultrasonic_crack_operator,
    strain_gauge_operator,
    acpd_operator,
)


SEED = 42


def _make_multistate_model(stress_range: float = 50.0) -> MultiStateParisModel:
    return MultiStateParisModel(
        dK_fn=paris_law_sif(stress_range=stress_range, geometry_factor=1.12),
    )


# ---------------------------------------------------------------------------
# MultiStateParisModel
# ---------------------------------------------------------------------------


def test_multistate_forward_model_matches_scalar():
    """Multi-state step with fixed (C, m) should match scalar ParisLawModel."""
    C = 1.0e-11
    m = 3.0
    dK_fn = paris_law_sif(stress_range=80.0, geometry_factor=1.12)

    scalar_model = ParisLawModel(C=C, m=m, dK_fn=dK_fn)
    multi_model = MultiStateParisModel(dK_fn=dK_fn)

    a0 = 0.5
    dn = 100.0

    a_scalar = scalar_model.step(a0, dn)
    state = np.array([a0, np.log(C), m])
    state_new = multi_model.step(state, dn)

    assert abs(state_new[0] - a_scalar) < 1e-12
    assert state_new[1] == np.log(C)  # logC unchanged
    assert state_new[2] == m           # m unchanged


# ---------------------------------------------------------------------------
# MultiStateCrackEnKF
# ---------------------------------------------------------------------------


def test_multistate_enkf_std_shrinks():
    model = _make_multistate_model(stress_range=60.0)
    enkf = MultiStateCrackEnKF(
        model, n_ensemble=80,
        initial_means=np.array([0.5, np.log(1e-11), 3.0]),
        initial_stds=np.array([0.15, 1.0, 0.3]),
        process_noise_stds=np.array([1e-5, 1e-5, 1e-5]),
        seed=SEED,
    )
    initial_std = enkf.std.copy()

    true_a = 0.5
    rng = np.random.default_rng(SEED + 1)
    scalar_model = ParisLawModel(C=1e-11, m=3.0, dK_fn=model.dK_fn)

    for _ in range(20):
        enkf.predict(dn=10.0)
        true_a = scalar_model.step(true_a, dn=10.0)
        obs = true_a + rng.normal(0.0, 0.005)
        ops = [ultrasonic_crack_operator()]
        enkf.update(
            observations=np.array([obs]),
            obs_stds=np.array([0.005]),
            operators=ops,
        )

    # Crack length std should shrink
    assert enkf.std[0] < initial_std[0]


def test_multistate_enkf_tracks_joint_state():
    """EnKF should converge toward true (C, m) values."""
    true_C = 1.5e-11
    true_m = 3.2
    dK_fn = paris_law_sif(stress_range=80.0)
    model = MultiStateParisModel(dK_fn=dK_fn)
    scalar = ParisLawModel(C=true_C, m=true_m, dK_fn=dK_fn)

    enkf = MultiStateCrackEnKF(
        model, n_ensemble=200,
        initial_means=np.array([0.3, np.log(1e-11), 3.0]),
        initial_stds=np.array([0.05, 0.5, 0.2]),
        process_noise_stds=np.array([1e-5, 1e-4, 1e-4]),
        seed=SEED,
    )

    rng = np.random.default_rng(SEED + 2)
    true_a = 0.3
    for _ in range(100):
        enkf.predict(dn=5.0)
        true_a = scalar.step(true_a, dn=5.0)
        obs = true_a + rng.normal(0.0, 0.01)
        enkf.update(
            observations=np.array([obs]),
            obs_stds=np.array([0.01]),
            operators=[ultrasonic_crack_operator()],
        )

    # Crack length should be close to truth
    assert abs(enkf.mean[0] - true_a) < 0.1 * true_a


def test_multistate_remaining_life_distribution():
    model = _make_multistate_model(stress_range=80.0)
    enkf = MultiStateCrackEnKF(
        model, n_ensemble=30,
        initial_means=np.array([1.0, np.log(1e-11), 3.0]),
        initial_stds=np.array([0.1, 0.2, 0.1]),
        seed=SEED,
    )
    lives = enkf.remaining_life_distribution(a_critical=25.0, dn_step=1000.0, max_steps=10000)
    assert lives.shape == (30,)
    finite = lives[np.isfinite(lives)]
    assert len(finite) > 0


# ---------------------------------------------------------------------------
# SIF Interpolator
# ---------------------------------------------------------------------------


def test_sif_interpolator_from_handbook():
    interp = SIFInterpolator.from_handbook(stress_range=100.0, geometry_factor=1.12)
    dk_5 = interp(5.0)
    expected = 1.12 * 100.0 * np.sqrt(np.pi * 5.0)
    assert abs(dk_5 - expected) < 0.5


def test_sif_interpolator_accuracy():
    a_vals = np.linspace(0.1, 30.0, 100)
    dk_vals = 1.12 * 80.0 * np.sqrt(np.pi * a_vals)
    table = SIFTable(crack_lengths=a_vals, sif_values=dk_vals, source="test")
    interp = SIFInterpolator(table)
    # Test interior interpolation
    for a_test in [0.5, 5.0, 15.0, 29.0]:
        expected = 1.12 * 80.0 * np.sqrt(np.pi * a_test)
        assert abs(interp(a_test) - expected) / expected < 0.01


def test_combined_sif():
    def applied(a): return 100.0 * np.sqrt(a)
    def residual(a): return 50.0
    combo = combined_sif(applied, residual)
    assert combo(4.0) == applied(4.0) + 50.0
    # Negative combined returns 0
    combo_neg = combined_sif(lambda a: 10.0, lambda a: -20.0)
    assert combo_neg(1.0) == 0.0


# ---------------------------------------------------------------------------
# Observation operators
# ---------------------------------------------------------------------------


def test_ultrasonic_operator():
    op = ultrasonic_crack_operator()
    state = np.array([5.0, -25.0, 3.0])
    assert op(state) == 5.0


def test_strain_gauge_operator():
    sif_fn = paris_law_sif(100.0, 1.12)
    op = strain_gauge_operator(sif_fn, gauge_distance_mm=2.0, E=210_000.0)
    state = np.array([1.0, -25.0, 3.0])
    strain = op(state)
    assert strain > 0.0
    assert np.isfinite(strain)


def test_acpd_operator():
    op = acpd_operator(calibration_slope=2.0, calibration_intercept=0.5)
    state = np.array([3.0, -25.0, 3.0])
    assert op(state) == 2.0 * 3.0 + 0.5


def test_multi_sensor_fusion_reduces_uncertainty():
    """Using 3 sensors should give lower uncertainty than 1."""
    dK_fn = paris_law_sif(80.0)
    model = MultiStateParisModel(dK_fn=dK_fn)

    # Single sensor
    enkf1 = MultiStateCrackEnKF(
        model, n_ensemble=100,
        initial_means=np.array([0.5, np.log(1e-11), 3.0]),
        initial_stds=np.array([0.15, 0.5, 0.2]),
        seed=SEED,
    )
    # Three sensors
    enkf3 = MultiStateCrackEnKF(
        model, n_ensemble=100,
        initial_means=np.array([0.5, np.log(1e-11), 3.0]),
        initial_stds=np.array([0.15, 0.5, 0.2]),
        seed=SEED,
    )

    true_a = 0.5
    rng = np.random.default_rng(SEED)
    scalar = ParisLawModel(C=1e-11, m=3.0, dK_fn=dK_fn)

    sif_fn = paris_law_sif(80.0)
    ut_op = ultrasonic_crack_operator()
    sg_op = strain_gauge_operator(sif_fn, gauge_distance_mm=2.0)
    acpd_op_fn = acpd_operator(calibration_slope=1.0)

    for _ in range(15):
        enkf1.predict(dn=10.0)
        enkf3.predict(dn=10.0)
        true_a = scalar.step(true_a, dn=10.0)

        obs_ut = true_a + rng.normal(0.0, 0.01)
        obs_sg = sg_op(np.array([true_a, 0, 0])) + rng.normal(0.0, 1e-5)
        obs_acpd = acpd_op_fn(np.array([true_a, 0, 0])) + rng.normal(0.0, 0.01)

        # Single sensor update
        enkf1.update(np.array([obs_ut]), np.array([0.01]), [ut_op])
        # Three sensor update
        enkf3.update(
            np.array([obs_ut, obs_sg, obs_acpd]),
            np.array([0.01, 1e-5, 0.01]),
            [ut_op, sg_op, acpd_op_fn],
        )

    # 3-sensor should have lower crack-length std
    assert enkf3.std[0] < enkf1.std[0]


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_scalar_enkf_unchanged():
    """Original CrackEnKF should still work identically."""
    model = ParisLawModel(C=1e-11, m=3.0, dK_fn=paris_law_sif(50.0))
    enkf = CrackEnKF(model, n_ensemble=30, seed=123)
    enkf.predict(dn=10.0)
    enkf.update(observation=0.11, obs_std=0.01)
    assert np.isfinite(enkf.mean)
    assert np.isfinite(enkf.std)
    assert len(enkf.history) == 3
