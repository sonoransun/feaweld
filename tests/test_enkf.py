"""Tests for the Ensemble Kalman Filter crack-length estimator."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.digital_twin.assimilation import (
    CrackEnKF,
    ParisLawModel,
    paris_law_sif,
)


SEED = 1234


def _make_model(stress_range: float = 50.0) -> ParisLawModel:
    # Paris constants tuned for SI-mm so 1000 blocks give sub-mm growth.
    return ParisLawModel(
        C=1.0e-11,
        m=3.0,
        dK_fn=paris_law_sif(stress_range=stress_range, geometry_factor=1.12),
    )


def test_paris_law_euler_step_monotone():
    model = _make_model(stress_range=80.0)
    a = 0.2
    history = [a]
    for _ in range(500):
        a = model.step(a, dn=10.0)
        history.append(a)
    diffs = np.diff(history)
    assert np.all(diffs >= 0.0)
    assert history[-1] > history[0]


def test_enkf_std_shrinks_with_observations():
    model = _make_model(stress_range=60.0)
    enkf = CrackEnKF(
        model,
        n_ensemble=80,
        initial_mean=0.5,
        initial_std=0.15,
        process_noise_std=1e-5,
        seed=SEED,
    )
    initial_std = enkf.std
    rng = np.random.default_rng(SEED + 1)
    true_a = 0.5
    for _ in range(20):
        enkf.predict(dn=10.0)
        true_a = model.step(true_a, dn=10.0)
        obs = true_a + rng.normal(0.0, 0.005)
        enkf.update(observation=obs, obs_std=0.005)
    assert enkf.std < initial_std


def test_enkf_mean_tracks_synthetic_twin():
    model = _make_model(stress_range=80.0)
    rng = np.random.default_rng(SEED)

    # Synthetic ground-truth twin
    true_a = 0.3
    obs_std = 0.01
    truth_history: list[float] = []
    observations: list[tuple[int, float]] = []
    for block in range(1000):
        true_a = model.step(true_a, dn=1.0)
        truth_history.append(true_a)
        if (block + 1) % 10 == 0:
            observations.append(
                (block, true_a + float(rng.normal(0.0, obs_std)))
            )

    # Filter initialized with a perturbed guess
    enkf = CrackEnKF(
        model,
        n_ensemble=100,
        initial_mean=0.45,
        initial_std=0.08,
        process_noise_std=1e-5,
        seed=SEED + 7,
    )

    obs_iter = iter(observations)
    next_obs = next(obs_iter, None)
    for block in range(1000):
        enkf.predict(dn=1.0)
        if next_obs is not None and block == next_obs[0]:
            enkf.update(observation=next_obs[1], obs_std=obs_std)
            next_obs = next(obs_iter, None)

    final_truth = truth_history[-1]
    rmse = abs(enkf.mean - final_truth)
    assert rmse < 0.10 * final_truth, (
        f"rmse={rmse:.4f}, truth={final_truth:.4f}, mean={enkf.mean:.4f}"
    )


def test_enkf_history_records_every_step():
    model = _make_model()
    enkf = CrackEnKF(model, n_ensemble=20, seed=SEED)
    assert len(enkf.history) == 1
    enkf.predict(dn=5.0)
    enkf.update(observation=0.11, obs_std=0.01)
    assert len(enkf.history) == 3
    for mean, std in enkf.history:
        assert np.isfinite(mean)
        assert np.isfinite(std)
