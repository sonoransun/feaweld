"""Tests for the Bayesian deep-ensemble fatigue surrogate (Track B1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("jax")

from feaweld.ml.bayesian_surrogate import (
    BayesianFatigueSurrogate,
    EnsembleConfig,
)


pytestmark = pytest.mark.requires_flax


def _sinusoid(rng: np.random.Generator, x: np.ndarray, noise: float = 0.05) -> np.ndarray:
    return np.sin(x).ravel() + rng.normal(0.0, noise, size=x.shape[0])


@pytest.fixture
def trained_surrogate() -> tuple[BayesianFatigueSurrogate, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 2.0 * np.pi, 200).reshape(-1, 1)
    y = _sinusoid(rng, x, noise=0.05)

    cfg = EnsembleConfig(
        n_members=5,
        hidden_sizes=(32, 32),
        n_epochs=150,
        learning_rate=5e-3,
        weight_decay=1e-4,
        seed=0,
    )
    surrogate = BayesianFatigueSurrogate(config=cfg)
    surrogate.fit(x, y)
    return surrogate, x, y


def test_fit_predict_shapes(trained_surrogate):
    surrogate, _, _ = trained_surrogate

    x_test = np.linspace(0.0, 2.0 * np.pi, 25).reshape(-1, 1)
    mean, epistemic_std, aleatoric_std = surrogate.predict(x_test)

    assert mean.shape == (25,)
    assert epistemic_std.shape == (25,)
    assert aleatoric_std.shape == (25,)
    assert np.all(np.isfinite(mean))
    assert np.all(epistemic_std >= 0.0)
    assert np.all(aleatoric_std >= 0.0)

    mean2, total_std = surrogate.predict_total_std(x_test)
    assert mean2.shape == (25,)
    assert total_std.shape == (25,)
    assert np.allclose(mean, mean2)
    assert np.all(total_std >= epistemic_std - 1e-9)
    assert np.all(total_std >= aleatoric_std - 1e-9)


def test_epistemic_uncertainty_larger_out_of_distribution(trained_surrogate):
    surrogate, _, _ = trained_surrogate

    x_in = np.linspace(0.0, 2.0 * np.pi, 40).reshape(-1, 1)
    x_ood = np.linspace(4.0 * np.pi, 5.0 * np.pi, 40).reshape(-1, 1)

    _, eps_in, _ = surrogate.predict(x_in)
    _, eps_ood, _ = surrogate.predict(x_ood)

    assert float(eps_ood.mean()) > float(eps_in.mean())


def test_predict_calibration_ks():
    rng = np.random.default_rng(1)
    x_train = np.linspace(0.0, 2.0 * np.pi, 200).reshape(-1, 1)
    y_train = _sinusoid(rng, x_train, noise=0.05)

    x_test = rng.uniform(0.0, 2.0 * np.pi, size=(100, 1))
    y_test = _sinusoid(rng, x_test, noise=0.05)

    cfg = EnsembleConfig(
        n_members=5,
        hidden_sizes=(32, 32),
        n_epochs=150,
        learning_rate=5e-3,
        weight_decay=1e-4,
        seed=1,
    )
    surrogate = BayesianFatigueSurrogate(config=cfg)
    surrogate.fit(x_train, y_train)

    mean, total_std = surrogate.predict_total_std(x_test)
    residuals = (y_test - mean) / np.maximum(total_std, 1e-6)
    r_std = float(np.std(residuals))

    assert 0.7 <= r_std <= 1.3, f"residual std out of bounds: {r_std}"
