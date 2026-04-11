"""Bayesian deep-ensemble fatigue surrogate (Track B1).

A small ensemble of heteroscedastic MLPs trained with different random seeds
that provides calibrated uncertainty decomposed into epistemic (ensemble
disagreement) and aleatoric (predicted noise) components.

Flax / Optax / JAX are heavy optional dependencies; this module imports
cleanly without them but any attempt to instantiate the surrogate will raise
``ImportError`` pointing the user at ``pip install 'feaweld[flax]'``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from flax import serialization as flax_serialization

    _HAS_FLAX = True
except ImportError:  # pragma: no cover - exercised only when extras missing
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optax = None  # type: ignore[assignment]
    flax_serialization = None  # type: ignore[assignment]
    _HAS_FLAX = False


_MISSING_FLAX_MSG = (
    "BayesianFatigueSurrogate requires Flax / Optax / JAX. "
    "Install with: pip install 'feaweld[flax]'"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Hyper-parameters for the deep-ensemble surrogate."""

    n_members: int = 5
    hidden_sizes: tuple[int, ...] = (64, 64)
    n_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


# ---------------------------------------------------------------------------
# Flax network
# ---------------------------------------------------------------------------


if _HAS_FLAX:

    class HeteroscedasticMLP(nn.Module):
        """MLP with two heads: predicted mean and predicted log-variance."""

        hidden_sizes: tuple[int, ...]

        @nn.compact
        def __call__(self, x):  # type: ignore[override]
            h = x
            for width in self.hidden_sizes:
                h = nn.Dense(width)(h)
                h = nn.swish(h)
            mean = nn.Dense(1)(h).squeeze(-1)
            log_var = nn.Dense(1)(h).squeeze(-1)
            # Clamp log_var to a numerically safe range
            log_var = jnp.clip(log_var, -10.0, 10.0)
            return mean, log_var

else:  # pragma: no cover - exercised only when extras missing

    class HeteroscedasticMLP:  # type: ignore[no-redef]
        """Placeholder that raises when Flax is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_MISSING_FLAX_MSG)


# ---------------------------------------------------------------------------
# Ensemble surrogate
# ---------------------------------------------------------------------------


class BayesianFatigueSurrogate:
    """Deep ensemble of heteroscedastic MLPs for fatigue life prediction.

    The surrogate is trained by minimizing Gaussian negative log-likelihood on
    each member independently with a distinct PRNG key.  At prediction time
    the ensemble mean is returned alongside an epistemic (disagreement)
    standard deviation and an aleatoric (predicted noise) standard deviation.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        if not _HAS_FLAX:
            raise ImportError(_MISSING_FLAX_MSG)

        self.config = config or EnsembleConfig()
        self._params: list[Any] = []
        self._model: HeteroscedasticMLP | None = None
        self._x_mean: NDArray[np.float64] | None = None
        self._x_std: NDArray[np.float64] | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None
        self._n_features: int | None = None
        self._is_trained: bool = False

    # --------------------------------------------------------------- fit

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Train the ensemble on ``(X, y)`` by Gaussian NLL."""

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")

        self._n_features = X.shape[1]
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = float(y.mean())
        self._y_std = float(y.std() + 1e-8)

        Xn = (X - self._x_mean) / self._x_std
        yn = (y - self._y_mean) / self._y_std

        Xj = jnp.asarray(Xn, dtype=jnp.float32)
        yj = jnp.asarray(yn, dtype=jnp.float32)

        cfg = self.config
        self._model = HeteroscedasticMLP(hidden_sizes=tuple(cfg.hidden_sizes))

        def nll_loss(params, xb, yb):
            mean, log_var = self._model.apply(params, xb)
            inv_var = jnp.exp(-log_var)
            return jnp.mean(0.5 * (log_var + (yb - mean) ** 2 * inv_var))

        grad_fn = jax.jit(jax.value_and_grad(nll_loss))

        root_key = jax.random.PRNGKey(cfg.seed)
        member_keys = jax.random.split(root_key, cfg.n_members)

        self._params = []
        for member_key in member_keys:
            init_key, _ = jax.random.split(member_key)
            params = self._model.init(init_key, Xj[:1])

            optimizer = optax.adamw(
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            opt_state = optimizer.init(params)

            @jax.jit
            def step(params, opt_state, xb, yb):
                loss, grads = grad_fn(params, xb, yb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            for _ in range(cfg.n_epochs):
                params, opt_state, _ = step(params, opt_state, Xj, yj)

            self._params.append(params)

        self._is_trained = True

    # ----------------------------------------------------------- predict

    def predict(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, epistemic_std, aleatoric_std)`` on original scale."""

        if not self._is_trained or self._model is None:
            raise RuntimeError("Surrogate is not trained. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        Xn = (X - self._x_mean) / self._x_std
        Xj = jnp.asarray(Xn, dtype=jnp.float32)

        member_means: list[NDArray[np.float64]] = []
        member_vars: list[NDArray[np.float64]] = []
        for params in self._params:
            mean_n, log_var_n = self._model.apply(params, Xj)
            member_means.append(np.asarray(mean_n, dtype=np.float64))
            member_vars.append(np.asarray(jnp.exp(log_var_n), dtype=np.float64))

        means = np.stack(member_means, axis=0)  # (n_members, n_samples)
        vars_ = np.stack(member_vars, axis=0)

        y_std = self._y_std or 1.0
        y_mean = self._y_mean or 0.0

        ensemble_mean_n = means.mean(axis=0)
        epistemic_var_n = means.var(axis=0, ddof=0)
        aleatoric_var_n = vars_.mean(axis=0)

        mean = ensemble_mean_n * y_std + y_mean
        epistemic_std = np.sqrt(epistemic_var_n) * y_std
        aleatoric_std = np.sqrt(aleatoric_var_n) * y_std

        return mean, epistemic_std, aleatoric_std

    # ----------------------------------------------------- predict_total_std

    def predict_total_std(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, total_std)`` combining epistemic + aleatoric."""

        mean, epistemic_std, aleatoric_std = self.predict(X)
        total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
        return mean, total_std

    # ---------------------------------------------------------------- save

    def save(self, path: str) -> None:
        """Persist the ensemble to ``path`` via flax.serialization + joblib."""

        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained surrogate.")

        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc

        serialized_params = [
            flax_serialization.to_bytes(p) for p in self._params
        ]

        payload = {
            "config": self.config,
            "params_bytes": serialized_params,
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "n_features": self._n_features,
        }
        joblib.dump(payload, path)

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: str) -> BayesianFatigueSurrogate:
        """Load a previously saved ensemble."""

        if not _HAS_FLAX:
            raise ImportError(_MISSING_FLAX_MSG)

        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc

        payload = joblib.load(path)

        obj = cls(config=payload["config"])
        obj._x_mean = payload["x_mean"]
        obj._x_std = payload["x_std"]
        obj._y_mean = payload["y_mean"]
        obj._y_std = payload["y_std"]
        obj._n_features = payload["n_features"]

        cfg = obj.config
        obj._model = HeteroscedasticMLP(hidden_sizes=tuple(cfg.hidden_sizes))

        dummy = jnp.zeros((1, obj._n_features), dtype=jnp.float32)
        root_key = jax.random.PRNGKey(cfg.seed)
        member_keys = jax.random.split(root_key, cfg.n_members)

        obj._params = []
        for member_key, blob in zip(member_keys, payload["params_bytes"]):
            init_key, _ = jax.random.split(member_key)
            template = obj._model.init(init_key, dummy)
            restored = flax_serialization.from_bytes(template, blob)
            obj._params.append(restored)

        obj._is_trained = True
        return obj
