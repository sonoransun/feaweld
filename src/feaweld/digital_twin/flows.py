"""RealNVP normalizing flow for posterior calibration (Track C3).

A small RealNVP affine-coupling flow trained on posterior samples over
log-fatigue-life (or any low-dimensional scalar quantity of interest) to
enable quantile estimation and posterior calibration.

Flax / Optax / JAX are optional dependencies; this module imports cleanly
without them but any attempt to instantiate the flow raises ``ImportError``
pointing the user at ``pip install 'feaweld[flax]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "RealNVPFlow requires Flax / Optax / JAX. "
    "Install with: pip install 'feaweld[flax]'"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RealNVPConfig:
    """Hyper-parameters for the RealNVP normalizing flow."""

    n_layers: int = 6
    hidden_sizes: tuple[int, ...] = (32, 32)
    n_epochs: int = 500
    learning_rate: float = 1e-3
    seed: int = 0


# ---------------------------------------------------------------------------
# Flax coupling network
# ---------------------------------------------------------------------------


if _HAS_FLAX:

    class _CouplingMLP(nn.Module):
        """Small MLP that outputs ``(shift, scale)`` for an affine coupling."""

        hidden_sizes: tuple[int, ...]
        out_dim: int

        @nn.compact
        def __call__(self, x):  # type: ignore[override]
            h = x
            for width in self.hidden_sizes:
                h = nn.Dense(width)(h)
                h = nn.swish(h)
            # Zero-init final layers so the flow starts as the identity map.
            zero_init = nn.initializers.zeros
            shift = nn.Dense(self.out_dim, kernel_init=zero_init, bias_init=zero_init)(h)
            raw_scale = nn.Dense(
                self.out_dim, kernel_init=zero_init, bias_init=zero_init
            )(h)
            # Bound log-scale with tanh * 2 to keep the Jacobian finite.
            log_scale = 2.0 * jnp.tanh(raw_scale)
            return shift, log_scale

    class _RealNVPModule(nn.Module):
        """Stack of affine coupling layers with alternating masks."""

        dim: int
        n_layers: int
        hidden_sizes: tuple[int, ...]

        def setup(self) -> None:
            half = self.dim // 2
            self.couplings = [
                _CouplingMLP(
                    hidden_sizes=tuple(self.hidden_sizes),
                    out_dim=self.dim - half,
                )
                for _ in range(self.n_layers)
            ]

        def _split(self, x, layer_idx: int):
            half = self.dim // 2
            if layer_idx % 2 == 0:
                x_a = x[..., :half]
                x_b = x[..., half:]
            else:
                x_a = x[..., half:]
                x_b = x[..., :half]
            return x_a, x_b

        def _merge(self, x_a, x_b, layer_idx: int):
            if layer_idx % 2 == 0:
                return jnp.concatenate([x_a, x_b], axis=-1)
            return jnp.concatenate([x_b, x_a], axis=-1)

        def forward(self, x):
            """Map data -> latent; returns (z, log_det_jac)."""
            log_det = jnp.zeros(x.shape[:-1])
            z = x
            for i, coupling in enumerate(self.couplings):
                a, b = self._split(z, i)
                shift, log_scale = coupling(a)
                b = (b - shift) * jnp.exp(-log_scale)
                log_det = log_det - jnp.sum(log_scale, axis=-1)
                z = self._merge(a, b, i)
            return z, log_det

        def inverse(self, z):
            """Map latent -> data; returns x."""
            x = z
            for i in reversed(range(len(self.couplings))):
                a, b = self._split(x, i)
                shift, log_scale = self.couplings[i](a)
                b = b * jnp.exp(log_scale) + shift
                x = self._merge(a, b, i)
            return x

        def log_prob(self, x):
            z, log_det = self.forward(x)
            log_base = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * self.dim * jnp.log(
                2.0 * jnp.pi
            )
            return log_base + log_det

        def __call__(self, x):  # type: ignore[override]
            return self.log_prob(x)

else:  # pragma: no cover - exercised only when extras missing

    class _CouplingMLP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_MISSING_FLAX_MSG)

    class _RealNVPModule:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_MISSING_FLAX_MSG)


# ---------------------------------------------------------------------------
# Public flow class
# ---------------------------------------------------------------------------


class RealNVPFlow:
    """1-D or low-dim RealNVP normalizing flow for posterior calibration.

    Usage::

        flow = RealNVPFlow(dim=1, config=RealNVPConfig())
        flow.fit(samples)                 # (n_samples, dim)
        log_prob = flow.log_prob(x)       # (n,)
        gen = flow.sample(n=1000)         # (1000, dim)
        q = flow.quantile(alpha=0.95)     # scalar for 1-D flows
    """

    def __init__(self, dim: int, config: RealNVPConfig | None = None) -> None:
        if not _HAS_FLAX:
            raise ImportError(_MISSING_FLAX_MSG)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        self.dim = int(dim)
        self.config = config or RealNVPConfig()
        # Internal flow works on at least 2 dimensions; 1-D samples are
        # augmented with an auxiliary standard normal column.
        self._augmented = self.dim == 1
        self._internal_dim = 2 if self._augmented else self.dim

        self._model: _RealNVPModule | None = None
        self._params: Any = None
        self._x_mean: NDArray[np.float64] | None = None
        self._x_std: NDArray[np.float64] | None = None
        self._is_trained: bool = False

    # --------------------------------------------------------------- helpers

    def _augment(self, samples: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        if not self._augmented:
            return samples
        aux = rng.standard_normal(size=(samples.shape[0], 1))
        return np.concatenate([samples, aux], axis=1)

    def _normalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self._x_mean) / self._x_std

    def _denormalize(self, xn: NDArray[np.float64]) -> NDArray[np.float64]:
        return xn * self._x_std + self._x_mean

    # --------------------------------------------------------------- fit

    def fit(self, samples: NDArray[np.float64]) -> None:
        """Train the flow on ``samples`` by maximum likelihood."""

        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2:
            raise ValueError(f"samples must be 1-D or 2-D, got shape {samples.shape}")
        if samples.shape[1] != self.dim:
            raise ValueError(
                f"Sample dim {samples.shape[1]} does not match flow dim {self.dim}"
            )

        rng = np.random.default_rng(self.config.seed)
        aug = self._augment(samples, rng)

        self._x_mean = aug.mean(axis=0)
        self._x_std = aug.std(axis=0) + 1e-8

        aug_n = self._normalize(aug)
        xj = jnp.asarray(aug_n, dtype=jnp.float32)

        cfg = self.config
        self._model = _RealNVPModule(
            dim=self._internal_dim,
            n_layers=cfg.n_layers,
            hidden_sizes=tuple(cfg.hidden_sizes),
        )

        key = jax.random.PRNGKey(cfg.seed)
        init_key, _ = jax.random.split(key)
        params = self._model.init(init_key, xj[:1])

        def nll_loss(params, xb):
            log_p = self._model.apply(params, xb)
            return -jnp.mean(log_p)

        grad_fn = jax.jit(jax.value_and_grad(nll_loss))
        optimizer = optax.adam(cfg.learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, xb):
            loss, grads = grad_fn(params, xb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for _ in range(cfg.n_epochs):
            params, opt_state, _ = step(params, opt_state, xj)

        self._params = params
        self._is_trained = True

    # --------------------------------------------------------------- log_prob

    def log_prob(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log-density of ``x`` under the trained flow.

        For the 1-D augmented case the returned log-density is the joint
        log-density over ``(x, aux)`` where ``aux`` is drawn from a fresh
        standard normal; the marginal over ``x`` is consistent in expectation
        but the per-sample values include the auxiliary contribution.
        """

        if not self._is_trained or self._model is None:
            raise RuntimeError("Flow is not trained. Call fit() first.")

        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Input dim {x.shape[1]} does not match flow dim {self.dim}"
            )

        rng = np.random.default_rng(self.config.seed + 1)
        aug = self._augment(x, rng)
        aug_n = self._normalize(aug)

        # Account for the normalizing linear change of variables: the density
        # in the original (augmented) space is obtained by dividing by the
        # product of per-dim standard deviations.
        log_p_n = np.asarray(
            self._model.apply(self._params, jnp.asarray(aug_n, dtype=jnp.float32)),
            dtype=np.float64,
        )
        log_det_norm = float(np.sum(np.log(self._x_std)))
        return log_p_n - log_det_norm

    # --------------------------------------------------------------- sample

    def sample(self, n: int, seed: int | None = None) -> NDArray[np.float64]:
        """Draw ``n`` samples from the trained flow (in original space)."""

        if not self._is_trained or self._model is None:
            raise RuntimeError("Flow is not trained. Call fit() first.")

        seed = self.config.seed + 7 if seed is None else seed
        key = jax.random.PRNGKey(seed)
        z = jax.random.normal(key, shape=(n, self._internal_dim))
        x_n = self._model.apply(
            self._params, z, method=self._model.inverse
        )
        x_n_np = np.asarray(x_n, dtype=np.float64)
        x_aug = self._denormalize(x_n_np)
        if self._augmented:
            return x_aug[:, :1]
        return x_aug

    # --------------------------------------------------------------- quantile

    def quantile(self, alpha: float, n_samples: int = 10000) -> float:
        """Monte-Carlo empirical ``alpha``-quantile of the first marginal.

        Returned value is a Monte-Carlo estimate (consistent as
        ``n_samples -> inf``) rather than an analytic inverse CDF.
        """

        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        samples = self.sample(n_samples)
        first = samples[:, 0]
        return float(np.quantile(first, alpha))

    # ---------------------------------------------------------------- save

    def save(self, path: str) -> None:
        """Persist the flow to ``path`` via flax.serialization + joblib."""

        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained flow.")

        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc

        payload = {
            "config": self.config,
            "dim": self.dim,
            "params_bytes": flax_serialization.to_bytes(self._params),
            "x_mean": self._x_mean,
            "x_std": self._x_std,
        }
        joblib.dump(payload, path)

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: str) -> RealNVPFlow:
        """Load a previously saved flow."""

        if not _HAS_FLAX:
            raise ImportError(_MISSING_FLAX_MSG)

        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc

        payload = joblib.load(path)

        obj = cls(dim=payload["dim"], config=payload["config"])
        obj._x_mean = payload["x_mean"]
        obj._x_std = payload["x_std"]

        cfg = obj.config
        obj._model = _RealNVPModule(
            dim=obj._internal_dim,
            n_layers=cfg.n_layers,
            hidden_sizes=tuple(cfg.hidden_sizes),
        )
        dummy = jnp.zeros((1, obj._internal_dim), dtype=jnp.float32)
        key = jax.random.PRNGKey(cfg.seed)
        init_key, _ = jax.random.split(key)
        template = obj._model.init(init_key, dummy)
        obj._params = flax_serialization.from_bytes(template, payload["params_bytes"])
        obj._is_trained = True
        return obj


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def calibrate_posterior_life(
    surrogate_samples: NDArray[np.float64],
    config: RealNVPConfig | None = None,
) -> RealNVPFlow:
    """Fit a 1-D RealNVP flow to posterior samples over ``log10(N_f)``.

    Parameters
    ----------
    surrogate_samples : NDArray
        Shape ``(n_samples,)`` or ``(n_samples, 1)`` posterior samples of
        log-fatigue-life.
    config : RealNVPConfig, optional
        Flow hyper-parameters; defaults to :class:`RealNVPConfig`.

    Returns
    -------
    RealNVPFlow
        A trained 1-D flow exposing ``.quantile``, ``.sample`` and
        ``.log_prob`` methods.
    """

    samples = np.asarray(surrogate_samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[1] != 1:
        raise ValueError(
            "calibrate_posterior_life expects 1-D posterior samples "
            f"(shape (n,) or (n, 1)); got {samples.shape}"
        )

    flow = RealNVPFlow(dim=1, config=config)
    flow.fit(samples)
    return flow
