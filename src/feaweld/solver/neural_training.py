"""DeepONet training harness for the neural operator surrogate backend.

This module glues together three pieces:

1. An :class:`~feaweld.pipeline.workflow.AnalysisCase` describing the
   fixed geometry / mesh / material / BCs that the surrogate is
   trained for.
2. A ``param_sweep`` dictionary giving ``(min, max)`` ranges for
   dot-paths like ``"load.axial_force"``. Random samples are drawn
   uniformly from each range.
3. A ground-truth :class:`~feaweld.solver.backend.SolverBackend`
   (``"jax"``, ``"fenics"``, ``"calculix"``) that produces the
   reference displacement field for each sampled parameter combo.

The trained Flax parameters and metadata (including mesh topology
hash) are written to ``<output_path>/params.msgpack`` +
``<output_path>/meta.json`` and can be loaded at inference time via
:class:`~feaweld.solver.neural_backend.NeuralBackend`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    _HAS_FLAX = True
except ImportError:
    _HAS_FLAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    optax = None  # type: ignore

from feaweld.core.types import FEMesh, LoadCase
from feaweld.pipeline.workflow import AnalysisCase, load_case
from feaweld.solver.backend import SolverBackend, get_backend
from feaweld.solver.neural_backend import (
    DeepONet,
    NeuralBackend,
    NeuralBackendMetadata,
    _DEFAULT_FEATURE_NAMES,
    _load_features,
    mesh_hash,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for :func:`train_deeponet`.

    The defaults are sized for real research runs; CI uses small
    overrides (``n_samples=100, n_epochs=30``) to keep total wall time
    under a minute on CPU.
    """

    n_samples: int = 2000
    n_epochs: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 128
    seed: int = 0
    latent_dim: int = 64


# ---------------------------------------------------------------------------
# Helpers: pydantic dot-path setattr (mirrors pipeline.study._set_nested_attr
# but operates on an in-memory deepcopy so the caller's base_case is not
# mutated across the sample loop).
# ---------------------------------------------------------------------------


def _set_nested(case: AnalysisCase, path: str, value: Any) -> AnalysisCase:
    parts = path.split(".")
    if len(parts) == 1:
        return case.model_copy(update={parts[0]: value})
    top_key = parts[0]
    sub_model = getattr(case, top_key)
    updated_sub = sub_model.model_copy(update={parts[1]: value})
    return case.model_copy(update={top_key: updated_sub})


# ---------------------------------------------------------------------------
# Build mesh + material + loadcase from an AnalysisCase without running the
# full workflow. We reuse the same helpers the workflow uses so the
# ground-truth sample sees exactly what a production run would.
# ---------------------------------------------------------------------------


def _realize_case(case: AnalysisCase) -> tuple[FEMesh, Any, LoadCase]:
    """Build (mesh, material, load_case) from an AnalysisCase.

    Kept private: the workflow module already owns this translation
    but wraps it in a top-level run. We replicate just the deterministic
    pre-solver steps here so training can sweep tens of thousands of
    cases without re-running post-processing each time.
    """
    from feaweld.core.materials import load_material
    from feaweld.mesh.generator import generate_mesh, WeldMeshConfig
    from feaweld.pipeline.workflow import _build_geometry, _build_load_case

    base = load_material(case.material.base_metal)
    joint = _build_geometry(case.geometry)
    mesh_cfg = WeldMeshConfig(
        global_size=case.mesh.global_size,
        weld_toe_size=case.mesh.weld_toe_size,
        element_order=case.mesh.element_order,
        element_type_2d=case.mesh.element_type,
    )
    mesh = generate_mesh(joint, mesh_cfg)
    lc = _build_load_case(case.load, mesh)
    return mesh, base, lc


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------


def _sample_parameters(
    base_case: AnalysisCase,
    param_sweep: dict[str, tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[AnalysisCase]:
    """Draw `n_samples` random perturbations of `base_case`."""
    param_names = list(param_sweep.keys())
    lows = np.array([param_sweep[k][0] for k in param_names], dtype=np.float64)
    highs = np.array([param_sweep[k][1] for k in param_names], dtype=np.float64)
    values = rng.uniform(lows, highs, size=(n_samples, len(param_names)))

    cases: list[AnalysisCase] = []
    for row in values:
        case = copy.deepcopy(base_case)
        for name, v in zip(param_names, row):
            case = _set_nested(case, name, float(v))
        cases.append(case)
    return cases


def _collect_training_data(
    cases: list[AnalysisCase],
    mesh: FEMesh,
    material: Any,
    backend: SolverBackend,
    feature_names: tuple[str, ...],
    temperature_from_case: Callable[[AnalysisCase], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ground-truth backend on each case and stack (X, y).

    Returns
    -------
    X : (n_samples, n_features) float64
    y : (n_samples, n_nodes, 3) float64
    """
    X = np.zeros((len(cases), len(feature_names)), dtype=np.float64)
    y = np.zeros((len(cases), mesh.n_nodes, 3), dtype=np.float64)

    for i, case in enumerate(cases):
        # Build a LoadCase from the perturbed config (same mesh).
        from feaweld.pipeline.workflow import _build_load_case
        lc = _build_load_case(case.load, mesh)
        temp = temperature_from_case(case)
        result = backend.solve_static(mesh, material, lc, temperature=temp)

        X[i] = _load_features(lc, material, temp, feature_names)
        if result.displacement is not None:
            disp = np.asarray(result.displacement, dtype=np.float64)
            if disp.shape[1] == 2:
                disp_full = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
                disp_full[:, :2] = disp
                disp = disp_full
            y[i] = disp
    return X, y


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _standardize(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std_safe = np.where(x_std == 0.0, 1.0, x_std)
    X_n = (X - x_mean) / x_std_safe

    # Single global (scalar) scale for y so the DeepONet output can be
    # denormalized with a tiny sidecar on the inference path.
    y_mean = np.array([y.mean()], dtype=np.float64)
    y_std_val = float(y.std())
    if y_std_val == 0.0:
        y_std_val = 1.0
    y_std = np.array([y_std_val], dtype=np.float64)
    y_n = (y - y_mean) / y_std

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std_safe.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    return X_n, y_n, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_flax_deeponet(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    config: TrainingConfig,
) -> tuple[Any, list[float]]:
    """Train a DeepONet with Adam + MSE. Returns (params, loss_curve)."""
    if not _HAS_FLAX:
        raise ImportError(
            "train_deeponet requires Flax / Optax. Install with "
            "`pip install 'feaweld[neural]'`."
        )

    n_samples, n_features = X.shape
    n_nodes, n_components = y.shape[1], y.shape[2]

    module = DeepONet(
        branch_layers=(config.latent_dim, config.latent_dim, config.latent_dim),
        trunk_layers=(config.latent_dim, config.latent_dim, config.latent_dim),
        latent_dim=config.latent_dim,
        out_components=n_components,
    )
    rng = jax.random.PRNGKey(config.seed)

    X_j = jnp.asarray(X, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    coords_j = jnp.asarray(coords, dtype=jnp.float32)

    params = module.init(rng, X_j[0], coords_j)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(p, xb, yb):
        # Vmap over the batch dimension of the load params; coords is
        # shared across the batch.
        pred = jax.vmap(lambda x: module.apply(p, x, coords_j))(xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(p, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    loss_curve: list[float] = []
    batch_size = min(config.batch_size, n_samples)
    perm_rng = np.random.default_rng(config.seed + 1)

    for epoch in range(config.n_epochs):
        order = perm_rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = order[start:start + batch_size]
            xb = X_j[idx]
            yb = y_j[idx]
            params, opt_state, loss = step(params, opt_state, xb, yb)
            epoch_loss += float(loss)
            n_batches += 1
        loss_curve.append(epoch_loss / max(n_batches, 1))

    return params, loss_curve


# ---------------------------------------------------------------------------
# Public training entrypoint
# ---------------------------------------------------------------------------


def train_deeponet(
    base_case: AnalysisCase,
    param_sweep: dict[str, tuple[float, float]],
    output_path: str,
    ground_truth_backend_name: str = "jax",
    config: TrainingConfig | None = None,
    *,
    backend: SolverBackend | None = None,
    realize_case_fn: Callable[[AnalysisCase], tuple[FEMesh, Any, LoadCase]] | None = None,
) -> None:
    """Sweep `base_case` and train a DeepONet on the resulting dataset.

    Parameters
    ----------
    base_case
        Starting analysis case. Perturbations are applied as deepcopies
        so the caller's instance is untouched.
    param_sweep
        Mapping from dot-path (e.g. ``"load.axial_force"``) to
        ``(min, max)`` uniform sampling range.
    output_path
        Directory that will receive ``params.msgpack`` + ``meta.json``.
        Created if it does not exist.
    ground_truth_backend_name
        Preference string passed to :func:`get_backend` to acquire the
        reference solver. Ignored if ``backend`` is passed.
    config
        :class:`TrainingConfig`. Defaults to the class defaults.
    backend
        Pre-built ground-truth backend. Overrides
        ``ground_truth_backend_name``; primarily for tests that want a
        fake analytic backend.
    realize_case_fn
        Advanced hook to replace :func:`_realize_case` (e.g. for tests
        that want to skip gmsh entirely by providing a synthetic mesh).
    """
    if not _HAS_FLAX:
        raise ImportError(
            "train_deeponet requires Flax / Optax. Install with "
            "`pip install 'feaweld[neural]'`."
        )
    cfg = config or TrainingConfig()
    rng = np.random.default_rng(cfg.seed)

    gt_backend = backend if backend is not None else get_backend(
        ground_truth_backend_name
    )

    realize = realize_case_fn or _realize_case
    mesh, material, _base_lc = realize(base_case)
    topo_hash = mesh_hash(mesh)
    feature_names = _DEFAULT_FEATURE_NAMES

    cases = _sample_parameters(base_case, param_sweep, cfg.n_samples, rng)
    X, y = _collect_training_data(
        cases,
        mesh,
        material,
        gt_backend,
        feature_names,
        temperature_from_case=lambda c: float(c.material.temperature),
    )

    X_n, y_n, stats = _standardize(X, y)

    coords = mesh.nodes
    if coords.shape[1] == 2:
        coords = np.concatenate(
            [coords, np.zeros((coords.shape[0], 1))], axis=1
        )

    params, loss_curve = _train_flax_deeponet(X_n, y_n, coords, cfg)

    meta = NeuralBackendMetadata(
        mesh_hash=topo_hash,
        load_feature_names=list(feature_names),
        output_shape=(mesh.n_nodes, 3),
        normalization=stats,
        architecture={
            "branch_layers": [cfg.latent_dim, cfg.latent_dim, cfg.latent_dim],
            "trunk_layers": [cfg.latent_dim, cfg.latent_dim, cfg.latent_dim],
            "latent_dim": cfg.latent_dim,
        },
        training={
            "n_samples": cfg.n_samples,
            "n_epochs": cfg.n_epochs,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "latent_dim": cfg.latent_dim,
            "loss_curve": loss_curve,
            "ground_truth_backend": ground_truth_backend_name,
        },
    )

    out_backend = NeuralBackend()
    out_backend.save_model(output_path, params, meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover — thin argparse shim
    parser = argparse.ArgumentParser(
        description="Train a DeepONet surrogate for a fixed-mesh AnalysisCase."
    )
    parser.add_argument(
        "--base-case",
        required=True,
        help="Path to an AnalysisCase YAML file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write params.msgpack + meta.json.",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--backend",
        default="jax",
        help="Ground-truth backend preference (jax / fenics / calculix).",
    )
    parser.add_argument(
        "--axial-range",
        type=float,
        nargs=2,
        default=(0.0, 10_000.0),
        help="(min, max) uniform range for load.axial_force.",
    )
    args = parser.parse_args()

    base = load_case(args.base_case)
    cfg = TrainingConfig(n_samples=args.n_samples, n_epochs=args.epochs)
    train_deeponet(
        base_case=base,
        param_sweep={"load.axial_force": tuple(args.axial_range)},
        output_path=args.output,
        ground_truth_backend_name=args.backend,
        config=cfg,
    )


if __name__ == "__main__":  # pragma: no cover
    _cli()
