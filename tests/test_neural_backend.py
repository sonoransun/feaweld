"""Tests for the Track A5 neural operator surrogate backend.

Every test in this module requires the ``[neural]`` / ``[flax]`` extras
(``flax`` + ``optax``). Collection works without them via
``pytest.importorskip``.
"""

from __future__ import annotations

import numpy as np
import pytest

flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402  (imported after importorskip)

from feaweld.core.materials import Material  # noqa: E402
from feaweld.core.types import (  # noqa: E402
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend  # noqa: E402
from feaweld.solver.neural_backend import (  # noqa: E402
    DeepONet,
    NeuralBackend,
    NeuralBackendMetadata,
    _DEFAULT_FEATURE_NAMES,
    _load_features,
    mesh_hash,
)
from feaweld.solver.neural_training import (  # noqa: E402
    TrainingConfig,
    train_deeponet,
)


pytestmark = pytest.mark.requires_flax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_mesh() -> FEMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={
            "bottom": np.array([0, 1]),
            "top": np.array([2, 3]),
        },
    )


@pytest.fixture
def steel() -> Material:
    return Material(
        name="A36",
        density=7850.0,
        elastic_modulus={20.0: 200_000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 51.9},
        specific_heat={20.0: 440.0},
        thermal_expansion={20.0: 11.7e-6},
    )


@pytest.fixture
def pull_load_case() -> LoadCase:
    return LoadCase(
        name="tension",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, 1000.0, 0.0]),
            )
        ],
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0, 0.0]),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Unloaded backend
# ---------------------------------------------------------------------------


def test_neural_backend_raises_without_model(tiny_mesh, steel, pull_load_case):
    backend = NeuralBackend()
    with pytest.raises(RuntimeError, match="no trained model"):
        backend.solve_static(tiny_mesh, steel, pull_load_case)


def test_neural_backend_thermal_stubs_raise(tiny_mesh, steel, pull_load_case):
    backend = NeuralBackend()
    with pytest.raises(NotImplementedError):
        backend.solve_thermal_steady(tiny_mesh, steel, pull_load_case)
    with pytest.raises(NotImplementedError):
        backend.solve_thermal_transient(
            tiny_mesh, steel, pull_load_case, np.linspace(0, 1, 5)
        )
    with pytest.raises(NotImplementedError):
        backend.solve_coupled(
            tiny_mesh, steel, pull_load_case, pull_load_case, np.linspace(0, 1, 5)
        )


# ---------------------------------------------------------------------------
# Mesh hashing
# ---------------------------------------------------------------------------


def test_mesh_hash_is_stable(tiny_mesh):
    h1 = mesh_hash(tiny_mesh)
    h2 = mesh_hash(tiny_mesh)
    assert h1 == h2
    assert len(h1) == 16


def test_mesh_hash_ignores_node_coordinate_perturbations(tiny_mesh):
    """Small geometric perturbations must *not* invalidate the surrogate.

    The hash is specified to key on topology only so that shape
    optimization loops (which wiggle node coordinates) can reuse the
    same trained model.
    """
    perturbed = FEMesh(
        nodes=tiny_mesh.nodes + 1e-6,
        elements=tiny_mesh.elements.copy(),
        element_type=tiny_mesh.element_type,
        node_sets={k: v.copy() for k, v in tiny_mesh.node_sets.items()},
    )
    assert mesh_hash(perturbed) == mesh_hash(tiny_mesh)


def test_mesh_hash_detects_topology_change(tiny_mesh):
    changed_elements = tiny_mesh.elements.copy()
    # Swap two nodes in the first element → different connectivity.
    changed_elements[0, 1], changed_elements[0, 2] = (
        changed_elements[0, 2],
        changed_elements[0, 1],
    )
    changed = FEMesh(
        nodes=tiny_mesh.nodes.copy(),
        elements=changed_elements,
        element_type=tiny_mesh.element_type,
        node_sets={k: v.copy() for k, v in tiny_mesh.node_sets.items()},
    )
    assert mesh_hash(changed) != mesh_hash(tiny_mesh)


def test_mesh_hash_detects_element_count_change(tiny_mesh):
    reduced = FEMesh(
        nodes=tiny_mesh.nodes.copy(),
        elements=tiny_mesh.elements[:1].copy(),
        element_type=tiny_mesh.element_type,
        node_sets={k: v.copy() for k, v in tiny_mesh.node_sets.items()},
    )
    assert mesh_hash(reduced) != mesh_hash(tiny_mesh)


# ---------------------------------------------------------------------------
# DeepONet forward pass
# ---------------------------------------------------------------------------


def test_deeponet_forward_pass():
    module = DeepONet(
        branch_layers=(16, 16),
        trunk_layers=(16, 16),
        latent_dim=8,
        out_components=3,
    )
    rng = jax.random.PRNGKey(0)
    n_features = 5
    n_nodes = 7
    load_params = jnp.zeros((n_features,))
    coords = jnp.zeros((n_nodes, 3))
    params = module.init(rng, load_params, coords)
    out = module.apply(params, load_params, coords)
    assert out.shape == (n_nodes, 3)


def test_deeponet_batched_forward_pass():
    module = DeepONet(
        branch_layers=(8,), trunk_layers=(8,), latent_dim=4, out_components=2
    )
    rng = jax.random.PRNGKey(1)
    n_features = 3
    n_nodes = 5
    batch = 4
    params = module.init(rng, jnp.zeros((n_features,)), jnp.zeros((n_nodes, 3)))
    out = module.apply(
        params, jnp.ones((batch, n_features)), jnp.zeros((n_nodes, 3))
    )
    assert out.shape == (batch, n_nodes, 2)


# ---------------------------------------------------------------------------
# End-to-end round trip: save → load → solve → mesh-hash check
# ---------------------------------------------------------------------------


def test_neural_backend_roundtrip_and_topology_mismatch(tmp_path, tiny_mesh):
    """Manually stash a trained model and verify load + topology guard."""
    module = DeepONet(
        branch_layers=(8,), trunk_layers=(8,), latent_dim=4, out_components=3
    )
    rng = jax.random.PRNGKey(0)
    feature_names = list(_DEFAULT_FEATURE_NAMES)
    params = module.init(
        rng,
        jnp.zeros((len(feature_names),)),
        jnp.asarray(tiny_mesh.nodes),
    )

    meta = NeuralBackendMetadata(
        mesh_hash=mesh_hash(tiny_mesh),
        load_feature_names=feature_names,
        output_shape=(tiny_mesh.n_nodes, 3),
        normalization={
            "x_mean": [0.0] * len(feature_names),
            "x_std": [1.0] * len(feature_names),
            "y_mean": [0.0],
            "y_std": [1.0],
        },
        architecture={
            "branch_layers": [8],
            "trunk_layers": [8],
            "latent_dim": 4,
        },
    )
    backend = NeuralBackend()
    backend.save_model(str(tmp_path), params, meta)

    loaded = NeuralBackend(model_path=str(tmp_path))
    mat = Material(
        name="dummy",
        density=1.0,
        elastic_modulus={20.0: 1.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 1.0},
        ultimate_strength={20.0: 1.0},
        thermal_conductivity={20.0: 1.0},
        specific_heat={20.0: 1.0},
        thermal_expansion={20.0: 0.0},
    )
    lc = LoadCase(name="zero")
    result = loaded.solve_static(tiny_mesh, mat, lc)
    assert isinstance(result, FEAResults)
    assert result.displacement is not None
    assert result.displacement.shape == (tiny_mesh.n_nodes, 3)
    assert result.metadata["backend"] == "neural"

    # Change topology → mismatch must raise.
    bad_elements = tiny_mesh.elements.copy()
    bad_elements[0, 0], bad_elements[0, 1] = (
        bad_elements[0, 1],
        bad_elements[0, 0],
    )
    bad_mesh = FEMesh(
        nodes=tiny_mesh.nodes.copy(),
        elements=bad_elements,
        element_type=tiny_mesh.element_type,
        node_sets={k: v.copy() for k, v in tiny_mesh.node_sets.items()},
    )
    with pytest.raises(ValueError, match="mesh-topology hash mismatch"):
        loaded.solve_static(bad_mesh, mat, lc)


# ---------------------------------------------------------------------------
# Feature extraction sanity
# ---------------------------------------------------------------------------


def test_load_features_defaults_zero_for_empty_loadcase(steel):
    lc = LoadCase(name="empty")
    feats = _load_features(lc, steel, 20.0)
    assert feats.shape == (len(_DEFAULT_FEATURE_NAMES),)
    # Only material-derived entries should be non-zero.
    idx_force = _DEFAULT_FEATURE_NAMES.index("axial_force")
    assert feats[idx_force] == 0.0


# ---------------------------------------------------------------------------
# Trivial analytic training: u(x) = coeff * load at every node
# ---------------------------------------------------------------------------


class _FakeAnalyticBackend(SolverBackend):
    """Ground-truth backend whose displacement = coeff * axial_force everywhere.

    Used to drive :func:`train_deeponet` on a trivial regression target
    so the test can verify the harness end-to-end without any real FEA.
    """

    def __init__(self, mesh: FEMesh, coeff: float = 1e-3) -> None:
        self._mesh = mesh
        self._coeff = coeff

    def solve_static(self, mesh, material, load_case, temperature=20.0):
        # Extract the "axial_force" feature the same way the backend does.
        feats = _load_features(load_case, material, temperature)
        idx = _DEFAULT_FEATURE_NAMES.index("axial_force")
        val = float(feats[idx])
        disp = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
        disp[:, 1] = self._coeff * val
        return FEAResults(
            mesh=mesh,
            displacement=disp,
            stress=StressField(values=np.zeros((mesh.n_nodes, 6))),
            strain=np.zeros((mesh.n_nodes, 6)),
        )

    def solve_thermal_steady(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def solve_thermal_transient(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def solve_coupled(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError


def test_train_deeponet_on_trivial_analytic_problem(tmp_path, tiny_mesh, steel):
    """Train end-to-end on a linear target and verify <20% error on holdout."""
    from feaweld.pipeline.workflow import AnalysisCase

    base_case = AnalysisCase()
    coeff = 1e-4
    fake = _FakeAnalyticBackend(tiny_mesh, coeff=coeff)

    def fake_realize(case):
        return tiny_mesh, steel, LoadCase(name="train")

    cfg = TrainingConfig(
        n_samples=100,
        n_epochs=30,
        learning_rate=5e-3,
        batch_size=32,
        seed=7,
        latent_dim=16,
    )
    out_dir = tmp_path / "trained"
    train_deeponet(
        base_case=base_case,
        param_sweep={"load.axial_force": (0.0, 10_000.0)},
        output_path=str(out_dir),
        config=cfg,
        backend=fake,
        realize_case_fn=fake_realize,
    )

    loaded = NeuralBackend(model_path=str(out_dir))

    # Held-out axial force value not drawn from the training RNG.
    test_force = 6543.0
    holdout_case = LoadCase(
        name="holdout",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, test_force, 0.0]),
            )
        ],
    )
    pred = loaded.solve_static(tiny_mesh, steel, holdout_case)
    expected = coeff * test_force
    err = np.abs(pred.displacement[:, 1] - expected) / max(abs(expected), 1e-9)
    # Keep the tolerance loose — this is a CI sanity check, not a
    # convergence study. 20% matches the Track A5 plan.
    assert np.mean(err) < 0.2
