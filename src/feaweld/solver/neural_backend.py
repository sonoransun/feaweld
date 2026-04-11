"""Neural operator surrogate backend (Track A5).

A fixed-mesh DeepONet backend that loads a trained Flax model and
returns :class:`FEAResults` in O(10 ms). This is intended as a fast
surrogate for interactive / inverse / active-learning loops where the
mesh topology is fixed but loads and material parameters vary.

Design constraints
------------------
- DeepONet / FNO architectures do not generalize across mesh topology.
  The training metadata records a stable hash of the mesh topology,
  and :meth:`NeuralBackend.solve_static` hard-fails on a mismatch. The
  user is expected to retrain when the mesh changes.
- Mesh hash is **position-independent**: it hashes shapes, element
  type, and a prefix of the element connectivity table. Small
  geometric perturbations of node coordinates do *not* invalidate the
  surrogate.
- Flax / Optax / msgpack imports are guarded by ``_HAS_FLAX``; the
  module must be importable (for registration in :func:`get_backend`)
  even in a minimal environment without the ``[neural]`` extras.
- MVP scope: displacement only. Stress / strain are not predicted;
  callers that need them should re-run a physics backend on the
  surrogate's displacement output (or wait for a future track).

The training harness lives in :mod:`feaweld.solver.neural_training`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax import serialization
    import optax  # noqa: F401  (re-exported for training harness)
    import msgpack  # noqa: F401  (used for params.msgpack round-trip)
    _HAS_FLAX = True
except ImportError:
    _HAS_FLAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nn = None  # type: ignore
    serialization = None  # type: ignore
    optax = None  # type: ignore
    msgpack = None  # type: ignore

from feaweld.core.materials import Material
from feaweld.core.types import (
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend


# ---------------------------------------------------------------------------
# Mesh hashing: topology only, not geometry
# ---------------------------------------------------------------------------


def mesh_hash(mesh: FEMesh) -> str:
    """Hash the topological identity of a mesh.

    The hash combines node count, element count, element type, and a
    512-byte prefix of the connectivity table. Node coordinates are
    **not** hashed so that small geometric perturbations of the same
    topology still resolve to the same mesh (important for design /
    shape optimization loops that reuse the same surrogate).

    Parameters
    ----------
    mesh : FEMesh
        Mesh to hash.

    Returns
    -------
    str
        16-character lowercase hex digest.
    """
    h = hashlib.sha256()
    h.update(repr(mesh.nodes.shape).encode("utf-8"))
    h.update(b"|")
    h.update(repr(mesh.elements.shape).encode("utf-8"))
    h.update(b"|")
    h.update(mesh.element_type.value.encode("utf-8"))
    h.update(b"|")
    # Prefix of the connectivity bytes. Using a prefix keeps the hash
    # cheap for huge meshes while still being sensitive to any change
    # in the early rows of the connectivity table.
    elements_bytes = np.ascontiguousarray(
        mesh.elements.astype(np.int64)
    ).tobytes()
    h.update(elements_bytes[:512])
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


@dataclass
class NeuralBackendMetadata:
    """Training metadata persisted alongside the Flax parameters.

    Attributes
    ----------
    mesh_hash
        Topology hash from :func:`mesh_hash` at training time.
    load_feature_names
        Ordered names of features extracted from a
        ``(load_case, material, temperature)`` tuple.
    output_shape
        ``(n_nodes, n_components)`` shape of the displacement tensor
        produced by the trunk network.
    normalization
        Normalization statistics for inputs and outputs
        (``x_mean``, ``x_std``, ``y_mean``, ``y_std``) as lists so the
        dataclass serializes cleanly as JSON.
    architecture
        DeepONet architecture hyperparameters needed to reconstruct
        a matching :class:`DeepONet` module at load time.
    training
        Free-form training diagnostics (loss curve, epochs, etc.).
    """

    mesh_hash: str
    load_feature_names: list[str]
    output_shape: tuple[int, int]
    normalization: dict[str, list[float]]
    architecture: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        # output_shape is a tuple → list for JSON
        payload["output_shape"] = list(payload["output_shape"])
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "NeuralBackendMetadata":
        data = json.loads(text)
        data["output_shape"] = tuple(data["output_shape"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Feature extraction: LoadCase + Material + temperature -> flat vector
# ---------------------------------------------------------------------------


_DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "axial_force",
    "bending_moment",
    "shear_force",
    "pressure",
    "temperature",
    "E",
    "sigma_y",
)


def _scalar_force(values: Any, idx: int) -> float:
    """Pick component `idx` from a BC value vector, defaulting to 0."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if idx < arr.size:
        return float(arr[idx])
    return 0.0


def _load_features(
    load_case: LoadCase,
    material: Material | None,
    temperature: float,
    feature_names: tuple[str, ...] = _DEFAULT_FEATURE_NAMES,
) -> NDArray[np.float64]:
    """Pack ``(load_case, material, temperature)`` into a flat feature vector.

    The extractor intentionally accepts missing data: any feature that
    cannot be recovered defaults to 0.0, which keeps the contract
    between the training harness (which sets features directly) and
    the inference path (which reads from a LoadCase built by the
    pipeline) symmetric.
    """
    # Aggregate force components across all FORCE loads.
    fx = fy = fz = 0.0
    for bc in load_case.loads:
        if bc.bc_type != LoadType.FORCE:
            continue
        fx += _scalar_force(bc.values, 0)
        fy += _scalar_force(bc.values, 1)
        fz += _scalar_force(bc.values, 2)

    # Heuristic mapping from the standard pipeline force convention
    # (axial on y, bending encoded via moment-like BCs). These names
    # align with LoadConfig in feaweld.pipeline.workflow so dot-path
    # sweeps like "load.axial_force" feed through transparently.
    axial_force = fy
    bending_moment = 0.0  # pipeline converts moments to force pairs upstream
    shear_force = fx
    pressure = 0.0

    E_val = 0.0
    sigma_y = 0.0
    if material is not None:
        try:
            E_val = float(material.E(temperature))
        except Exception:
            E_val = 0.0
        try:
            sigma_y = float(material.sigma_y(temperature))
        except Exception:
            sigma_y = 0.0

    lookup = {
        "axial_force": axial_force,
        "bending_moment": bending_moment,
        "shear_force": shear_force,
        "pressure": pressure,
        "temperature": float(temperature),
        "E": E_val,
        "sigma_y": sigma_y,
    }
    return np.array([lookup.get(name, 0.0) for name in feature_names],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# DeepONet module (Flax)
# ---------------------------------------------------------------------------


if _HAS_FLAX:

    class DeepONet(nn.Module):
        """Standard vanilla DeepONet with MLP branch and trunk.

        The branch network encodes the load-parameter vector; the trunk
        encodes spatial coordinates (one row per node on the fixed
        training mesh). The output at a node is the inner product of
        the two latent vectors, replicated over the output components
        via a final linear head.
        """

        branch_layers: tuple[int, ...] = (64, 64, 64)
        trunk_layers: tuple[int, ...] = (64, 64, 64)
        latent_dim: int = 64
        out_components: int = 3

        @nn.compact
        def __call__(
            self,
            load_params: jnp.ndarray,
            coords: jnp.ndarray,
        ) -> jnp.ndarray:
            """Predict the displacement field.

            Parameters
            ----------
            load_params : (n_features,) or (batch, n_features)
                Load/material feature vector(s).
            coords : (n_nodes, ndim)
                Coordinates of the fixed training mesh nodes.

            Returns
            -------
            jnp.ndarray
                Shape ``(n_nodes, out_components)`` if load_params is 1D,
                otherwise ``(batch, n_nodes, out_components)``.
            """
            # Branch
            b = load_params
            for width in self.branch_layers:
                b = nn.Dense(width)(b)
                b = nn.tanh(b)
            b = nn.Dense(self.latent_dim * self.out_components)(b)
            # Reshape to (..., out_components, latent_dim)
            if b.ndim == 1:
                b = b.reshape(self.out_components, self.latent_dim)
            else:
                b = b.reshape(b.shape[0], self.out_components, self.latent_dim)

            # Trunk
            t = coords
            for width in self.trunk_layers:
                t = nn.Dense(width)(t)
                t = nn.tanh(t)
            t = nn.Dense(self.latent_dim)(t)  # (n_nodes, latent_dim)

            # Inner product:
            #   unbatched:  out[n, c] = sum_l b[c, l] * t[n, l]
            #   batched:    out[B, n, c] = sum_l b[B, c, l] * t[n, l]
            if b.ndim == 2:
                out = jnp.einsum("cl,nl->nc", b, t)
            else:
                out = jnp.einsum("bcl,nl->bnc", b, t)
            return out

else:  # pragma: no cover — no Flax installed
    class DeepONet:  # type: ignore[no-redef]
        """Stub that raises on instantiation when Flax is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "DeepONet requires Flax. Install with `pip install 'feaweld[neural]'`."
            )


def _build_module_from_meta(meta: "NeuralBackendMetadata") -> Any:
    """Instantiate a DeepONet whose shape matches the saved parameters.

    Missing architecture keys default to the :class:`DeepONet` class
    defaults so older metadata files remain readable (as long as they
    were trained with those defaults).
    """
    arch = meta.architecture or {}
    branch = tuple(arch.get("branch_layers", (64, 64, 64)))
    trunk = tuple(arch.get("trunk_layers", (64, 64, 64)))
    latent = int(arch.get("latent_dim", 64))
    n_out = int(meta.output_shape[1])
    return DeepONet(
        branch_layers=branch,
        trunk_layers=trunk,
        latent_dim=latent,
        out_components=n_out,
    )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class NeuralBackend(SolverBackend):
    """Fixed-mesh DeepONet surrogate backend.

    This backend only implements :meth:`solve_static`. Thermal /
    coupled solves raise :class:`NotImplementedError` because neural
    operators trained on a fixed mesh topology are a poor fit for
    transient / multiphysics problems in the MVP.

    Parameters
    ----------
    model_path : str or None
        If provided, load a trained model from this directory via
        :meth:`load_model`. If ``None``, the backend is constructed in
        an unloaded state and :meth:`solve_static` will raise a clear
        :class:`RuntimeError` pointing at the training harness.

    Notes
    -----
    The backend hashes the incoming mesh topology via :func:`mesh_hash`
    on every ``solve_static`` call and compares against the training
    metadata. Any mismatch raises :class:`ValueError` — the user is
    expected to retrain rather than silently extrapolate.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._loaded = False
        self._params: Any = None
        self._meta: NeuralBackendMetadata | None = None
        self._module: Any = None
        if model_path is not None:
            self.load_model(model_path)

    # -- Persistence ---------------------------------------------------------

    def load_model(self, path: str) -> None:
        """Load Flax params + metadata from a directory.

        Expected layout::

            <path>/
                params.msgpack    # flax.serialization dump of the Pytree
                meta.json         # NeuralBackendMetadata
        """
        if not _HAS_FLAX:
            raise ImportError(
                "NeuralBackend.load_model requires Flax. "
                "Install with `pip install 'feaweld[neural]'`."
            )
        root = Path(path)
        meta_path = root / "meta.json"
        params_path = root / "params.msgpack"
        if not meta_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"NeuralBackend model directory '{path}' must contain "
                "params.msgpack and meta.json."
            )

        self._meta = NeuralBackendMetadata.from_json(meta_path.read_text())

        # Build a module with the architecture recorded at training time.
        self._module = _build_module_from_meta(self._meta)

        # Initialize dummy params so we have a valid target for
        # flax.serialization.from_bytes.
        rng = jax.random.PRNGKey(0)
        dummy_loads = jnp.zeros((len(self._meta.load_feature_names),))
        dummy_coords = jnp.zeros((int(self._meta.output_shape[0]), 3))
        init_params = self._module.init(rng, dummy_loads, dummy_coords)

        raw = params_path.read_bytes()
        self._params = serialization.from_bytes(init_params, raw)
        self._loaded = True

    def save_model(
        self,
        path: str,
        params: Any,
        meta: NeuralBackendMetadata,
    ) -> None:
        """Write params + metadata to ``path`` (created if missing).

        Invoked by the training harness. Exposed on the backend so
        tests and scripts that already hold the backend instance can
        persist without pulling in the training module.
        """
        if not _HAS_FLAX:
            raise ImportError(
                "NeuralBackend.save_model requires Flax. "
                "Install with `pip install 'feaweld[neural]'`."
            )
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "params.msgpack").write_bytes(serialization.to_bytes(params))
        (root / "meta.json").write_text(meta.to_json())

    # -- Core solve ----------------------------------------------------------

    def solve_static(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> FEAResults:
        if not self._loaded:
            raise RuntimeError(
                "NeuralBackend has no trained model loaded. Either pass "
                "`model_path` to the constructor or train one via "
                "`feaweld.solver.neural_training.train_deeponet`."
            )
        assert self._meta is not None  # for the type checker
        assert self._module is not None

        got_hash = mesh_hash(mesh)
        if got_hash != self._meta.mesh_hash:
            raise ValueError(
                "NeuralBackend mesh-topology hash mismatch: "
                f"got {got_hash}, expected {self._meta.mesh_hash}. "
                "DeepONet surrogates are trained for a single fixed mesh; "
                "retrain the model with `train_deeponet` for this mesh."
            )

        feature_names = tuple(self._meta.load_feature_names)
        x = _load_features(load_case, material, temperature, feature_names)

        norm = self._meta.normalization
        x_mean = np.asarray(norm.get("x_mean", [0.0] * x.size), dtype=np.float64)
        x_std = np.asarray(norm.get("x_std", [1.0] * x.size), dtype=np.float64)
        y_mean = np.asarray(norm.get("y_mean", [0.0]), dtype=np.float64)
        y_std = np.asarray(norm.get("y_std", [1.0]), dtype=np.float64)

        x_std_safe = np.where(x_std == 0.0, 1.0, x_std)
        x_norm = (x - x_mean) / x_std_safe

        # Trunk expects (n_nodes, ndim). Pad 2D meshes to 3D on the fly
        # so a single module handles both.
        coords = mesh.nodes
        if coords.shape[1] == 2:
            coords = np.concatenate(
                [coords, np.zeros((coords.shape[0], 1))], axis=1
            )

        pred = self._module.apply(
            self._params,
            jnp.asarray(x_norm, dtype=jnp.float32),
            jnp.asarray(coords, dtype=jnp.float32),
        )
        pred_np = np.asarray(pred, dtype=np.float64)
        # De-normalize
        pred_np = pred_np * y_std + y_mean

        n_out = int(self._meta.output_shape[1])
        disp = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
        disp[:, :n_out] = pred_np[:, :n_out]

        return FEAResults(
            mesh=mesh,
            displacement=disp,
            stress=None,  # MVP: no stress/strain from surrogate
            strain=None,
            metadata={
                "backend": "neural",
                "architecture": "deeponet",
                "mesh_hash": got_hash,
                "features": list(feature_names),
                "note": (
                    "Surrogate prediction: displacement only. For stress "
                    "recover, re-run a physics backend on the predicted "
                    "displacement field."
                ),
            },
        )

    # -- Unsupported analyses ------------------------------------------------

    def solve_thermal_steady(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
    ) -> FEAResults:
        raise NotImplementedError(
            "NeuralBackend only supports solve_static in the A5 MVP. "
            "Use FEniCSBackend or JAXBackend for thermal analyses."
        )

    def solve_thermal_transient(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        time_steps: NDArray,
        heat_source: object | None = None,
    ) -> FEAResults:
        raise NotImplementedError(
            "NeuralBackend only supports solve_static in the A5 MVP. "
            "Use FEniCSBackend for transient thermal analyses."
        )

    def solve_coupled(
        self,
        mesh: FEMesh,
        material: Material,
        mechanical_lc: LoadCase,
        thermal_lc: LoadCase,
        time_steps: NDArray,
    ) -> FEAResults:
        raise NotImplementedError(
            "NeuralBackend only supports solve_static in the A5 MVP. "
            "Use FEniCSBackend for thermomechanically coupled analyses."
        )
