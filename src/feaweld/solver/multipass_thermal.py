"""Multi-pass welding thermal cycle solver with element birth-death.

Orchestrates per-pass transient thermal solves, interpass cooldown
monitoring, and temperature history accumulation across an entire
:class:`~feaweld.core.types.WeldSequence`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.logging import get_logger
from feaweld.core.materials import Material
from feaweld.core.types import (
    FEAResults,
    FEMesh,
    LoadCase,
    StressField,
    WeldPass,
    WeldSequence,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiPassThermalConfig:
    """Configuration for :func:`solve_multipass_thermal`."""

    preheat_temp: float = 20.0
    interpass_temp_max: float = 250.0
    interpass_cooldown_s: float = 0.0
    cooldown_dt: float = 1.0
    ambient_temp: float = 20.0
    steps_per_pass: int = 50
    compute_residual_stress: bool = False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class InterpassCheckResult:
    """Outcome of one interpass temperature check."""

    pass_order: int
    max_temperature: float
    cooldown_time: float
    passed: bool


@dataclass
class MultiPassThermalResult:
    """Results from a multi-pass thermal cycle simulation."""

    temperature_history: NDArray
    time_steps: NDArray
    interpass_checks: list[InterpassCheckResult]
    per_pass_results: list[FEAResults]
    final_fea_results: FEAResults
    element_activation_log: list[dict[str, Any]] = field(default_factory=list)
    residual_stress: NDArray | None = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def compute_element_centroids(mesh: FEMesh) -> NDArray:
    """Return ``(n_elements, 3)`` element centroids from node coordinates."""
    coords = np.asarray(mesh.nodes, dtype=np.float64)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    elems = mesh.elements.astype(np.int64)
    return coords[elems].mean(axis=1)


def build_element_birth_death(
    mesh: FEMesh, sequence: WeldSequence,
) -> dict[int, Any]:
    """Build per-pass :class:`ElementBirthDeath` from mesh physical groups.

    Looks for groups named ``pass_<order>`` matching the passes in
    *sequence*.  Returns an empty dict if no matching groups exist.
    """
    from feaweld.solver.thermal import ElementBirthDeath

    centroids = compute_element_centroids(mesh)
    result: dict[int, ElementBirthDeath] = {}

    for wp in sequence.passes:
        group_name = f"pass_{wp.order}"
        elem_ids: NDArray | None = None
        if hasattr(mesh, "physical_groups") and mesh.physical_groups:
            elem_ids = mesh.physical_groups.get(group_name)
        if hasattr(mesh, "element_sets") and mesh.element_sets:
            if elem_ids is None:
                elem_ids = mesh.element_sets.get(group_name)
        if elem_ids is not None:
            result[wp.order] = ElementBirthDeath(
                element_centroids=centroids,
                weld_element_ids=np.asarray(elem_ids, dtype=np.int64),
            )
    return result


class _MaskedHeatSource:
    """Wraps a heat source and zeros output at dead-element nodes."""

    def __init__(
        self,
        inner: Any,
        mesh: FEMesh,
        birth_death: Any | None,
        travel_direction: NDArray | None = None,
    ) -> None:
        self._inner = inner
        self._bd = birth_death
        self._direction = (
            np.asarray(travel_direction, dtype=np.float64)
            if travel_direction is not None
            else np.array([1.0, 0.0, 0.0])
        )
        # Build node-to-element adjacency for dead-node detection
        self._node_elems: dict[int, set[int]] = {}
        elems = mesh.elements.astype(np.int64)
        for eidx in range(len(elems)):
            for nid in elems[eidx]:
                self._node_elems.setdefault(int(nid), set()).add(eidx)
        self._mesh = mesh

    def evaluate(
        self, x: NDArray, y: NDArray, z: NDArray, t: float,
    ) -> NDArray:
        q = self._inner.evaluate(x, y, z, t)
        if self._bd is None:
            return np.asarray(q)

        # Update activation based on current torch position
        if hasattr(self._inner, "start_position") and hasattr(self._inner, "travel_speed"):
            torch_pos = (
                self._inner.start_position + self._direction * self._inner.travel_speed * t
            )
            self._bd.update(torch_pos, self._direction)

        q = np.asarray(q, dtype=np.float64).copy()
        dead_ids = set(self._bd.dead_element_ids.tolist())
        if not dead_ids:
            return q

        # Zero heat at nodes that belong exclusively to dead elements
        flat_q = q.ravel()
        coords = np.asarray(self._mesh.nodes, dtype=np.float64)
        n_nodes = len(coords)
        if len(flat_q) == n_nodes:
            for nid, elem_set in self._node_elems.items():
                if elem_set.issubset(dead_ids):
                    flat_q[nid] = 0.0
        return q

    def total_energy_rate(self) -> float:
        return self._inner.total_energy_rate()


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_multipass_thermal(
    backend: Any,
    mesh: FEMesh,
    material: Material,
    sequence: WeldSequence,
    thermal_lc: LoadCase,
    config: MultiPassThermalConfig | None = None,
) -> MultiPassThermalResult:
    """Run a multi-pass transient thermal simulation.

    For each pass in *sequence*, runs a transient thermal solve using the
    appropriate Goldak heat source, then an interpass cooldown phase.
    Temperature is carried between passes via the ``initial_temperature``
    parameter.

    Parameters
    ----------
    backend
        A :class:`~feaweld.solver.backend.SolverBackend` instance.
    mesh
        Finite element mesh, optionally with ``pass_<N>`` physical groups
        for element birth-death.
    material
        Temperature-dependent material.
    sequence
        Ordered :class:`WeldSequence` defining the weld passes.
    thermal_lc
        Thermal boundary conditions (convection, etc.).
    config
        Optional :class:`MultiPassThermalConfig`.

    Returns
    -------
    MultiPassThermalResult
    """
    from feaweld.solver.thermal import GoldakHeatSource

    cfg = config or MultiPassThermalConfig()
    n_nodes = mesh.n_nodes

    # Initialise temperature field
    T_current = np.full(n_nodes, cfg.preheat_temp, dtype=np.float64)

    # Build element birth-death (empty dict if no physical groups)
    bd_map = build_element_birth_death(mesh, sequence)

    all_temps: list[NDArray] = []
    all_times: list[NDArray] = []
    per_pass_results: list[FEAResults] = []
    interpass_checks: list[InterpassCheckResult] = []
    activation_log: list[dict[str, Any]] = []
    global_offset = 0.0

    for pass_idx, wp in enumerate(sequence.passes):
        logger.info("Multi-pass thermal: pass %d (%s)", wp.order, wp.pass_type)

        # Construct per-pass Goldak heat source
        power = wp.voltage * wp.current * wp.efficiency
        src = GoldakHeatSource(
            power=power,
            a_f=5.0, a_r=10.0, b=5.0, c=5.0,
            travel_speed=wp.travel_speed,
        )

        bd = bd_map.get(wp.order)
        masked = _MaskedHeatSource(src, mesh, bd)

        pass_times = np.linspace(0.0, wp.duration, cfg.steps_per_pass)

        pass_result = backend.solve_thermal_transient(
            mesh=mesh,
            material=material,
            load_case=thermal_lc,
            time_steps=pass_times,
            heat_source=masked,
            initial_temperature=T_current,
        )
        per_pass_results.append(pass_result)

        # Extract temperature history
        if pass_result.temperature is not None:
            temp_arr = np.asarray(pass_result.temperature)
            if temp_arr.ndim == 1:
                temp_arr = temp_arr.reshape(1, -1)
            all_temps.append(temp_arr)
            T_current = temp_arr[-1].copy()
        shifted = pass_times + global_offset
        all_times.append(shifted)
        global_offset = shifted[-1]

        if bd is not None:
            activation_log.append({
                "pass_order": wp.order,
                "alive_count": int(np.sum(bd.alive_mask)),
            })

        # Interpass cooldown (skip for last pass)
        if pass_idx < len(sequence.passes) - 1:
            T_max = float(np.max(T_current))
            needs_cooldown = (
                T_max > cfg.interpass_temp_max or cfg.interpass_cooldown_s > 0
            )
            if needs_cooldown:
                cooldown_dur = max(
                    cfg.interpass_cooldown_s,
                    10.0,  # minimum 10s cooldown
                )
                cool_times = np.arange(0.0, cooldown_dur + cfg.cooldown_dt, cfg.cooldown_dt)
                cool_result = backend.solve_thermal_transient(
                    mesh=mesh,
                    material=material,
                    load_case=thermal_lc,
                    time_steps=cool_times,
                    heat_source=None,
                    initial_temperature=T_current,
                )
                if cool_result.temperature is not None:
                    cool_temp = np.asarray(cool_result.temperature)
                    if cool_temp.ndim == 1:
                        cool_temp = cool_temp.reshape(1, -1)
                    # Skip first timestep to avoid duplicate with pass end
                    if cool_temp.shape[0] > 1:
                        all_temps.append(cool_temp[1:])
                        shifted_cool = cool_times[1:] + global_offset
                        all_times.append(shifted_cool)
                        global_offset = shifted_cool[-1]
                    T_current = cool_temp[-1].copy()

                T_max_after = float(np.max(T_current))
                interpass_checks.append(InterpassCheckResult(
                    pass_order=wp.order,
                    max_temperature=T_max_after,
                    cooldown_time=cooldown_dur,
                    passed=T_max_after <= cfg.interpass_temp_max,
                ))
                if T_max_after > cfg.interpass_temp_max:
                    logger.warning(
                        "Interpass T_max=%.1f C still exceeds limit %.1f C "
                        "after %.1f s cooldown (pass %d)",
                        T_max_after, cfg.interpass_temp_max, cooldown_dur, wp.order,
                    )

    # Concatenate histories
    if all_temps:
        combined_temp = np.concatenate(all_temps, axis=0)
        combined_time = np.concatenate(all_times, axis=0)
    else:
        combined_temp = T_current.reshape(1, -1)
        combined_time = np.array([0.0])

    # Optional residual stress estimate
    residual = None
    if cfg.compute_residual_stress:
        from feaweld.solver.thermomechanical import compute_thermal_stress
        residual = compute_thermal_stress(material, T_current, T_ref=cfg.ambient_temp)

    # Build combined FEAResults
    final_fea = FEAResults(
        mesh=mesh,
        displacement=None,
        stress=StressField(values=residual) if residual is not None else None,
        strain=None,
        temperature=combined_temp,
        time_steps=combined_time,
        time_history=None,
        metadata={
            "analysis_type": "multipass_thermal",
            "n_passes": len(sequence.passes),
            "preheat_temp": cfg.preheat_temp,
            "interpass_temp_max": cfg.interpass_temp_max,
        },
    )

    return MultiPassThermalResult(
        temperature_history=combined_temp,
        time_steps=combined_time,
        interpass_checks=interpass_checks,
        per_pass_results=per_pass_results,
        final_fea_results=final_fea,
        element_activation_log=activation_log,
        residual_stress=residual,
    )
