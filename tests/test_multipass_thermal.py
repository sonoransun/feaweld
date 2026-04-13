"""Tests for multi-pass welding thermal cycle solver."""

from __future__ import annotations
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.types import (
    ElementType, FEMesh, FEAResults, LoadCase, StressField, WeldPass, WeldSequence,
)
from feaweld.solver.multipass_thermal import (
    MultiPassThermalConfig,
    InterpassCheckResult,
    compute_element_centroids,
    build_element_birth_death,
    solve_multipass_thermal,
)


@pytest.fixture
def simple_mesh() -> FEMesh:
    """Simple 2-element TRI3 mesh."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return FEMesh(
        nodes=nodes, elements=elements, element_type=ElementType.TRI3,
        node_sets={"bottom": np.array([0, 1]), "top": np.array([2, 3])},
    )


@pytest.fixture
def steel() -> Material:
    return Material(
        name="TestSteel", density=7850.0,
        elastic_modulus={20.0: 210_000.0}, poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0}, ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 50.0}, specific_heat={20.0: 450.0},
        thermal_expansion={20.0: 12.0e-6},
    )


@pytest.fixture
def two_pass_sequence() -> WeldSequence:
    return WeldSequence(passes=[
        WeldPass(order=1, pass_type="root", start_time=0.0, duration=5.0),
        WeldPass(order=2, pass_type="cap", start_time=8.0, duration=5.0),
    ])


def _make_mock_backend(n_nodes: int, peak_temp: float = 300.0):
    """Create a mock backend that returns synthetic temperature histories."""
    backend = MagicMock()

    def fake_transient(mesh, material, load_case, time_steps,
                       heat_source=None, initial_temperature=None):
        n = mesh.n_nodes
        n_t = len(time_steps)
        init_t = initial_temperature if initial_temperature is not None else np.full(n, 20.0)
        temp = np.zeros((n_t, n))
        for i in range(n_t):
            frac = (i + 1) / n_t
            if heat_source is not None:
                temp[i] = init_t + frac * (peak_temp - init_t)
            else:
                # Cooldown: exponential decay toward ambient
                temp[i] = 20.0 + (init_t - 20.0) * np.exp(-0.1 * time_steps[i])
        return FEAResults(
            mesh=mesh, displacement=None, stress=None, strain=None,
            temperature=temp, time_steps=time_steps, time_history=None,
            metadata={},
        )

    backend.solve_thermal_transient = MagicMock(side_effect=fake_transient)
    return backend


def test_compute_element_centroids(simple_mesh):
    centroids = compute_element_centroids(simple_mesh)
    assert centroids.shape == (2, 3)
    # First tri: (0,0), (1,0), (1,1) -> centroid (2/3, 1/3, 0)
    np.testing.assert_allclose(centroids[0], [2.0/3, 1.0/3, 0.0], atol=1e-10)


def test_build_element_birth_death_no_groups(simple_mesh, two_pass_sequence):
    result = build_element_birth_death(simple_mesh, two_pass_sequence)
    assert result == {}


def test_multipass_thermal_config_defaults():
    cfg = MultiPassThermalConfig()
    assert cfg.preheat_temp == 20.0
    assert cfg.interpass_temp_max == 250.0
    assert cfg.steps_per_pass == 50


def test_interpass_check_result():
    r = InterpassCheckResult(pass_order=1, max_temperature=200.0, cooldown_time=10.0, passed=True)
    assert r.passed
    assert r.pass_order == 1


def test_solve_multipass_thermal_basic(simple_mesh, steel, two_pass_sequence):
    """Full orchestration with mock backend."""
    backend = _make_mock_backend(simple_mesh.n_nodes, peak_temp=200.0)
    cfg = MultiPassThermalConfig(preheat_temp=25.0, steps_per_pass=10)
    lc = LoadCase(name="thermal", loads=[], constraints=[])

    result = solve_multipass_thermal(
        backend=backend, mesh=simple_mesh, material=steel,
        sequence=two_pass_sequence, thermal_lc=lc, config=cfg,
    )

    assert result.temperature_history.shape[1] == simple_mesh.n_nodes
    assert len(result.time_steps) == result.temperature_history.shape[0]
    assert len(result.per_pass_results) == 2
    assert result.final_fea_results is not None
    assert result.final_fea_results.metadata["analysis_type"] == "multipass_thermal"


def test_interpass_cooldown_triggered(simple_mesh, steel):
    """Cooldown runs when T_max exceeds interpass limit."""
    seq = WeldSequence(passes=[
        WeldPass(order=1, pass_type="root", start_time=0.0, duration=5.0),
        WeldPass(order=2, pass_type="cap", start_time=8.0, duration=5.0),
    ])
    backend = _make_mock_backend(simple_mesh.n_nodes, peak_temp=400.0)
    cfg = MultiPassThermalConfig(
        interpass_temp_max=250.0, steps_per_pass=5,
    )
    lc = LoadCase(name="thermal", loads=[], constraints=[])

    result = solve_multipass_thermal(
        backend=backend, mesh=simple_mesh, material=steel,
        sequence=seq, thermal_lc=lc, config=cfg,
    )

    # Should have at least one interpass check
    assert len(result.interpass_checks) >= 1
    # Backend called more than twice (passes + cooldowns)
    assert backend.solve_thermal_transient.call_count >= 3


def test_preheat_temperature_applied(simple_mesh, steel, two_pass_sequence):
    """First pass should start with preheat temperature."""
    backend = _make_mock_backend(simple_mesh.n_nodes)
    cfg = MultiPassThermalConfig(preheat_temp=150.0, steps_per_pass=5)
    lc = LoadCase(name="thermal", loads=[], constraints=[])

    solve_multipass_thermal(
        backend=backend, mesh=simple_mesh, material=steel,
        sequence=two_pass_sequence, thermal_lc=lc, config=cfg,
    )

    # Check first call's initial_temperature
    first_call = backend.solve_thermal_transient.call_args_list[0]
    init_temp = first_call.kwargs.get("initial_temperature")
    if init_temp is None:
        init_temp = first_call[1].get("initial_temperature")
    assert init_temp is not None
    np.testing.assert_allclose(init_temp, 150.0)
