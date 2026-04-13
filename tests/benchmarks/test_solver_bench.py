"""Performance benchmarks using pytest-benchmark.

Run with: pytest tests/benchmarks/ --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import (
    StressField,
    WeldGroupShape,
)
from feaweld.fatigue.rainflow import rainflow_count
from feaweld.postprocess.blodgett import weld_group_properties, weld_stress


# ---------------------------------------------------------------------------
# StressField.von_mises on large arrays
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_von_mises_10k_nodes(benchmark):
    """Benchmark von Mises computation on 10 000 nodes."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal((10_000, 6)) * 100.0
    sf = StressField(values=values)

    result = benchmark(lambda: sf.von_mises)

    assert result.shape == (10_000,)
    assert np.all(result >= 0)


@pytest.mark.benchmark
def test_tresca_1k_nodes(benchmark):
    """Benchmark Tresca computation on 1 000 nodes (involves eigenvalues)."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal((1_000, 6)) * 100.0
    sf = StressField(values=values)

    result = benchmark(lambda: sf.tresca)

    assert result.shape == (1_000,)


# ---------------------------------------------------------------------------
# Rainflow counting on large signal
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_rainflow_10k_points(benchmark):
    """Benchmark rainflow cycle counting on a 10 000-point signal."""
    rng = np.random.default_rng(42)
    # Create a realistic-looking stress signal: low frequency + noise
    t = np.linspace(0, 100, 10_000)
    signal = 100.0 * np.sin(2 * np.pi * 0.5 * t) + 20.0 * rng.standard_normal(10_000)

    cycles = benchmark(lambda: rainflow_count(signal))

    assert len(cycles) > 0
    # Each cycle is (range, mean, count)
    for rng_val, mean_val, count in cycles:
        assert rng_val >= 0
        assert count in (0.5, 1.0)


# ---------------------------------------------------------------------------
# Blodgett weld group properties for all shapes
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_blodgett_all_shapes(benchmark):
    """Benchmark computing weld group properties for all standard shapes."""
    shapes_with_params = [
        (WeldGroupShape.LINE, 100.0, 0.0),
        (WeldGroupShape.PARALLEL, 100.0, 50.0),
        (WeldGroupShape.C_SHAPE, 100.0, 50.0),
        (WeldGroupShape.L_SHAPE, 100.0, 50.0),
        (WeldGroupShape.BOX, 100.0, 50.0),
        (WeldGroupShape.CIRCULAR, 100.0, 0.0),
        (WeldGroupShape.I_SHAPE, 100.0, 50.0),
        (WeldGroupShape.T_SHAPE, 100.0, 50.0),
    ]

    def compute_all():
        results = []
        for shape, d, b in shapes_with_params:
            props = weld_group_properties(shape, d, b)
            stress = weld_stress(props, throat=5.0, P=10000.0, M=50000.0)
            results.append((props, stress))
        return results

    results = benchmark(compute_all)

    assert len(results) == len(shapes_with_params)
    for props, stress in results:
        assert props.A_w > 0
        assert stress["von_mises"] >= 0


@pytest.mark.benchmark
def test_principal_stress_1k_nodes(benchmark):
    """Benchmark principal stress computation on 1 000 nodes."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal((1_000, 6)) * 100.0
    sf = StressField(values=values)

    result = benchmark(lambda: sf.principal)

    assert result.shape == (1_000, 3)
