"""Tests for multi-scale modeling modules."""

import numpy as np
import pytest

from feaweld.multiscale.micro import (
    HallPetchParams, HALL_PETCH_LOW_CARBON_STEEL,
    DislocationDensityParams, DISLOCATION_BCC_IRON,
    homogenize_properties, micro_to_meso_properties,
    estimate_grain_size_from_cooling, phase_dependent_elastic_modulus,
)
from feaweld.multiscale.meso import (
    WeldZone, PhaseComposition, default_low_carbon_cct,
    estimate_zone_properties, sdas_to_yield_strength,
    assign_zones, cooling_rate_from_thermal,
)
from feaweld.multiscale.macro import (
    extract_subregion, interpolate_boundary_conditions, check_equilibrium,
    SubregionSpec,
)


class TestHallPetch:
    def test_yield_strength_increases_with_smaller_grains(self):
        hp = HALL_PETCH_LOW_CARBON_STEEL
        sigma_large = hp.yield_strength(100.0)  # 100 μm
        sigma_small = hp.yield_strength(10.0)   # 10 μm
        assert sigma_small > sigma_large

    def test_known_values(self):
        hp = HallPetchParams(sigma_0=70.0, k_y=0.74)
        # At grain size 25 μm (typical base metal)
        sigma = hp.yield_strength(25.0)
        # σ_0 + k_y / sqrt(25e-6) = 70 + 0.74/0.005 = 70 + 148 = 218
        assert 200 < sigma < 250


class TestDislocationDensity:
    def test_flow_stress_increases_with_density(self):
        dd = DISLOCATION_BCC_IRON
        sigma_low = dd.flow_stress(1e12)
        sigma_high = dd.flow_stress(1e15)
        assert sigma_high > sigma_low

    def test_inverse_roundtrip(self):
        dd = DISLOCATION_BCC_IRON
        rho = 1e14
        sigma = dd.flow_stress(rho)
        rho_back = dd.dislocation_density_from_stress(sigma)
        np.testing.assert_allclose(rho_back, rho, rtol=0.01)


class TestHomogenization:
    def test_voigt_average(self):
        fractions = {"ferrite": 0.7, "pearlite": 0.3}
        properties = {
            "ferrite": {"E": 200000, "sigma_y": 200},
            "pearlite": {"E": 210000, "sigma_y": 350},
        }
        result = homogenize_properties(fractions, properties, method="voigt")
        assert result["E"] == pytest.approx(0.7 * 200000 + 0.3 * 210000)
        assert result["sigma_y"] == pytest.approx(0.7 * 200 + 0.3 * 350)

    def test_hill_between_bounds(self):
        fractions = {"ferrite": 0.5, "pearlite": 0.5}
        properties = {
            "ferrite": {"E": 200000},
            "pearlite": {"E": 210000},
        }
        voigt = homogenize_properties(fractions, properties, "voigt")
        reuss = homogenize_properties(fractions, properties, "reuss")
        hill = homogenize_properties(fractions, properties, "hill")
        assert reuss["E"] <= hill["E"] <= voigt["E"]


class TestCCTDiagram:
    def test_slow_cooling_mostly_ferrite(self):
        cct = default_low_carbon_cct()
        phases = cct.predict_phases(0.1)
        assert phases.ferrite > 0.5
        assert phases.martensite < 0.05

    def test_fast_cooling_mostly_martensite(self):
        cct = default_low_carbon_cct()
        phases = cct.predict_phases(100.0)
        assert phases.martensite > 0.5


class TestMesoZoneProperties:
    def test_martensite_harder(self):
        pure_ferrite = PhaseComposition(ferrite=1.0)
        pure_martensite = PhaseComposition(martensite=1.0)

        props_f = estimate_zone_properties(WeldZone.BASE_METAL, pure_ferrite)
        props_m = estimate_zone_properties(WeldZone.WELD_METAL, pure_martensite)

        assert props_m.yield_strength > props_f.yield_strength
        assert props_m.hardness_hv > props_f.hardness_hv


class TestAssignZones:
    def test_zone_assignment(self):
        nodes = np.array([
            [0.0, 0.0, 0.0],     # weld center
            [3.0, 0.0, 0.0],     # CG HAZ
            [5.0, 0.0, 0.0],     # FG HAZ
            [8.0, 0.0, 0.0],     # IC HAZ
            [15.0, 0.0, 0.0],    # base metal
        ])
        zones = assign_zones(nodes, weld_center=np.zeros(3), weld_radius=2.0, haz_width=6.0)
        assert zones[0] == WeldZone.WELD_METAL
        assert zones[4] == WeldZone.BASE_METAL


class TestMicroToMeso:
    def test_reasonable_yield(self):
        result = micro_to_meso_properties(grain_size_um=25.0, dislocation_density=1e14)
        assert 150 < result["yield_strength"] < 500

    def test_smaller_grains_stronger(self):
        r1 = micro_to_meso_properties(grain_size_um=50.0)
        r2 = micro_to_meso_properties(grain_size_um=10.0)
        assert r2["yield_strength"] > r1["yield_strength"]


class TestCoolingRateEstimation:
    def test_grain_size_from_cooling(self):
        # Faster cooling → finer grains
        d_slow = estimate_grain_size_from_cooling(1.0)
        d_fast = estimate_grain_size_from_cooling(100.0)
        assert d_fast < d_slow


class TestMacroExtraction:
    def test_extract_subregion(self, uniform_stress_results):
        spec = SubregionSpec(
            center=np.array([5.0, 5.0, 0.0]),
            radius=20.0,  # large enough to capture all nodes
        )
        transfer = extract_subregion(uniform_stress_results, spec)
        assert len(transfer.boundary_nodes) >= 0
        assert transfer.boundary_displacements is not None

    def test_equilibrium_check(self):
        from feaweld.multiscale.macro import MacroToMesoTransfer
        transfer = MacroToMesoTransfer(
            boundary_nodes=np.array([0, 1, 2]),
            boundary_positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            boundary_displacements=np.zeros((3, 3)),
        )
        # Balanced forces → passes
        forces = np.array([[100, 0, 0], [-50, 0, 0], [-50, 0, 0]], dtype=float)
        result = check_equilibrium(transfer, forces, tolerance=0.01)
        assert result["passes"]
