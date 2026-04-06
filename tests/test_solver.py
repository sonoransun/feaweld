"""Tests for the feaweld.solver subpackage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.loads import PWHTSchedule
from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    StressField,
)
from feaweld.solver.thermal import GoldakHeatSource, ElementBirthDeath
from feaweld.solver.mechanical import (
    deviatoric_stress,
    j2_return_mapping,
    linear_elastic_stress,
    von_mises,
)
from feaweld.solver.creep import norton_bailey_rate, simulate_pwht
from feaweld.solver.constitutive import (
    J2Plastic,
    LinearElastic,
    MaterialState,
    TemperatureDependent,
)
from feaweld.solver.backend import get_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_material(
    E: float = 210000.0,
    nu: float = 0.3,
    sigma_y: float = 350.0,
    hardening_modulus: float = 1000.0,
    alpha: float = 12e-6,
    k: float = 50.0,
    cp: float = 500.0,
    density: float = 7850.0,
    creep_A: float = 1e-12,
    creep_n: float = 5.0,
    creep_m: float = 0.0,
) -> Material:
    """Create a simple constant-property material for testing."""
    return Material(
        name="test_steel",
        density=density,
        elastic_modulus={20.0: E, 500.0: E},
        poisson_ratio={20.0: nu, 500.0: nu},
        yield_strength={20.0: sigma_y, 500.0: sigma_y * 0.5},
        ultimate_strength={20.0: sigma_y * 1.3, 500.0: sigma_y * 0.65},
        thermal_conductivity={20.0: k, 500.0: k},
        specific_heat={20.0: cp, 500.0: cp},
        thermal_expansion={20.0: alpha, 500.0: alpha},
        creep_A=creep_A,
        creep_n=creep_n,
        creep_m=creep_m,
        hardening_modulus=hardening_modulus,
    )


def _make_simple_mesh() -> FEMesh:
    """Create a minimal tetrahedral mesh (single element) for testing."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    elements = np.array([[0, 1, 2, 3]])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
    )


# ---------------------------------------------------------------------------
# Tests: GoldakHeatSource
# ---------------------------------------------------------------------------

class TestGoldakHeatSource:
    """Tests for the Goldak double-ellipsoid heat source model."""

    def test_evaluate_at_centre_t0(self):
        """At t=0 the source centre is at start_position; peak heat flux there."""
        source = GoldakHeatSource(
            power=1000.0,
            a_f=5.0,
            a_r=10.0,
            b=5.0,
            c=5.0,
            f_f=0.6,
            f_r=1.4,
            travel_speed=5.0,
            start_position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
        )
        # Evaluate exactly at the source centre
        q = source.evaluate(
            np.array([0.0]), np.array([0.0]), np.array([0.0]), t=0.0
        )
        assert q.shape == (1,)
        assert q[0] > 0.0, "Heat flux at centre must be positive"

        # The front formula at origin: Q_f = norm_const * f_f * P / (a_f*b*c)
        pi = np.pi
        norm_const = 6.0 * np.sqrt(3.0) / (pi * np.sqrt(pi))
        expected = norm_const * 0.6 * 1000.0 / (5.0 * 5.0 * 5.0)
        np.testing.assert_allclose(q[0], expected, rtol=1e-10)

    def test_evaluate_symmetry(self):
        """Heat flux should be symmetric in y (width direction)."""
        source = GoldakHeatSource(
            power=1000.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0,
        )
        q_pos = source.evaluate(
            np.array([0.0]), np.array([2.0]), np.array([0.0]), t=0.0
        )
        q_neg = source.evaluate(
            np.array([0.0]), np.array([-2.0]), np.array([0.0]), t=0.0
        )
        np.testing.assert_allclose(q_pos, q_neg, rtol=1e-12)

    def test_evaluate_decays_with_distance(self):
        """Heat flux should decay with distance from centre."""
        source = GoldakHeatSource(
            power=1000.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0,
        )
        q_near = source.evaluate(
            np.array([1.0]), np.array([0.0]), np.array([0.0]), t=0.0
        )
        q_far = source.evaluate(
            np.array([10.0]), np.array([0.0]), np.array([0.0]), t=0.0
        )
        assert q_near[0] > q_far[0]

    def test_source_moves_with_time(self):
        """The peak should follow the torch at travel_speed * t."""
        source = GoldakHeatSource(
            power=1000.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0,
            travel_speed=10.0,
            start_position=np.zeros(3),
            direction=np.array([1.0, 0.0, 0.0]),
        )
        # At t=1, centre should be at x=10
        q_at_new_centre = source.evaluate(
            np.array([10.0]), np.array([0.0]), np.array([0.0]), t=1.0
        )
        q_at_old_centre = source.evaluate(
            np.array([0.0]), np.array([0.0]), np.array([0.0]), t=1.0
        )
        # Peak should now be at x=10
        assert q_at_new_centre[0] > q_at_old_centre[0]

    def test_vectorised_evaluation(self):
        """Should handle arrays of coordinates."""
        source = GoldakHeatSource(
            power=1000.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0,
        )
        x = np.linspace(-20, 20, 50)
        y = np.zeros(50)
        z = np.zeros(50)
        q = source.evaluate(x, y, z, t=0.0)
        assert q.shape == (50,)
        assert np.all(q >= 0.0)

    def test_total_energy_rate(self):
        source = GoldakHeatSource(power=1500.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0)
        assert source.total_energy_rate() == 1500.0


# ---------------------------------------------------------------------------
# Tests: ElementBirthDeath
# ---------------------------------------------------------------------------

class TestElementBirthDeath:
    def test_initial_state_all_dead(self):
        centroids = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        ebd = ElementBirthDeath(
            element_centroids=centroids,
            weld_element_ids=np.array([0, 1, 2]),
        )
        assert len(ebd.alive_element_ids) == 0
        assert len(ebd.dead_element_ids) == 3

    def test_activation_behind_torch(self):
        centroids = np.array([
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        ebd = ElementBirthDeath(
            element_centroids=centroids,
            weld_element_ids=np.array([0, 1, 2]),
            activation_distance=2.0,
        )
        # Torch at x=6, travelling in +x: element at x=1 and x=5 are behind
        newly = ebd.update(
            torch_position=np.array([6.0, 0.0, 0.0]),
            travel_direction=np.array([1.0, 0.0, 0.0]),
        )
        assert 0 in newly  # x=1 is behind
        assert 1 in newly  # x=5 is behind (within activation_distance)
        assert 2 not in newly  # x=10 is ahead

    def test_reset(self):
        centroids = np.array([[1.0, 0.0, 0.0]])
        ebd = ElementBirthDeath(
            element_centroids=centroids,
            weld_element_ids=np.array([0]),
        )
        ebd.activate_all()
        assert len(ebd.alive_element_ids) == 1
        ebd.reset()
        assert len(ebd.alive_element_ids) == 0


# ---------------------------------------------------------------------------
# Tests: linear_elastic_stress
# ---------------------------------------------------------------------------

class TestLinearElasticStress:
    def test_uniaxial_tension(self):
        """Uniaxial strain: eps_xx != 0, all others zero."""
        mat = _make_material()
        C = mat.elasticity_tensor_3d(20.0)
        eps_xx = 0.001  # 0.1% strain
        strain = np.array([eps_xx, 0, 0, 0, 0, 0])
        stress = linear_elastic_stress(strain, C)

        lam = mat.lame_lambda(20.0)
        mu = mat.lame_mu(20.0)
        expected_xx = (lam + 2 * mu) * eps_xx
        expected_yy = lam * eps_xx
        expected_zz = lam * eps_xx

        np.testing.assert_allclose(stress[0], expected_xx, rtol=1e-10)
        np.testing.assert_allclose(stress[1], expected_yy, rtol=1e-10)
        np.testing.assert_allclose(stress[2], expected_zz, rtol=1e-10)
        np.testing.assert_allclose(stress[3:], 0.0, atol=1e-15)

    def test_pure_shear(self):
        """Pure shear strain: gamma_xy != 0, all others zero."""
        mat = _make_material()
        C = mat.elasticity_tensor_3d(20.0)
        gamma_xy = 0.001
        strain = np.array([0, 0, 0, gamma_xy, 0, 0])
        stress = linear_elastic_stress(strain, C)

        mu = mat.lame_mu(20.0)
        expected_tau = mu * gamma_xy

        np.testing.assert_allclose(stress[3], expected_tau, rtol=1e-10)
        np.testing.assert_allclose(stress[:3], 0.0, atol=1e-15)

    def test_batch_evaluation(self):
        """Multiple strain states at once."""
        mat = _make_material()
        C = mat.elasticity_tensor_3d(20.0)
        strains = np.array([
            [0.001, 0, 0, 0, 0, 0],
            [0, 0.001, 0, 0, 0, 0],
        ])
        stresses = linear_elastic_stress(strains, C)
        assert stresses.shape == (2, 6)
        # Symmetry: stress[0][0] should equal stress[1][1]
        np.testing.assert_allclose(stresses[0, 0], stresses[1, 1], rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: j2_return_mapping
# ---------------------------------------------------------------------------

class TestJ2ReturnMapping:
    def test_elastic_case_no_yielding(self):
        """Small strain within elastic range: no plastic correction."""
        mat = _make_material(sigma_y=350.0)
        C = mat.elasticity_tensor_3d(20.0)

        # Very small strain -> well below yield
        eps = 1e-5
        strain = np.array([eps, 0, 0, 0, 0, 0])
        eps_p_n = np.zeros(6)

        stress, eps_p_new, d_gamma = j2_return_mapping(strain, eps_p_n, mat, 20.0)

        # Should be purely elastic
        np.testing.assert_allclose(d_gamma, 0.0, atol=1e-20)
        np.testing.assert_allclose(eps_p_new, np.zeros(6), atol=1e-20)

        # Stress should match linear elastic
        expected = C @ strain
        np.testing.assert_allclose(stress, expected, rtol=1e-10)

    def test_plastic_case_yielding_occurs(self):
        """Large strain exceeding yield: plastic correction should occur."""
        mat = _make_material(sigma_y=350.0, hardening_modulus=1000.0, E=210000.0)

        # Large uniaxial strain to push well past yield
        # Yield strain ~ 350 / 210000 ~ 0.00167
        eps = 0.01  # well past yield
        strain = np.array([eps, 0, 0, 0, 0, 0])
        eps_p_n = np.zeros(6)

        stress, eps_p_new, d_gamma = j2_return_mapping(strain, eps_p_n, mat, 20.0)

        # Plastic increment should be positive
        assert d_gamma > 0.0, "Plastic correction should have occurred"

        # Plastic strain should be non-zero
        assert np.linalg.norm(eps_p_new) > 0.0

        # Von Mises stress should be close to (but slightly above due to
        # hardening) the yield strength
        s = stress.copy()
        hydro = (s[0] + s[1] + s[2]) / 3.0
        s[:3] -= hydro
        s_sq = s[0]**2 + s[1]**2 + s[2]**2 + 2*(s[3]**2 + s[4]**2 + s[5]**2)
        vm = np.sqrt(1.5 * s_sq)

        # Von Mises should be reduced from the trial elastic value
        # and plastic strain should absorb the difference
        trial_stress = mat.elasticity_tensor_3d(20.0) @ strain
        s_trial = trial_stress.copy()
        hydro_t = (s_trial[0] + s_trial[1] + s_trial[2]) / 3.0
        s_trial[:3] -= hydro_t
        s_sq_t = s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 + 2*(s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        vm_trial = np.sqrt(1.5 * s_sq_t)
        assert vm < vm_trial, "Returned stress should be less than elastic trial"

    def test_plastic_strain_is_deviatoric(self):
        """Plastic strain increment should be trace-free (incompressible)."""
        mat = _make_material(sigma_y=100.0, hardening_modulus=500.0)

        strain = np.array([0.01, -0.003, 0.005, 0.002, 0, 0])
        eps_p_n = np.zeros(6)

        stress, eps_p_new, d_gamma = j2_return_mapping(strain, eps_p_n, mat, 20.0)

        if d_gamma > 0:
            # Plastic strain trace should be zero (volumetrically incompressible)
            trace = eps_p_new[0] + eps_p_new[1] + eps_p_new[2]
            np.testing.assert_allclose(trace, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: von_mises / deviatoric_stress
# ---------------------------------------------------------------------------

class TestVonMises:
    def test_uniaxial_stress(self):
        """For uniaxial stress sigma_xx, von Mises = |sigma_xx|."""
        sigma = np.array([100.0, 0, 0, 0, 0, 0])
        vm = von_mises(sigma)
        np.testing.assert_allclose(vm, 100.0, rtol=1e-10)

    def test_hydrostatic_stress(self):
        """Pure hydrostatic stress has zero von Mises."""
        sigma = np.array([100.0, 100.0, 100.0, 0, 0, 0])
        vm = von_mises(sigma)
        np.testing.assert_allclose(vm, 0.0, atol=1e-10)

    def test_pure_shear(self):
        """Pure shear tau_xy: von Mises = sqrt(3) * |tau|."""
        tau = 100.0
        sigma = np.array([0, 0, 0, tau, 0, 0])
        vm = von_mises(sigma)
        np.testing.assert_allclose(vm, np.sqrt(3) * tau, rtol=1e-10)


class TestDeviatoricStress:
    def test_deviatoric_trace_is_zero(self):
        sigma = np.array([100.0, 200.0, 300.0, 50.0, 30.0, 20.0])
        s = deviatoric_stress(sigma)
        trace = s[0] + s[1] + s[2]
        np.testing.assert_allclose(trace, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: norton_bailey_rate
# ---------------------------------------------------------------------------

class TestNortonBaileyRate:
    def test_zero_stress(self):
        """Zero stress should give zero creep rate."""
        stress = np.zeros(6)
        rate = norton_bailey_rate(stress, time=3600.0, A=1e-12, n=5.0, m=0.0)
        np.testing.assert_allclose(rate, np.zeros(6), atol=1e-30)

    def test_uniaxial_stress(self):
        """Check rate magnitude for uniaxial stress."""
        sigma_val = 100.0  # MPa
        stress = np.array([sigma_val, 0, 0, 0, 0, 0])
        A = 1e-12
        n = 5.0
        m = 0.0
        rate = norton_bailey_rate(stress, time=1.0, A=A, n=n, m=m)

        # Expected equivalent rate: A * sigma_vm^n * t^m
        sigma_vm = sigma_val  # uniaxial -> vm = |sigma|
        expected_eq = A * sigma_vm ** n * 1.0 ** m
        # Rate should be in the direction of deviatoric stress
        # For uniaxial: s = [2/3*sigma, -1/3*sigma, -1/3*sigma, 0, 0, 0]
        # eps_dot_eq = sqrt(2/3 * eps_dot : eps_dot)
        # The returned rate should have magnitude consistent with expected_eq
        # eps_dot_11 = (3/2) * eps_dot_eq / sigma_vm * s_11
        s_11 = 2.0 / 3.0 * sigma_val
        expected_rate_11 = 1.5 * expected_eq / sigma_vm * s_11
        np.testing.assert_allclose(rate[0], expected_rate_11, rtol=1e-10)

    def test_time_dependence(self):
        """Non-zero m exponent should produce time-dependent rate."""
        stress = np.array([100.0, 0, 0, 0, 0, 0])
        rate_t1 = norton_bailey_rate(stress, time=1.0, A=1e-12, n=5.0, m=0.5)
        rate_t4 = norton_bailey_rate(stress, time=4.0, A=1e-12, n=5.0, m=0.5)
        # t^0.5: rate at t=4 should be 2x rate at t=1
        np.testing.assert_allclose(rate_t4, rate_t1 * 2.0, rtol=1e-10)

    def test_batch_evaluation(self):
        """Should handle (n, 6) stress arrays."""
        stresses = np.array([
            [100.0, 0, 0, 0, 0, 0],
            [200.0, 0, 0, 0, 0, 0],
        ])
        rates = norton_bailey_rate(stresses, time=1.0, A=1e-12, n=5.0, m=0.0)
        assert rates.shape == (2, 6)
        # Second stress is 2x first, rate should be 2^5 = 32x
        ratio = rates[1, 0] / rates[0, 0]
        np.testing.assert_allclose(ratio, 32.0, rtol=1e-10)

    def test_creep_is_deviatoric(self):
        """Creep strain rate should be trace-free."""
        stress = np.array([100.0, 50.0, -30.0, 20.0, 10.0, 5.0])
        rate = norton_bailey_rate(stress, time=100.0, A=1e-12, n=5.0, m=0.0)
        trace = rate[0] + rate[1] + rate[2]
        np.testing.assert_allclose(trace, 0.0, atol=1e-20)


# ---------------------------------------------------------------------------
# Tests: PWHT simulation
# ---------------------------------------------------------------------------

class TestSimulatePWHT:
    def test_pwht_temperature_profile(self):
        """Check PWHT schedule generates reasonable temperature profile."""
        schedule = PWHTSchedule(
            heating_rate=100.0,
            holding_temperature=620.0,
            holding_time=2.0,
            cooling_rate=100.0,
        )
        times, temps = schedule.temperature_profile(dt=60.0)

        # Should start at ambient
        np.testing.assert_allclose(temps[0], 20.0, atol=1.0)

        # Should reach holding temperature
        assert np.any(np.abs(temps - 620.0) < 5.0)

        # Should come back down
        assert temps[-1] < 100.0

    def test_pwht_reduces_stress(self):
        """PWHT should reduce residual stress (at least for non-zero creep params)."""
        mat = _make_material(creep_A=1e-10, creep_n=3.0, creep_m=0.0)
        mesh = _make_simple_mesh()

        # Create an artificial high residual stress state
        n_pts = 4
        stress_vals = np.zeros((n_pts, 6))
        stress_vals[:, 0] = 300.0  # 300 MPa in x-direction

        initial_results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        schedule = PWHTSchedule(
            heating_rate=200.0,
            holding_temperature=620.0,
            holding_time=2.0,
            cooling_rate=200.0,
        )

        relaxed = simulate_pwht(initial_results, mat, schedule, dt=60.0)

        assert relaxed.stress is not None
        # Von Mises should be lower after PWHT
        vm_initial = initial_results.stress.von_mises
        vm_relaxed = relaxed.stress.von_mises
        assert np.all(vm_relaxed <= vm_initial + 1e-6)
        # Should have actually reduced
        assert np.mean(vm_relaxed) < np.mean(vm_initial)

    def test_pwht_no_stress_raises(self):
        """simulate_pwht should raise if results have no stress field."""
        mat = _make_material()
        mesh = _make_simple_mesh()
        results = FEAResults(mesh=mesh)
        schedule = PWHTSchedule(
            heating_rate=100.0, holding_temperature=620.0,
            holding_time=1.0, cooling_rate=100.0,
        )
        with pytest.raises(ValueError, match="stress field"):
            simulate_pwht(results, mat, schedule)


# ---------------------------------------------------------------------------
# Tests: Constitutive model wrappers
# ---------------------------------------------------------------------------

class TestConstitutiveModels:
    def test_linear_elastic_model(self):
        mat = _make_material()
        model = LinearElastic(mat, temperature=20.0)
        state = MaterialState()
        strain = np.array([0.001, 0, 0, 0, 0, 0])
        stress, new_state = model.stress(strain, state)

        C = mat.elasticity_tensor_3d(20.0)
        expected = C @ strain
        np.testing.assert_allclose(stress, expected, rtol=1e-10)

        # Tangent should be the elasticity tensor
        tangent = model.tangent(strain, state)
        np.testing.assert_allclose(tangent, C, rtol=1e-10)

    def test_j2_plastic_model_elastic_range(self):
        mat = _make_material(sigma_y=350.0)
        model = J2Plastic(mat, temperature=20.0)
        state = MaterialState()
        strain = np.array([1e-5, 0, 0, 0, 0, 0])
        stress, new_state = model.stress(strain, state)
        # Should be elastic
        np.testing.assert_allclose(new_state.equiv_plastic_strain, 0.0, atol=1e-20)

    def test_j2_plastic_model_plastic_range(self):
        mat = _make_material(sigma_y=100.0, hardening_modulus=500.0)
        model = J2Plastic(mat, temperature=20.0)
        state = MaterialState()
        strain = np.array([0.01, 0, 0, 0, 0, 0])
        stress, new_state = model.stress(strain, state)
        assert new_state.equiv_plastic_strain > 0.0

    def test_temperature_dependent_wrapper(self):
        mat = _make_material()
        base = LinearElastic(mat, temperature=20.0)
        model = TemperatureDependent(base, mat)

        state = MaterialState(temperature=300.0)
        strain = np.array([0.001, 0, 0, 0, 0, 0])
        stress, _ = model.stress(strain, state)

        # Should use properties at 300 C
        C300 = mat.elasticity_tensor_3d(300.0)
        expected = C300 @ strain
        np.testing.assert_allclose(stress, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: get_backend
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_get_backend_auto_no_backends(self):
        """When no backends are available, should raise ImportError."""
        with patch.dict("sys.modules", {"dolfinx": None}):
            # Also ensure CalculiX is not found
            with patch(
                "feaweld.solver.calculix_backend.CalculiXBackend.__init__",
                side_effect=ImportError("no ccx"),
            ):
                with pytest.raises(ImportError):
                    get_backend("auto")

    def test_get_backend_calculix(self):
        """Should return CalculiXBackend when requested (import succeeds)."""
        try:
            backend = get_backend("calculix")
            assert backend is not None
            assert type(backend).__name__ == "CalculiXBackend"
        except ImportError:
            pytest.skip("CalculiX backend not available")

    def test_get_backend_fenics(self):
        """Should return FEniCSBackend when requested (import succeeds)."""
        try:
            backend = get_backend("fenics")
            assert backend is not None
            assert type(backend).__name__ == "FEniCSBackend"
        except ImportError:
            pytest.skip("FEniCS backend not available")


# ---------------------------------------------------------------------------
# Tests: .inp generation (CalculiX)
# ---------------------------------------------------------------------------

class TestInpGeneration:
    def test_generate_static_inp(self, tmp_path):
        """Verify that a valid .inp file is generated for static analysis."""
        from feaweld.solver.calculix_backend import generate_inp

        mesh = _make_simple_mesh()
        mat = _make_material()
        lc = LoadCase(name="test")

        inp_path = generate_inp(
            mesh=mesh, material=mat, load_case=lc,
            path=tmp_path / "test.inp", analysis="static",
        )

        assert inp_path.exists()
        content = inp_path.read_text()

        assert "*NODE" in content
        assert "*ELEMENT" in content
        assert "*MATERIAL" in content
        assert "*ELASTIC" in content
        assert "*STEP" in content
        assert "*STATIC" in content
        assert "*END STEP" in content

    def test_generate_thermal_inp(self, tmp_path):
        """Verify .inp for thermal steady-state analysis."""
        from feaweld.solver.calculix_backend import generate_inp

        mesh = _make_simple_mesh()
        mat = _make_material()
        lc = LoadCase(name="thermal_test")

        inp_path = generate_inp(
            mesh=mesh, material=mat, load_case=lc,
            path=tmp_path / "thermal.inp", analysis="thermal_steady",
        )

        content = inp_path.read_text()
        assert "*HEAT TRANSFER, STEADY STATE" in content
        assert "*CONDUCTIVITY" in content


# ---------------------------------------------------------------------------
# Tests: .frd parser
# ---------------------------------------------------------------------------

class TestFrdParser:
    def test_parse_empty_file(self, tmp_path):
        """Parser should return empty dict for a file with no result blocks."""
        from feaweld.solver.calculix_backend import parse_frd

        frd = tmp_path / "empty.frd"
        frd.write_text("    1C\n    9999\n")  # minimal header and footer
        result = parse_frd(frd)
        assert isinstance(result, dict)

    def test_parse_missing_file(self, tmp_path):
        """Parser should raise FileNotFoundError for missing file."""
        from feaweld.solver.calculix_backend import parse_frd

        with pytest.raises(FileNotFoundError):
            parse_frd(tmp_path / "nonexistent.frd")


# ---------------------------------------------------------------------------
# Tests: compute_thermal_stress
# ---------------------------------------------------------------------------

class TestComputeThermalStress:
    def test_uniform_temperature_no_stress(self):
        """At reference temperature, thermal stress should be zero."""
        from feaweld.solver.thermomechanical import compute_thermal_stress

        mat = _make_material()
        temps = np.full(5, 20.0)
        stress = compute_thermal_stress(mat, temps, T_ref=20.0)
        np.testing.assert_allclose(stress, np.zeros((5, 6)), atol=1e-10)

    def test_heated_produces_compressive_stress(self):
        """Fully constrained heated body should develop compressive stress."""
        from feaweld.solver.thermomechanical import compute_thermal_stress

        mat = _make_material(alpha=12e-6)
        temps = np.full(3, 100.0)  # 80 C above reference
        stress = compute_thermal_stress(mat, temps, T_ref=20.0)
        # sigma_th = -C : eps_th  with eps_th > 0, so sigma_xx < 0 (compressive)
        assert np.all(stress[:, 0] < 0)
        assert np.all(stress[:, 1] < 0)
        assert np.all(stress[:, 2] < 0)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_j2_zero_strain_elastic(self):
        """Zero strain increment should return zero stress and no plasticity."""
        from feaweld.solver.mechanical import j2_return_mapping

        mat = _make_material()
        sigma, eps_p, dgamma = j2_return_mapping(
            np.zeros(6), np.zeros(6), mat, 20.0,
        )
        np.testing.assert_allclose(sigma, 0.0, atol=1e-12)
        np.testing.assert_allclose(eps_p, 0.0, atol=1e-12)
        assert dgamma == pytest.approx(0.0)

    def test_single_point_material_interpolation(self):
        """Material with a single temperature data point should return constant."""
        from feaweld.core.materials import Material

        mat = Material(
            name="test_single",
            density=7850.0,
            elastic_modulus={20.0: 200000.0},
            poisson_ratio={20.0: 0.3},
            yield_strength={20.0: 250.0},
            ultimate_strength={20.0: 400.0},
            thermal_conductivity={20.0: 50.0},
            specific_heat={20.0: 500.0},
            thermal_expansion={20.0: 12e-6},
        )
        # Should return same value at any temperature
        assert mat.E(20.0) == pytest.approx(200000.0)
        assert mat.E(500.0) == pytest.approx(200000.0)
        assert mat.nu(100.0) == pytest.approx(0.3)
