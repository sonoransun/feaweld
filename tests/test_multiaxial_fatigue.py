"""Tests for multi-axial fatigue criteria."""

from __future__ import annotations

import numpy as np

from feaweld.postprocess.multiaxial import (
    crossland_criterion,
    dang_van_criterion,
    fatemi_socie_criterion,
    fibonacci_sphere_grid,
    findley_criterion,
    mcdiarmid_criterion,
    resolve_on_plane,
    sines_criterion,
)


def test_fibonacci_grid_unit_vectors() -> None:
    pts = fibonacci_sphere_grid(100)
    assert pts.shape == (100, 3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_resolve_on_plane_uniaxial_xx() -> None:
    stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    normal_stress, shear_vec = resolve_on_plane(stress, np.array([1.0, 0.0, 0.0]))
    assert normal_stress.shape == (1,)
    np.testing.assert_allclose(normal_stress[0], 100.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(shear_vec[0]), 0.0, atol=1e-12)

    n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    normal_stress, shear_vec = resolve_on_plane(stress, n)
    np.testing.assert_allclose(normal_stress[0], 50.0, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(shear_vec[0]), 50.0, atol=1e-12
    )


def test_findley_uniaxial() -> None:
    # Reversed uniaxial: sigma_xx oscillating +/- 100 so shear amplitude is
    # nontrivial on tilted planes.
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    sxx = 100.0 * np.cos(t)
    stress = np.zeros((len(t), 6))
    stress[:, 0] = sxx
    res = findley_criterion(stress, k_param=0.3, n_planes=400)
    assert np.isfinite(res.damage_parameter)
    assert res.damage_parameter > 0.0
    # Critical plane normal should have meaningful x-component (tilted ~45°).
    # For uniaxial tension, the plane of max shear has n at 45° to x-axis.
    nvec = res.critical_plane_normal
    np.testing.assert_allclose(np.linalg.norm(nvec), 1.0, atol=1e-10)
    # |n_x| should be near sqrt(1/2) ~ 0.707 and away from 0 and 1.
    assert 0.35 < abs(nvec[0]) < 0.9


def test_dang_van_pure_shear() -> None:
    # Pure shear tau_xy oscillating +/- 100; hydrostatic = 0.
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    stress = np.zeros((len(t), 6))
    stress[:, 3] = 100.0 * np.cos(t)  # tau_xy
    res = dang_van_criterion(stress, alpha_dv=0.3)
    assert abs(res.hydrostatic_mean) < 1e-10
    # Damage ~ tau_meso max which for pure shear equals shear amplitude.
    assert abs(res.damage_parameter - 100.0) < 1.0


def test_sines_adds_mean_stress_component() -> None:
    # Use oscillating shear (deviatoric only) and add an isotropic mean
    # stress via sigma_xx=sigma_yy=sigma_zz=const. Pure hydrostatic contributes
    # nothing to J2 so J2_a is identical between the two cases.
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    stress_a = np.zeros((len(t), 6))
    stress_a[:, 3] = 100.0 * np.cos(t)  # oscillating tau_xy, zero hydrostatic
    stress_b = stress_a.copy()
    stress_b[:, 0] += 50.0
    stress_b[:, 1] += 50.0
    stress_b[:, 2] += 50.0  # isotropic mean -> hydrostatic = 50, J2 unchanged
    res_a = sines_criterion(stress_a, alpha_s=0.2)
    res_b = sines_criterion(stress_b, alpha_s=0.2)
    assert abs(res_a.shear_amplitude - res_b.shear_amplitude) < 1e-8
    assert abs(res_a.damage_parameter - res_b.damage_parameter) > 1.0


def test_crossland_and_sines_agree_when_mean_equals_max() -> None:
    # Constant hydrostatic stress -> mean = max.
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    stress = np.zeros((len(t), 6))
    # Oscillating shear keeps hydrostatic = 0 constant (mean == max).
    stress[:, 3] = 80.0 * np.cos(t)
    res_c = crossland_criterion(stress, alpha_c=0.3)
    res_s = sines_criterion(stress, alpha_s=0.3)
    np.testing.assert_allclose(
        res_c.damage_parameter, res_s.damage_parameter, atol=1e-10
    )


def test_fatemi_socie_returns_valid_plane() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    stress = np.zeros((len(t), 6))
    stress[:, 0] = 150.0 * np.cos(t)
    res = fatemi_socie_criterion(stress, k_fs=0.5, sigma_y=250.0, n_planes=200)
    n = res.critical_plane_normal
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-10)
    assert np.isfinite(res.damage_parameter)


def test_mcdiarmid_formula() -> None:
    # Static uniaxial tension of 100 in x. Max shear plane at 45° has
    # tau_a = 0 (static, no amplitude) and sigma_n_max = 50 at the
    # 45° plane but max normal = 100 on plane [1,0,0].
    # The criterion max over planes of tau_a / t_a_limit + sigma_n/(2*sigma_u)
    # with tau_a = 0 becomes max sigma_n / (2 sigma_u) = 100 / (2*400) = 0.125.
    stress = np.zeros((1, 6))
    stress[0, 0] = 100.0
    res = mcdiarmid_criterion(
        stress, t_a_limit=200.0, sigma_u=400.0, n_planes=600
    )
    expected = 100.0 / (2.0 * 400.0)
    assert abs(res.damage_parameter - expected) / expected < 0.01
