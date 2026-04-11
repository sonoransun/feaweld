"""Tests for Moulinec-Suquet FFT homogenization."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.multiscale.fft_homogenization import (
    MAX_VOXELS_PER_SIDE,
    fft_homogenize,
    isotropic_moduli_from_stiffness,
    isotropic_stiffness,
    make_sphere_rve,
    mori_tanaka_two_phase,
)


def _bulk_shear_from_E_nu(E: float, nu: float) -> tuple[float, float]:
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return K, G


def test_homogeneous_rve_returns_input_stiffness():
    K, G = _bulk_shear_from_E_nu(210000.0, 0.3)
    C_phase = isotropic_stiffness(K, G)

    phase_map = np.zeros((16, 16, 16), dtype=np.int_)
    C_eff = fft_homogenize(phase_map, {0: C_phase}, tol=1e-8, max_iter=50)

    rel_err = np.linalg.norm(C_eff - C_phase) / np.linalg.norm(C_phase)
    assert rel_err < 1e-2, f"homogeneous RVE deviated by {rel_err:.2e}"


@pytest.mark.slow
def test_sphere_in_matrix_matches_mori_tanaka():
    K_m, G_m = _bulk_shear_from_E_nu(70000.0, 0.33)
    K_i, G_i = _bulk_shear_from_E_nu(400000.0, 0.2)
    C_m = isotropic_stiffness(K_m, G_m)
    C_i = isotropic_stiffness(K_i, G_i)

    Nx = 64
    radius_frac = 0.289
    phase_map = make_sphere_rve(Nx, radius_frac=radius_frac)
    f_i = float(np.mean(phase_map == 1))
    assert 0.08 < f_i < 0.12, f"unexpected volume fraction {f_i}"

    C_eff = fft_homogenize(
        phase_map,
        {0: C_m, 1: C_i},
        tol=1e-4,
        max_iter=200,
    )

    mod_fft = isotropic_moduli_from_stiffness(C_eff)
    C_mt = mori_tanaka_two_phase(C_m, C_i, f_i)
    mod_mt = isotropic_moduli_from_stiffness(C_mt)

    rel_k = abs(mod_fft.bulk - mod_mt.bulk) / mod_mt.bulk
    rel_g = abs(mod_fft.shear - mod_mt.shear) / mod_mt.shear
    assert rel_k < 0.05, f"bulk modulus off by {rel_k:.2%}"
    assert rel_g < 0.05, f"shear modulus off by {rel_g:.2%}"


def test_rve_size_cap():
    big = np.zeros((MAX_VOXELS_PER_SIDE * 2, 4, 4), dtype=np.int_)
    K, G = _bulk_shear_from_E_nu(200000.0, 0.3)
    C = isotropic_stiffness(K, G)
    with pytest.raises(ValueError, match="cap"):
        fft_homogenize(big, {0: C})
