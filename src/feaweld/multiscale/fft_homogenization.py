"""Moulinec-Suquet FFT homogenization for voxel RVEs.

Computes an effective 6x6 Voigt stiffness tensor from a periodic voxel
microstructure using the basic Moulinec-Suquet fixed-point scheme with a
homogeneous reference medium.

Voigt convention used throughout:
    [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
    [eps_xx,   eps_yy,   eps_zz,   gamma_xy, gamma_yz, gamma_xz]
(engineering shear strains, factor of two on off-diagonals).

Units are SI-mm: stiffness in MPa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MAX_VOXELS_PER_SIDE = 128


@dataclass(frozen=True)
class IsotropicModuli:
    """Isotropic elastic moduli derived from a 6x6 Voigt stiffness."""

    bulk: float
    shear: float

    @property
    def youngs(self) -> float:
        denom = 3.0 * self.bulk + self.shear
        return 9.0 * self.bulk * self.shear / denom

    @property
    def poisson(self) -> float:
        denom = 2.0 * (3.0 * self.bulk + self.shear)
        return (3.0 * self.bulk - 2.0 * self.shear) / denom


def isotropic_stiffness(bulk: float, shear: float) -> NDArray[np.float64]:
    """Build a 6x6 Voigt stiffness tensor from bulk and shear moduli."""
    lam = bulk - 2.0 * shear / 3.0
    mu = shear
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def isotropic_moduli_from_stiffness(C: NDArray[np.float64]) -> IsotropicModuli:
    """Extract bulk and shear moduli from an approximately isotropic stiffness.

    Uses the Voigt averages of the bulk and shear moduli, which reduce to the
    exact values when the input tensor is isotropic in Voigt form.
    """
    bulk = (C[0, 0] + C[1, 1] + C[2, 2] + 2.0 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    shear = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15.0
    return IsotropicModuli(bulk=float(bulk), shear=float(shear))


def _build_stiffness_field(
    phase_map: NDArray[np.int_],
    phase_stiffness: dict[int, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return a (6,6,Nx,Ny,Nz) per-voxel stiffness field."""
    Nx, Ny, Nz = phase_map.shape
    C_field = np.zeros((6, 6, Nx, Ny, Nz), dtype=np.float64)
    for phase_id, C_phase in phase_stiffness.items():
        mask = phase_map == phase_id
        if not mask.any():
            continue
        C_arr = np.asarray(C_phase, dtype=np.float64)
        if C_arr.shape != (6, 6):
            raise ValueError(
                f"phase_stiffness[{phase_id}] must be a 6x6 matrix, got {C_arr.shape}"
            )
        C_field[:, :, mask] = C_arr[:, :, None]
    covered = np.zeros_like(phase_map, dtype=bool)
    for phase_id in phase_stiffness:
        covered |= phase_map == phase_id
    missing = np.unique(phase_map[~covered])
    if missing.size:
        raise ValueError(
            f"phase_map contains ids without stiffness definitions: {missing.tolist()}"
        )
    return C_field


def _reference_moduli(
    phase_map: NDArray[np.int_],
    phase_stiffness: dict[int, NDArray[np.float64]],
) -> tuple[float, float]:
    """Reference bulk and shear moduli for the Moulinec-Suquet scheme.

    Uses the midpoint of the softest and stiffest phases present in the RVE,
    which is the standard optimal choice for convergence speed of the basic
    Moulinec-Suquet fixed point (spectral radius of the iteration operator is
    minimized when ``C0 = (C_min + C_max) / 2``).
    """
    present_ids = [pid for pid in phase_stiffness if np.any(phase_map == pid)]
    if not present_ids:
        raise ValueError("No phases from phase_stiffness present in phase_map")

    k_values = []
    g_values = []
    for pid in present_ids:
        mod = isotropic_moduli_from_stiffness(
            np.asarray(phase_stiffness[pid], dtype=np.float64)
        )
        k_values.append(mod.bulk)
        g_values.append(mod.shear)
    k0 = 0.5 * (min(k_values) + max(k_values))
    g0 = 0.5 * (min(g_values) + max(g_values))
    return k0, g0


_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _green_operator_voigt(
    shape: tuple[int, int, int],
    lam0: float,
    mu0: float,
) -> NDArray[np.float64]:
    """Moulinec-Suquet Green's operator as a (6, 6, Nx, Ny, Nz) Voigt field.

    The operator maps a polarization stress in Voigt notation (no factor of 2
    on off-diagonals) to a strain fluctuation in Voigt notation with
    engineering shears (factor of 2 on off-diagonals). The zero-frequency
    component is zero, which enforces the zero-mean fluctuation constraint.
    """
    Nx, Ny, Nz = shape
    kx = np.fft.fftfreq(Nx) * 2.0 * np.pi * Nx
    ky = np.fft.fftfreq(Ny) * 2.0 * np.pi * Ny
    kz = np.fft.fftfreq(Nz) * 2.0 * np.pi * Nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = np.where(K2 == 0.0, 1.0, K2)
    K = np.stack([KX, KY, KZ], axis=0)
    xi = K / np.sqrt(K2_safe)

    factor = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0))
    inv_4mu0 = 1.0 / (4.0 * mu0)

    def gamma_tensor(i: int, j: int, k: int, l: int) -> NDArray[np.float64]:
        delta_ik = 1.0 if i == k else 0.0
        delta_il = 1.0 if i == l else 0.0
        delta_jk = 1.0 if j == k else 0.0
        delta_jl = 1.0 if j == l else 0.0
        term1 = -inv_4mu0 * (
            delta_ik * xi[j] * xi[l]
            + delta_il * xi[j] * xi[k]
            + delta_jk * xi[i] * xi[l]
            + delta_jl * xi[i] * xi[k]
        )
        term2 = factor * xi[i] * xi[j] * xi[k] * xi[l]
        return term1 + term2

    gamma_v = np.zeros((6, 6, Nx, Ny, Nz), dtype=np.float64)
    for I, (i, j) in enumerate(_VOIGT_IJ):
        eng_i = 2.0 if I >= 3 else 1.0
        for J, (k, l) in enumerate(_VOIGT_IJ):
            sym_k = 2.0 if J >= 3 else 1.0
            gamma_v[I, J] = eng_i * sym_k * gamma_tensor(i, j, k, l)
    gamma_v[..., 0, 0, 0] = 0.0
    return gamma_v


def _apply_green_operator_voigt(
    gamma_v: NDArray[np.float64],
    tau_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the Voigt Green's operator to a (6, Nx, Ny, Nz) polarization field."""
    tau_hat = np.fft.fftn(tau_voigt, axes=(-3, -2, -1))
    eps_hat = np.einsum("ijxyz,jxyz->ixyz", gamma_v, tau_hat)
    eps = np.fft.ifftn(eps_hat, axes=(-3, -2, -1)).real
    return eps


def _apply_stiffness(
    C_field: NDArray[np.float64],
    eps_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Contract a (6,6,Nx,Ny,Nz) stiffness field with a (6,Nx,Ny,Nz) strain field."""
    return np.einsum("ijxyz,jxyz->ixyz", C_field, eps_voigt)


def fft_homogenize(
    phase_map: NDArray[np.int_],
    phase_stiffness: dict[int, NDArray[np.float64]],
    loading: str = "all_six",
    tol: float = 1e-6,
    max_iter: int = 500,
) -> NDArray[np.float64]:
    """Compute the effective 6x6 Voigt stiffness of a voxel RVE.

    Args:
        phase_map: Integer array of shape (Nx, Ny, Nz) giving phase id per voxel.
        phase_stiffness: Mapping phase_id -> 6x6 Voigt stiffness (MPa).
        loading: ``"all_six"`` to assemble the full 6x6 tensor, or one of
            ``"xx","yy","zz","xy","yz","xz"`` for a single macro strain direction.
        tol: Convergence tolerance on the equilibrium residual.
        max_iter: Maximum fixed-point iterations per load case.

    Returns:
        Effective 6x6 Voigt stiffness tensor (MPa). When ``loading`` selects a
        single direction the corresponding column is populated and the other
        columns are left as zeros.
    """
    phase_map = np.asarray(phase_map)
    if phase_map.ndim != 3:
        raise ValueError(f"phase_map must be 3D, got shape {phase_map.shape}")
    Nx, Ny, Nz = phase_map.shape
    if max(Nx, Ny, Nz) > MAX_VOXELS_PER_SIDE:
        raise ValueError(
            f"RVE voxel grid exceeds the {MAX_VOXELS_PER_SIDE}^3 cap "
            f"(got {Nx}x{Ny}x{Nz}); reduce resolution or use a coarser model"
        )

    C_field = _build_stiffness_field(phase_map.astype(np.int_), phase_stiffness)

    k0, mu0 = _reference_moduli(phase_map.astype(np.int_), phase_stiffness)
    if mu0 <= 0.0 or k0 <= 0.0:
        raise ValueError("Non-positive reference moduli; check phase stiffnesses")
    lam0 = k0 - 2.0 * mu0 / 3.0
    C0 = isotropic_stiffness(k0, mu0)
    dC_field = C_field - C0[:, :, None, None, None]

    gamma_v = _green_operator_voigt((Nx, Ny, Nz), lam0, mu0)

    loading_map = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}
    if loading == "all_six":
        directions = list(range(6))
    elif loading in loading_map:
        directions = [loading_map[loading]]
    else:
        raise ValueError(f"Unknown loading direction: {loading!r}")

    C_eff = np.zeros((6, 6), dtype=np.float64)
    volume = float(Nx * Ny * Nz)

    for col in directions:
        E_macro = np.zeros(6, dtype=np.float64)
        E_macro[col] = 1.0
        eps = np.zeros((6, Nx, Ny, Nz), dtype=np.float64)
        for i in range(6):
            eps[i] = E_macro[i]

        prev_eps = eps.copy()
        for iteration in range(max_iter):
            polarization_voigt = _apply_stiffness(dC_field, eps)
            delta_eps_voigt = _apply_green_operator_voigt(gamma_v, polarization_voigt)
            eps = delta_eps_voigt
            for i in range(6):
                eps[i] += E_macro[i]

            if iteration > 0:
                change = np.max(np.abs(eps - prev_eps))
                if change < tol:
                    break
            prev_eps = eps.copy()

        sigma = _apply_stiffness(C_field, eps)
        sigma_avg = sigma.reshape(6, -1).sum(axis=1) / volume
        C_eff[:, col] = sigma_avg

    C_eff = 0.5 * (C_eff + C_eff.T)
    return C_eff


def mori_tanaka_two_phase(
    C_matrix: NDArray[np.float64],
    C_inclusion: NDArray[np.float64],
    f_inclusion: float,
) -> NDArray[np.float64]:
    """Mori-Tanaka effective stiffness for a two-phase isotropic composite.

    Assumes spherical inclusions in an isotropic matrix, both phases isotropic.
    Returns the effective 6x6 Voigt stiffness tensor.
    """
    if not 0.0 <= f_inclusion <= 1.0:
        raise ValueError(f"f_inclusion must be in [0,1], got {f_inclusion}")

    mod_m = isotropic_moduli_from_stiffness(np.asarray(C_matrix, dtype=np.float64))
    mod_i = isotropic_moduli_from_stiffness(np.asarray(C_inclusion, dtype=np.float64))
    K_m, G_m = mod_m.bulk, mod_m.shear
    K_i, G_i = mod_i.bulk, mod_i.shear

    alpha = (3.0 * K_m) / (3.0 * K_m + 4.0 * G_m)
    beta = (6.0 * (K_m + 2.0 * G_m)) / (5.0 * (3.0 * K_m + 4.0 * G_m))

    A_k = 1.0 / (1.0 + alpha * (K_i / K_m - 1.0))
    A_g = 1.0 / (1.0 + beta * (G_i / G_m - 1.0))

    f_i = f_inclusion
    f_m = 1.0 - f_i
    K_eff = (f_m * K_m + f_i * K_i * A_k) / (f_m + f_i * A_k)
    G_eff = (f_m * G_m + f_i * G_i * A_g) / (f_m + f_i * A_g)
    return isotropic_stiffness(K_eff, G_eff)


def make_sphere_rve(
    Nx: int,
    radius_frac: float = 0.3,
    phase_matrix: int = 0,
    phase_inclusion: int = 1,
) -> NDArray[np.int_]:
    """Build a cubic voxel RVE with a single centered spherical inclusion."""
    if Nx <= 0:
        raise ValueError(f"Nx must be positive, got {Nx}")
    if not 0.0 < radius_frac < 0.5:
        raise ValueError(f"radius_frac must be in (0, 0.5), got {radius_frac}")

    coords = (np.arange(Nx) + 0.5) / Nx - 0.5
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    mask = r2 <= radius_frac * radius_frac
    phase_map = np.full((Nx, Nx, Nx), phase_matrix, dtype=np.int_)
    phase_map[mask] = phase_inclusion
    return phase_map
