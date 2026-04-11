"""Multi-axial fatigue criteria with critical-plane search.

Provides six classical multi-axial fatigue criteria operating on full
stress tensor histories: Findley, Dang Van, Sines, Crossland,
Fatemi-Socie, and McDiarmid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MultiAxialResult:
    criterion: str
    damage_parameter: float
    critical_plane_normal: NDArray
    shear_amplitude: float
    normal_stress_max: float
    hydrostatic_mean: float
    life_estimate: float | None = None


def _as_history(stress_history: NDArray) -> NDArray:
    arr = np.asarray(stress_history, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError(
                f"stress_history must have last dimension 6, got shape {arr.shape}"
            )
        arr = arr.reshape(1, 6)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(
            f"stress_history must have shape (n_t, 6), got {arr.shape}"
        )
    return arr


def _voigt_to_tensor(voigt: NDArray) -> NDArray:
    """Convert (n_t, 6) Voigt array to (n_t, 3, 3) full tensors."""
    s = voigt
    n_t = s.shape[0]
    t = np.empty((n_t, 3, 3), dtype=float)
    t[:, 0, 0] = s[:, 0]
    t[:, 1, 1] = s[:, 1]
    t[:, 2, 2] = s[:, 2]
    t[:, 0, 1] = s[:, 3]
    t[:, 1, 0] = s[:, 3]
    t[:, 1, 2] = s[:, 4]
    t[:, 2, 1] = s[:, 4]
    t[:, 0, 2] = s[:, 5]
    t[:, 2, 0] = s[:, 5]
    return t


def _deviatoric(voigt: NDArray) -> NDArray:
    """Deviatoric Voigt components, shape (n_t, 6)."""
    dev = voigt.copy()
    p = (voigt[:, 0] + voigt[:, 1] + voigt[:, 2]) / 3.0
    dev[:, 0] -= p
    dev[:, 1] -= p
    dev[:, 2] -= p
    return dev


def _j2(dev: NDArray) -> NDArray:
    """J2 invariant from deviatoric Voigt components, shape (n_t,)."""
    return 0.5 * (
        dev[:, 0] ** 2 + dev[:, 1] ** 2 + dev[:, 2] ** 2
    ) + (dev[:, 3] ** 2 + dev[:, 4] ** 2 + dev[:, 5] ** 2)


def _hydrostatic(voigt: NDArray) -> NDArray:
    return (voigt[:, 0] + voigt[:, 1] + voigt[:, 2]) / 3.0


def fibonacci_sphere_grid(n: int = 200) -> NDArray:
    """Quasi-uniform n-point grid on the upper hemisphere.

    Returns (n, 3) unit vectors. Uses the Fibonacci spiral construction.
    We restrict to the upper hemisphere because plane +n and -n define the
    same plane in these criteria.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    golden = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n, dtype=float)
    # Upper hemisphere: z in [0, 1]
    z = 1.0 - indices / max(n - 1, 1) if n > 1 else np.array([1.0])
    # But first point at z=1 is fine; keep cos uniform on [0, 1]
    z = (indices + 0.5) / n  # in (0, 1), gives z in (0, 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden * indices
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    pts = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def resolve_on_plane(
    stress_history: NDArray,
    normal: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return (normal_stress(t), shear_vector(t)) on the given plane.

    Uses the Cauchy relation: t = sigma . n; sigma_n = t . n; tau = t - sigma_n * n.
    Returns normal_stress as (n_t,) and shear_vector as (n_t, 3).
    """
    s = _as_history(stress_history)
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    tensors = _voigt_to_tensor(s)
    traction = np.einsum("tij,j->ti", tensors, n)
    normal_stress = traction @ n
    shear_vector = traction - normal_stress[:, None] * n[None, :]
    return normal_stress, shear_vector


def shear_amplitude_mrh(shear_history: NDArray) -> float:
    """Minimum-radius hypersphere (MRH) amplitude for a shear trajectory.

    For a 3D shear vector history, MRH finds the smallest sphere enclosing
    all points; its radius is the shear amplitude. For MVP, compute the
    max |tau - tau_mean| as a cheap proxy that matches MRH for symmetric
    loading.
    """
    arr = np.asarray(shear_history, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mean = arr.mean(axis=0)
    diffs = arr - mean
    radii = np.linalg.norm(diffs, axis=1)
    return float(radii.max())


def _plane_search_normal_shear(
    stress_history: NDArray,
    n_planes: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """For each plane in the grid, compute shear amplitude and max normal stress.

    Returns (normals, tau_a_per_plane, sigma_n_max_per_plane).
    """
    s = _as_history(stress_history)
    tensors = _voigt_to_tensor(s)
    normals = fibonacci_sphere_grid(n_planes)
    # traction[p, t, i] = tensors[t, i, j] normals[p, j]
    traction = np.einsum("tij,pj->pti", tensors, normals)
    # normal_stress[p, t] = traction[p, t, :] . normals[p, :]
    normal_stress = np.einsum("pti,pi->pt", traction, normals)
    shear_vec = traction - normal_stress[:, :, None] * normals[:, None, :]
    shear_mean = shear_vec.mean(axis=1, keepdims=True)
    shear_dev = shear_vec - shear_mean
    shear_radii = np.linalg.norm(shear_dev, axis=2)
    tau_a = shear_radii.max(axis=1)
    sigma_n_max = normal_stress.max(axis=1)
    return normals, tau_a, sigma_n_max


def findley_criterion(
    stress_history: NDArray,
    k_param: float = 0.3,
    n_planes: int = 200,
) -> MultiAxialResult:
    """Findley: damage = max over planes of (tau_a + k * sigma_n_max).

    Returns the plane that maximizes this combination.
    """
    s = _as_history(stress_history)
    normals, tau_a, sigma_n_max = _plane_search_normal_shear(s, n_planes)
    damage_per_plane = tau_a + k_param * sigma_n_max
    idx = int(np.argmax(damage_per_plane))
    return MultiAxialResult(
        criterion="findley",
        damage_parameter=float(damage_per_plane[idx]),
        critical_plane_normal=normals[idx].copy(),
        shear_amplitude=float(tau_a[idx]),
        normal_stress_max=float(sigma_n_max[idx]),
        hydrostatic_mean=float(_hydrostatic(s).mean()),
    )


def dang_van_criterion(
    stress_history: NDArray,
    alpha_dv: float = 0.3,
    beta_dv: float = 200.0,
) -> MultiAxialResult:
    """Dang Van: damage = max_t (tau_meso(t) + alpha * sigma_h(t)).

    Mesoscopic shear approximated as microscopic deviatoric shear invariant
    after hydrostatic subtraction. Does not require plane search (works in
    stress invariant space).
    """
    s = _as_history(stress_history)
    dev = _deviatoric(s)
    # Mean deviator (shakedown centre) subtracted to estimate mesoscopic shear.
    dev_mean = dev.mean(axis=0, keepdims=True)
    dev_meso = dev - dev_mean
    j2_meso = _j2(dev_meso)
    tau_meso = np.sqrt(np.maximum(j2_meso, 0.0))
    sigma_h = _hydrostatic(s)
    instantaneous = tau_meso + alpha_dv * sigma_h
    idx = int(np.argmax(instantaneous))
    return MultiAxialResult(
        criterion="dang_van",
        damage_parameter=float(instantaneous[idx]),
        critical_plane_normal=np.zeros(3),
        shear_amplitude=float(tau_meso.max()),
        normal_stress_max=float(sigma_h.max()),
        hydrostatic_mean=float(sigma_h.mean()),
    )


def sines_criterion(
    stress_history: NDArray,
    alpha_s: float = 0.2,
    beta_s: float = 200.0,
) -> MultiAxialResult:
    """Sines: damage = sqrt(J2_a) + alpha * sigma_h_mean."""
    s = _as_history(stress_history)
    dev = _deviatoric(s)
    j2 = _j2(dev)
    j2_a = 0.5 * (j2.max() - j2.min())
    sigma_h = _hydrostatic(s)
    sigma_h_mean = float(sigma_h.mean())
    damage = float(np.sqrt(max(j2_a, 0.0)) + alpha_s * sigma_h_mean)
    return MultiAxialResult(
        criterion="sines",
        damage_parameter=damage,
        critical_plane_normal=np.zeros(3),
        shear_amplitude=float(np.sqrt(max(j2_a, 0.0))),
        normal_stress_max=float(sigma_h.max()),
        hydrostatic_mean=sigma_h_mean,
    )


def crossland_criterion(
    stress_history: NDArray,
    alpha_c: float = 0.3,
    beta_c: float = 200.0,
) -> MultiAxialResult:
    """Crossland: damage = sqrt(J2_a) + alpha * sigma_h_max."""
    s = _as_history(stress_history)
    dev = _deviatoric(s)
    j2 = _j2(dev)
    j2_a = 0.5 * (j2.max() - j2.min())
    sigma_h = _hydrostatic(s)
    sigma_h_max = float(sigma_h.max())
    damage = float(np.sqrt(max(j2_a, 0.0)) + alpha_c * sigma_h_max)
    return MultiAxialResult(
        criterion="crossland",
        damage_parameter=damage,
        critical_plane_normal=np.zeros(3),
        shear_amplitude=float(np.sqrt(max(j2_a, 0.0))),
        normal_stress_max=sigma_h_max,
        hydrostatic_mean=float(sigma_h.mean()),
    )


def fatemi_socie_criterion(
    stress_history: NDArray,
    k_fs: float = 0.5,
    sigma_y: float = 250.0,
    n_planes: int = 200,
) -> MultiAxialResult:
    """Fatemi-Socie: max over planes of gamma_a * (1 + k * sigma_n_max / sigma_y).

    For MVP, convert stress to engineering shear strain via gamma = 2*tau/G
    with G = material shear modulus (passed via sigma_y context heuristic
    or default 80 GPa for steel).
    """
    s = _as_history(stress_history)
    normals, tau_a, sigma_n_max = _plane_search_normal_shear(s, n_planes)
    G = 80e3  # MPa
    gamma_a = 2.0 * tau_a / G
    damage_per_plane = gamma_a * (1.0 + k_fs * sigma_n_max / sigma_y)
    idx = int(np.argmax(damage_per_plane))
    return MultiAxialResult(
        criterion="fatemi_socie",
        damage_parameter=float(damage_per_plane[idx]),
        critical_plane_normal=normals[idx].copy(),
        shear_amplitude=float(tau_a[idx]),
        normal_stress_max=float(sigma_n_max[idx]),
        hydrostatic_mean=float(_hydrostatic(s).mean()),
    )


def mcdiarmid_criterion(
    stress_history: NDArray,
    t_a_limit: float = 200.0,
    sigma_u: float = 400.0,
    n_planes: int = 200,
) -> MultiAxialResult:
    """McDiarmid: damage = tau_a / t_a_limit + sigma_n_max / (2 * sigma_u)."""
    s = _as_history(stress_history)
    normals, tau_a, sigma_n_max = _plane_search_normal_shear(s, n_planes)
    damage_per_plane = tau_a / t_a_limit + sigma_n_max / (2.0 * sigma_u)
    idx = int(np.argmax(damage_per_plane))
    return MultiAxialResult(
        criterion="mcdiarmid",
        damage_parameter=float(damage_per_plane[idx]),
        critical_plane_normal=normals[idx].copy(),
        shear_amplitude=float(tau_a[idx]),
        normal_stress_max=float(sigma_n_max[idx]),
        hydrostatic_mean=float(_hydrostatic(s).mean()),
    )
