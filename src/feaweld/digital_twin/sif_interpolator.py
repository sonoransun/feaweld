"""FEA-informed stress intensity factor interpolation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class SIFTable:
    """Tabulated SIF vs crack length from FEA."""
    crack_lengths: NDArray  # sorted ascending, mm
    sif_values: NDArray     # dK in MPa*sqrt(mm)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SIFInterpolator:
    """Interpolated dK(a) from tabulated FEA results."""

    def __init__(self, table: SIFTable, extrapolation: str = "paris") -> None:
        self._table = table
        self._extrapolation = extrapolation

    def __call__(self, a: float) -> float:
        if a <= 0.0:
            return 0.0
        t = self._table
        if a <= t.crack_lengths[-1] and a >= t.crack_lengths[0]:
            return float(np.interp(a, t.crack_lengths, t.sif_values))
        if self._extrapolation == "constant":
            if a < t.crack_lengths[0]:
                return float(t.sif_values[0])
            return float(t.sif_values[-1])
        elif self._extrapolation == "zero":
            return 0.0
        else:  # "paris" — use handbook formula fitted to endpoints
            if len(t.crack_lengths) < 2:
                return float(t.sif_values[0]) if len(t.sif_values) > 0 else 0.0
            # Fit Y*S from last point: dK = Y*S*sqrt(pi*a) => Y*S = dK / sqrt(pi*a)
            a_ref = float(t.crack_lengths[-1])
            dk_ref = float(t.sif_values[-1])
            ys = dk_ref / np.sqrt(np.pi * max(a_ref, 1e-12))
            return float(ys * np.sqrt(np.pi * a))

    @classmethod
    def from_handbook(cls, stress_range: float, geometry_factor: float = 1.12,
                      n_points: int = 50, a_max: float = 50.0) -> SIFInterpolator:
        a_vals = np.linspace(0.01, a_max, n_points)
        dk_vals = geometry_factor * stress_range * np.sqrt(np.pi * a_vals)
        table = SIFTable(crack_lengths=a_vals, sif_values=dk_vals, source="handbook")
        return cls(table)


def residual_stress_sif(
    sigma_res_fn: Callable[[NDArray], NDArray],
    plate_thickness: float,
    geometry_factor: float = 1.0,
) -> Callable[[float], float]:
    """Return K_res(a) via Bueckner weight function integration.

    Parameters
    ----------
    sigma_res_fn
        Callable mapping z/t ratios (NDArray) to residual stress (MPa).
    plate_thickness
        Plate thickness in mm.
    geometry_factor
        Additional geometry correction factor.
    """
    def K_res(a: float) -> float:
        if a <= 0.0:
            return 0.0
        a_eff = min(a, plate_thickness * 0.99)
        n_quad = 20
        z_pts = np.linspace(1e-6, a_eff - 1e-6, n_quad)
        z_over_t = z_pts / plate_thickness
        sigma = np.asarray(sigma_res_fn(z_over_t), dtype=np.float64)
        kernel = 1.0 / np.sqrt(a_eff ** 2 - z_pts ** 2)
        integrand = sigma * kernel
        K_val = (2.0 / np.sqrt(np.pi * a_eff)) * float(np.trapz(integrand, z_pts))
        return K_val * geometry_factor
    return K_res


def combined_sif(
    applied_fn: Callable[[float], float],
    residual_fn: Callable[[float], float] | None = None,
) -> Callable[[float], float]:
    """Combine applied and residual SIF contributions."""
    if residual_fn is None:
        return applied_fn
    def dK_eff(a: float) -> float:
        return max(applied_fn(a) + residual_fn(a), 0.0)
    return dK_eff
