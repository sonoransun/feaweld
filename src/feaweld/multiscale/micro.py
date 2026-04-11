"""Micro-scale material property models for multi-scale weld analysis.

Implements grain-scale constitutive relationships:
- Hall-Petch grain size → yield strength
- Dislocation density hardening
- Property homogenization from micro to meso scale
- CCT diagram interpolation for phase predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from feaweld.multiscale.fft_homogenization import fft_homogenize


@dataclass
class HallPetchParams:
    """Hall-Petch parameters for grain-size strengthening.

    σ_y = σ_0 + k_y / √d

    where d is grain size in meters.
    """
    sigma_0: float  # Friction stress (MPa) — lattice resistance
    k_y: float      # Hall-Petch slope (MPa·√m)

    def yield_strength(self, grain_size_um: float) -> float:
        """Compute yield strength for given grain size.

        Args:
            grain_size_um: Average grain diameter (μm)

        Returns:
            Yield strength (MPa)
        """
        d_m = grain_size_um * 1e-6  # convert to meters
        return self.sigma_0 + self.k_y / np.sqrt(d_m)


# Common Hall-Petch parameters
HALL_PETCH_LOW_CARBON_STEEL = HallPetchParams(sigma_0=70.0, k_y=0.74)
HALL_PETCH_MILD_STEEL = HallPetchParams(sigma_0=50.0, k_y=0.70)
HALL_PETCH_304_STAINLESS = HallPetchParams(sigma_0=120.0, k_y=0.55)
HALL_PETCH_FERRITIC = HallPetchParams(sigma_0=70.0, k_y=0.74)


@dataclass
class DislocationDensityParams:
    """Dislocation density hardening parameters.

    σ = σ_0 + α·M·G·b·√ρ

    where:
        α = dislocation interaction parameter (~0.3)
        M = Taylor factor (~3.06 for BCC, ~3.1 for FCC)
        G = shear modulus (MPa)
        b = Burgers vector magnitude (m)
        ρ = dislocation density (1/m²)
    """
    alpha: float = 0.3      # Interaction parameter
    M: float = 3.06         # Taylor factor (BCC polycrystal)
    G: float = 80000.0      # Shear modulus (MPa)
    b: float = 2.48e-10     # Burgers vector for iron (m)
    sigma_0: float = 50.0   # Base friction stress (MPa)

    def flow_stress(self, dislocation_density: float) -> float:
        """Compute flow stress from dislocation density.

        Args:
            dislocation_density: ρ (1/m²), typical range 1e12 to 1e16

        Returns:
            Flow stress (MPa)
        """
        return self.sigma_0 + self.alpha * self.M * self.G * self.b * np.sqrt(dislocation_density)

    def dislocation_density_from_stress(self, stress: float) -> float:
        """Inverse: estimate dislocation density from measured stress.

        Args:
            stress: Measured flow stress (MPa)

        Returns:
            Estimated dislocation density (1/m²)
        """
        sigma_disl = max(stress - self.sigma_0, 0.0)
        factor = self.alpha * self.M * self.G * self.b
        if factor < 1e-20:
            return 0.0
        return (sigma_disl / factor) ** 2


# Default parameters for iron/steel
DISLOCATION_BCC_IRON = DislocationDensityParams(
    alpha=0.3, M=3.06, G=80000.0, b=2.48e-10, sigma_0=50.0
)
DISLOCATION_FCC_AUSTENITE = DislocationDensityParams(
    alpha=0.3, M=3.1, G=75000.0, b=2.55e-10, sigma_0=80.0
)


def homogenize_properties(
    phase_fractions: dict[str, float],
    phase_properties: dict[str, dict[str, float]],
    method: str = "voigt",
) -> dict[str, float]:
    """Homogenize micro-scale properties to meso-scale effective values.

    Args:
        phase_fractions: {"ferrite": 0.7, "pearlite": 0.2, "bainite": 0.1}
        phase_properties: {"ferrite": {"E": 210000, "sigma_y": 200}, ...}
        method: "voigt" (upper bound), "reuss" (lower bound), "hill" (average)

    Returns:
        Dict of effective properties {"E": ..., "sigma_y": ..., ...}
    """
    # Get all property names from first phase
    prop_names = set()
    for props in phase_properties.values():
        prop_names.update(props.keys())

    result = {}
    for prop in prop_names:
        if method == "voigt":
            # Voigt (iso-strain): arithmetic mean
            result[prop] = sum(
                phase_fractions.get(phase, 0.0) * phase_properties.get(phase, {}).get(prop, 0.0)
                for phase in phase_fractions
            )
        elif method == "reuss":
            # Reuss (iso-stress): harmonic mean
            denom = sum(
                phase_fractions.get(phase, 0.0) / max(phase_properties.get(phase, {}).get(prop, 1e-12), 1e-12)
                for phase in phase_fractions
            )
            result[prop] = 1.0 / max(denom, 1e-12) if denom > 0 else 0.0
        elif method == "hill":
            # Hill: average of Voigt and Reuss
            voigt = sum(
                phase_fractions.get(phase, 0.0) * phase_properties.get(phase, {}).get(prop, 0.0)
                for phase in phase_fractions
            )
            denom = sum(
                phase_fractions.get(phase, 0.0) / max(phase_properties.get(phase, {}).get(prop, 1e-12), 1e-12)
                for phase in phase_fractions
            )
            reuss = 1.0 / max(denom, 1e-12) if denom > 0 else 0.0
            result[prop] = (voigt + reuss) / 2.0
        else:
            raise ValueError(f"Unknown homogenization method: {method}")

    return result


def micro_to_meso_properties(
    grain_size_um: float,
    dislocation_density: float = 1e14,
    phase_fractions: dict[str, float] | None = None,
    hall_petch: HallPetchParams | None = None,
    dislocation_params: DislocationDensityParams | None = None,
) -> dict[str, float]:
    """Compute meso-scale properties from micro-scale parameters.

    Combines Hall-Petch grain-size strengthening with dislocation hardening.

    Args:
        grain_size_um: Average grain size (μm)
        dislocation_density: ρ (1/m²)
        phase_fractions: Optional phase composition
        hall_petch: Hall-Petch parameters (default: low-carbon steel)
        dislocation_params: Dislocation model parameters (default: BCC iron)

    Returns:
        Dict with yield_strength, hardening_contribution, total_strength
    """
    hp = hall_petch or HALL_PETCH_LOW_CARBON_STEEL
    dp = dislocation_params or DISLOCATION_BCC_IRON

    sigma_hp = hp.yield_strength(grain_size_um)
    sigma_disl = dp.flow_stress(dislocation_density) - dp.sigma_0  # just the hardening part

    # Superposition (root-sum-square for independent mechanisms)
    sigma_total = hp.sigma_0 + np.sqrt(
        (sigma_hp - hp.sigma_0) ** 2 + sigma_disl ** 2
    )

    return {
        "yield_strength": sigma_total,
        "hall_petch_contribution": sigma_hp,
        "dislocation_contribution": sigma_disl,
        "grain_size_um": grain_size_um,
        "dislocation_density": dislocation_density,
    }


def estimate_grain_size_from_cooling(
    cooling_rate: float,
    initial_austenite_grain_um: float = 50.0,
) -> float:
    """Estimate transformed grain size from cooling rate.

    Faster cooling → finer grain structure (more nucleation sites).
    Empirical correlation: d ∝ (cooling_rate)^(-0.5)

    Args:
        cooling_rate: Cooling rate (C/s)
        initial_austenite_grain_um: Prior austenite grain size (μm)

    Returns:
        Estimated grain size (μm)
    """
    # Reference: at 1 C/s, grain size ≈ initial_austenite_grain * 0.6
    d_ref = initial_austenite_grain_um * 0.6
    rate_ref = 1.0
    return float(d_ref * np.sqrt(rate_ref / max(cooling_rate, 0.01)))


def homogenized_stiffness_from_rve(rve_path: str | Path) -> NDArray[np.float64]:
    """Load a voxel RVE from a ``.npz`` archive and FFT-homogenize it.

    The archive must contain:
        - ``phase_map``: integer array of shape (Nx, Ny, Nz)
        - ``phase_stiffness``: either a dict-like ``.npz`` member or a
          structured array mapping integer phase ids to 6x6 Voigt stiffnesses.
          The simplest supported layout is a stacked array of shape
          ``(n_phases, 6, 6)`` together with a ``phase_ids`` vector giving the
          phase id for each row.

    Returns:
        Effective 6x6 Voigt stiffness tensor (MPa).
    """
    path = Path(rve_path)
    with np.load(path, allow_pickle=True) as data:
        phase_map = np.asarray(data["phase_map"], dtype=np.int_)
        if "phase_ids" in data.files:
            ids = np.asarray(data["phase_ids"], dtype=np.int_)
            stack = np.asarray(data["phase_stiffness"], dtype=np.float64)
            if stack.ndim != 3 or stack.shape[1:] != (6, 6):
                raise ValueError(
                    f"phase_stiffness must be (n_phases, 6, 6), got {stack.shape}"
                )
            if ids.shape[0] != stack.shape[0]:
                raise ValueError("phase_ids and phase_stiffness length mismatch")
            phase_stiffness = {int(pid): stack[k] for k, pid in enumerate(ids)}
        else:
            raw = data["phase_stiffness"]
            if raw.dtype == object:
                mapping = raw.item()
                phase_stiffness = {
                    int(pid): np.asarray(C, dtype=np.float64) for pid, C in mapping.items()
                }
            else:
                stack = np.asarray(raw, dtype=np.float64)
                if stack.ndim != 3 or stack.shape[1:] != (6, 6):
                    raise ValueError(
                        f"phase_stiffness must be (n_phases, 6, 6), got {stack.shape}"
                    )
                phase_stiffness = {k: stack[k] for k in range(stack.shape[0])}
    return fft_homogenize(phase_map, phase_stiffness)


def phase_dependent_elastic_modulus(
    phase_fractions: dict[str, float],
) -> float:
    """Estimate elastic modulus from phase composition.

    E is relatively insensitive to microstructure in steel (~200-210 GPa),
    but martensite can be slightly stiffer.

    Args:
        phase_fractions: Phase fractions dict

    Returns:
        Effective elastic modulus (MPa)
    """
    E_phases = {
        "ferrite": 200000.0,
        "pearlite": 205000.0,
        "bainite": 210000.0,
        "martensite": 215000.0,
        "austenite": 195000.0,
    }
    E_eff = sum(
        phase_fractions.get(phase, 0.0) * E_phases.get(phase, 200000.0)
        for phase in phase_fractions
    )
    return max(E_eff, 180000.0)  # floor at 180 GPa
