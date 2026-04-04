"""Meso-scale weld bead analysis.

Models the weld bead cross-section with distinct metallurgical zones:
weld metal, coarse-grain HAZ, fine-grain HAZ, and base metal. Estimates
phase fractions from cooling rate and assigns local material properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class WeldZone(str, Enum):
    WELD_METAL = "weld_metal"
    COARSE_GRAIN_HAZ = "cg_haz"
    FINE_GRAIN_HAZ = "fg_haz"
    INTERCRITICAL_HAZ = "ic_haz"
    BASE_METAL = "base_metal"


@dataclass
class PhaseComposition:
    """Phase fractions in a weld zone."""
    ferrite: float = 0.0
    pearlite: float = 0.0
    bainite: float = 0.0
    martensite: float = 0.0
    austenite: float = 0.0

    def __post_init__(self) -> None:
        total = self.ferrite + self.pearlite + self.bainite + self.martensite + self.austenite
        if total > 0 and abs(total - 1.0) > 0.01:
            # Normalize
            self.ferrite /= total
            self.pearlite /= total
            self.bainite /= total
            self.martensite /= total
            self.austenite /= total


@dataclass
class CCTDiagram:
    """Continuous Cooling Transformation diagram data.

    Stores transformation start/finish temperatures and phase fractions
    as functions of cooling rate.
    """
    # Cooling rates (C/s) — from slow to fast
    cooling_rates: NDArray[np.float64]

    # Transformation temperatures at each cooling rate
    Ac1: float = 727.0    # Lower critical temperature (C)
    Ac3: float = 870.0    # Upper critical temperature (C)
    Ms: float = 350.0     # Martensite start temperature (C)

    # Phase fractions at each cooling rate: shape (n_rates,)
    ferrite_fraction: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    pearlite_fraction: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    bainite_fraction: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    martensite_fraction: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def predict_phases(self, cooling_rate: float) -> PhaseComposition:
        """Predict phase composition for a given cooling rate."""
        if len(self.cooling_rates) == 0:
            return _default_phase_prediction(cooling_rate)

        rates = self.cooling_rates
        f_ferrite = float(np.interp(cooling_rate, rates, self.ferrite_fraction))
        f_pearlite = float(np.interp(cooling_rate, rates, self.pearlite_fraction))
        f_bainite = float(np.interp(cooling_rate, rates, self.bainite_fraction))
        f_martensite = float(np.interp(cooling_rate, rates, self.martensite_fraction))

        return PhaseComposition(
            ferrite=f_ferrite,
            pearlite=f_pearlite,
            bainite=f_bainite,
            martensite=f_martensite,
        )


def default_low_carbon_cct() -> CCTDiagram:
    """Default CCT diagram for low-carbon structural steel (A36-like)."""
    rates = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 50.0, 100.0])
    return CCTDiagram(
        cooling_rates=rates,
        Ac1=727.0,
        Ac3=870.0,
        Ms=350.0,
        ferrite_fraction=np.array([0.80, 0.75, 0.70, 0.55, 0.40, 0.15, 0.05, 0.0]),
        pearlite_fraction=np.array([0.20, 0.20, 0.18, 0.15, 0.10, 0.05, 0.02, 0.0]),
        bainite_fraction=np.array([0.00, 0.05, 0.10, 0.25, 0.35, 0.50, 0.43, 0.20]),
        martensite_fraction=np.array([0.00, 0.00, 0.02, 0.05, 0.15, 0.30, 0.50, 0.80]),
    )


def cct_for_grade(grade: str) -> CCTDiagram:
    """Get a CCT diagram for a specific steel grade from the database.

    Falls back to ``default_low_carbon_cct()`` if the grade is not found.

    Args:
        grade: Steel grade identifier (e.g., "A36", "4140", "X65", "304SS").

    Returns:
        CCTDiagram for the requested grade.
    """
    try:
        from feaweld.data.cct import get_cct_diagram
        return get_cct_diagram(grade)
    except (KeyError, ImportError):
        return default_low_carbon_cct()


def _default_phase_prediction(cooling_rate: float) -> PhaseComposition:
    """Simple phase prediction without full CCT data."""
    if cooling_rate < 1.0:
        return PhaseComposition(ferrite=0.70, pearlite=0.20, bainite=0.10)
    elif cooling_rate < 10.0:
        f_m = min(0.15 * (cooling_rate - 1.0) / 9.0, 0.15)
        return PhaseComposition(ferrite=0.50, pearlite=0.10, bainite=0.25, martensite=f_m)
    elif cooling_rate < 50.0:
        f_m = 0.15 + 0.35 * (cooling_rate - 10.0) / 40.0
        return PhaseComposition(ferrite=0.10, bainite=0.40, martensite=f_m)
    else:
        return PhaseComposition(bainite=0.20, martensite=0.80)


@dataclass
class MesoZoneProperties:
    """Mechanical properties for a meso-scale weld zone."""
    zone: WeldZone
    phases: PhaseComposition
    yield_strength: float    # MPa
    ultimate_strength: float # MPa
    hardness_hv: float       # Vickers hardness
    grain_size_um: float     # Average grain size (μm)
    elastic_modulus: float = 200000.0  # MPa (roughly constant)


def estimate_zone_properties(
    zone: WeldZone,
    phases: PhaseComposition,
    base_yield: float = 250.0,
    base_uts: float = 400.0,
) -> MesoZoneProperties:
    """Estimate mechanical properties from phase composition.

    Uses mixture rules and empirical correlations for each phase.

    Args:
        zone: Which weld zone
        phases: Phase fractions
        base_yield: Base metal yield strength (MPa)
        base_uts: Base metal UTS (MPa)

    Returns:
        MesoZoneProperties with estimated mechanical properties.
    """
    # Phase-specific properties (typical for low-carbon steel)
    phase_yield = {
        "ferrite": base_yield * 0.85,
        "pearlite": base_yield * 1.20,
        "bainite": base_yield * 1.80,
        "martensite": base_yield * 3.00,
    }
    phase_uts = {
        "ferrite": base_uts * 0.85,
        "pearlite": base_uts * 1.15,
        "bainite": base_uts * 1.60,
        "martensite": base_uts * 2.50,
    }
    phase_hardness = {
        "ferrite": 130,
        "pearlite": 200,
        "bainite": 300,
        "martensite": 500,
    }

    # Mixture rule (linear for yield/UTS, linear for hardness)
    fractions = {
        "ferrite": phases.ferrite,
        "pearlite": phases.pearlite,
        "bainite": phases.bainite,
        "martensite": phases.martensite,
    }

    sigma_y = sum(fractions[p] * phase_yield[p] for p in fractions)
    sigma_u = sum(fractions[p] * phase_uts[p] for p in fractions)
    hv = sum(fractions[p] * phase_hardness[p] for p in fractions)

    # Grain size estimation by zone
    grain_sizes = {
        WeldZone.WELD_METAL: 80.0,       # Coarse columnar
        WeldZone.COARSE_GRAIN_HAZ: 100.0, # Very coarse
        WeldZone.FINE_GRAIN_HAZ: 15.0,    # Fine recrystallized
        WeldZone.INTERCRITICAL_HAZ: 25.0, # Mixed
        WeldZone.BASE_METAL: 30.0,        # Original
    }

    return MesoZoneProperties(
        zone=zone,
        phases=phases,
        yield_strength=sigma_y,
        ultimate_strength=sigma_u,
        hardness_hv=hv,
        grain_size_um=grain_sizes.get(zone, 30.0),
    )


def sdas_to_yield_strength(sdas_um: float, base_yield: float = 250.0) -> float:
    """Estimate yield strength from secondary dendrite arm spacing (SDAS).

    Empirical correlation: σ_y ∝ (SDAS)^(-0.5) (Hall-Petch-like)

    Args:
        sdas_um: Secondary dendrite arm spacing (μm)
        base_yield: Base metal yield strength (MPa)

    Returns:
        Estimated yield strength (MPa)
    """
    # Reference: SDAS_ref=40μm gives base_yield for weld metal
    sdas_ref = 40.0
    k = base_yield * np.sqrt(sdas_ref)
    return float(k / np.sqrt(max(sdas_um, 1.0)))


def assign_zones(
    node_positions: NDArray[np.float64],
    weld_center: NDArray[np.float64],
    weld_radius: float,
    haz_width: float = 3.0,
) -> NDArray:
    """Assign weld zone labels to mesh nodes based on distance from weld center.

    Simple radial zone assignment:
    - Within weld_radius: WELD_METAL
    - weld_radius to +1mm: COARSE_GRAIN_HAZ
    - +1mm to +2mm: FINE_GRAIN_HAZ
    - +2mm to +haz_width: INTERCRITICAL_HAZ
    - Beyond: BASE_METAL

    Args:
        node_positions: (n_nodes, 3) node coordinates
        weld_center: (3,) center of weld
        weld_radius: Radius of weld metal region (mm)
        haz_width: Total HAZ width (mm)

    Returns:
        Array of WeldZone enum values for each node.
    """
    distances = np.linalg.norm(node_positions - weld_center, axis=1)
    zones = np.empty(len(distances), dtype=object)

    cg_boundary = weld_radius + 1.0
    fg_boundary = weld_radius + 2.0
    ic_boundary = weld_radius + haz_width

    for i, d in enumerate(distances):
        if d <= weld_radius:
            zones[i] = WeldZone.WELD_METAL
        elif d <= cg_boundary:
            zones[i] = WeldZone.COARSE_GRAIN_HAZ
        elif d <= fg_boundary:
            zones[i] = WeldZone.FINE_GRAIN_HAZ
        elif d <= ic_boundary:
            zones[i] = WeldZone.INTERCRITICAL_HAZ
        else:
            zones[i] = WeldZone.BASE_METAL

    return zones


def cooling_rate_from_thermal(
    temperature_history: NDArray[np.float64],
    time_steps: NDArray[np.float64],
    T_high: float = 800.0,
    T_low: float = 500.0,
) -> NDArray[np.float64]:
    """Estimate cooling rate (C/s) at each node from thermal history.

    Computes average cooling rate between T_high and T_low (t8/5 method).

    Args:
        temperature_history: (n_timesteps, n_nodes) temperatures
        time_steps: (n_timesteps,) time values (s)
        T_high: Upper temperature for cooling rate calculation (C)
        T_low: Lower temperature for cooling rate calculation (C)

    Returns:
        (n_nodes,) cooling rates in C/s
    """
    n_times, n_nodes = temperature_history.shape
    rates = np.zeros(n_nodes)

    for j in range(n_nodes):
        temp_curve = temperature_history[:, j]
        peak_idx = np.argmax(temp_curve)

        # Only consider cooling (after peak)
        cooling_temps = temp_curve[peak_idx:]
        cooling_times = time_steps[peak_idx:]

        if len(cooling_temps) < 2:
            continue

        # Find time at T_high and T_low during cooling
        t_high = None
        t_low = None

        for k in range(len(cooling_temps) - 1):
            if cooling_temps[k] >= T_high and cooling_temps[k + 1] < T_high:
                # Interpolate
                frac = (T_high - cooling_temps[k + 1]) / (cooling_temps[k] - cooling_temps[k + 1])
                t_high = cooling_times[k + 1] + frac * (cooling_times[k] - cooling_times[k + 1])
            if cooling_temps[k] >= T_low and cooling_temps[k + 1] < T_low:
                frac = (T_low - cooling_temps[k + 1]) / (cooling_temps[k] - cooling_temps[k + 1])
                t_low = cooling_times[k + 1] + frac * (cooling_times[k] - cooling_times[k + 1])

        if t_high is not None and t_low is not None and t_low > t_high:
            rates[j] = (T_high - T_low) / (t_low - t_high)

    return rates
