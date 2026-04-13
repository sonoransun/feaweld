"""Multiscale pipeline integration.

Connects the thermal solver output to the CCT diagram → micro-scale
property calculation chain, producing spatially varying material
properties for the HAZ.

Workflow::

    thermal_results (temperature history per node)
      → cooling_rate_from_thermal()  (t8/5 method)
      → CCTDiagram.predict_phases()  (per node)
      → micro_to_meso_properties()   (Hall-Petch + dislocation)
      → spatially varying yield strength / modulus
      → stored on WorkflowResult.extensions["microstructure"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.logging import get_logger
from feaweld.core.types import FEAResults, FEMesh

logger = get_logger(__name__)


@dataclass
class MultiscaleResult:
    """Result of a multiscale analysis pass."""

    # Per-node results (n_nodes,)
    cooling_rates: NDArray[np.floating]           # C/s
    yield_strength: NDArray[np.floating]           # MPa
    elastic_modulus: NDArray[np.floating]           # MPa

    # Per-node phase compositions (n_nodes, 5) — ferrite, pearlite, bainite, martensite, austenite
    phase_fractions: NDArray[np.floating]

    # Zone labels per node
    zone_labels: NDArray[np.intp] | None = None

    # Summary
    mean_yield: float = 0.0
    min_yield: float = 0.0
    max_yield: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "mean_yield_mpa": float(self.mean_yield),
            "min_yield_mpa": float(self.min_yield),
            "max_yield_mpa": float(self.max_yield),
            "mean_cooling_rate": float(np.mean(self.cooling_rates)),
            "n_nodes": len(self.cooling_rates),
        }


def run_multiscale(
    fea_results: FEAResults,
    mesh: FEMesh,
    *,
    grade: str = "A36",
    base_yield: float = 250.0,
    base_uts: float = 400.0,
    weld_center: NDArray[np.floating] | None = None,
    weld_radius: float = 5.0,
    haz_width: float = 3.0,
) -> MultiscaleResult:
    """Run the full multiscale analysis on thermal FEA results.

    Args:
        fea_results: Results from a thermal transient solve (must have
            ``temperature`` field with shape ``(n_timesteps, n_nodes)``
            or ``(n_nodes,)`` for steady-state).
        mesh: The finite element mesh.
        grade: Steel grade for CCT diagram lookup.
        base_yield: Base metal yield strength (MPa).
        base_uts: Base metal ultimate tensile strength (MPa).
        weld_center: Center of the weld for zone assignment.
        weld_radius: Approximate weld bead radius (mm).
        haz_width: HAZ width multiplier.

    Returns:
        MultiscaleResult with per-node properties.
    """
    from feaweld.multiscale.meso import (
        assign_zones,
        cct_for_grade,
        cooling_rate_from_thermal,
    )
    from feaweld.multiscale.micro import (
        estimate_grain_size_from_cooling,
        micro_to_meso_properties,
        phase_dependent_elastic_modulus,
    )

    n_nodes = mesh.n_nodes
    logger.info("Multiscale analysis: %d nodes, grade=%s", n_nodes, grade)

    # --- Step 1: Extract cooling rates ---
    temp = fea_results.temperature
    if temp is None:
        raise ValueError("FEA results must have temperature data for multiscale analysis")

    temp_arr = np.asarray(temp, dtype=float)

    if temp_arr.ndim == 2:
        # Transient: (n_timesteps, n_nodes) — need time_steps
        # Estimate time steps from the array shape.
        n_steps = temp_arr.shape[0]
        time_steps = np.linspace(0, 100, n_steps)  # default 0-100s
        cooling_rates = cooling_rate_from_thermal(temp_arr, time_steps)
    elif temp_arr.ndim == 1:
        # Steady-state: estimate uniform cooling rate
        cooling_rates = np.full(n_nodes, 10.0)  # default 10 C/s
        logger.debug("Steady-state temperature field, using default cooling rate 10 C/s")
    else:
        raise ValueError(f"Unexpected temperature shape: {temp_arr.shape}")

    # Clamp to reasonable range.
    cooling_rates = np.clip(cooling_rates, 0.1, 500.0)

    # --- Step 2: CCT phase prediction ---
    cct = cct_for_grade(grade)
    phase_fractions = np.zeros((n_nodes, 5), dtype=float)

    for i in range(n_nodes):
        phases = cct.predict_phases(float(cooling_rates[i]))
        phase_fractions[i] = [
            phases.ferrite,
            phases.pearlite,
            phases.bainite,
            phases.martensite,
            phases.austenite,
        ]

    logger.debug("Phase prediction complete: mean martensite=%.2f%%",
                  np.mean(phase_fractions[:, 3]) * 100)

    # --- Step 3: Micro-to-meso property mapping ---
    yield_arr = np.zeros(n_nodes, dtype=float)
    modulus_arr = np.zeros(n_nodes, dtype=float)

    for i in range(n_nodes):
        cr = float(cooling_rates[i])
        grain_size = estimate_grain_size_from_cooling(cr)
        props = micro_to_meso_properties(
            grain_size_um=grain_size,
            phase_fractions={
                "ferrite": float(phase_fractions[i, 0]),
                "pearlite": float(phase_fractions[i, 1]),
                "bainite": float(phase_fractions[i, 2]),
                "martensite": float(phase_fractions[i, 3]),
                "austenite": float(phase_fractions[i, 4]),
            },
        )
        yield_arr[i] = props.get("yield_strength", base_yield)
        modulus_arr[i] = phase_dependent_elastic_modulus({
            "ferrite": float(phase_fractions[i, 0]),
            "pearlite": float(phase_fractions[i, 1]),
            "bainite": float(phase_fractions[i, 2]),
            "martensite": float(phase_fractions[i, 3]),
            "austenite": float(phase_fractions[i, 4]),
        })

    # --- Step 4: Zone assignment ---
    zone_labels = None
    if weld_center is not None:
        zone_labels = assign_zones(
            mesh.nodes,
            np.asarray(weld_center, dtype=float),
            weld_radius,
            haz_width,
        )

    result = MultiscaleResult(
        cooling_rates=cooling_rates,
        yield_strength=yield_arr,
        elastic_modulus=modulus_arr,
        phase_fractions=phase_fractions,
        zone_labels=zone_labels,
        mean_yield=float(np.mean(yield_arr)),
        min_yield=float(np.min(yield_arr)),
        max_yield=float(np.max(yield_arr)),
    )

    logger.info(
        "Multiscale complete: yield=%.1f +/- %.1f MPa (range %.1f-%.1f)",
        result.mean_yield,
        float(np.std(yield_arr)),
        result.min_yield,
        result.max_yield,
    )

    return result


def multiscale_extensions(result: MultiscaleResult) -> dict[str, Any]:
    """Convert MultiscaleResult to a dict for WorkflowResult.extensions."""
    return {
        "microstructure": {
            **result.summary(),
            "phase_fractions_mean": {
                "ferrite": float(np.mean(result.phase_fractions[:, 0])),
                "pearlite": float(np.mean(result.phase_fractions[:, 1])),
                "bainite": float(np.mean(result.phase_fractions[:, 2])),
                "martensite": float(np.mean(result.phase_fractions[:, 3])),
                "austenite": float(np.mean(result.phase_fractions[:, 4])),
            },
        }
    }
