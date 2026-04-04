"""Feature engineering from FEA results for ML fatigue prediction.

Extracts physics-informed features from finite element analysis results and
weld line definitions that serve as inputs to machine-learning fatigue life
models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, WeldLineDefinition


# ---------------------------------------------------------------------------
# Feature data container
# ---------------------------------------------------------------------------


@dataclass
class FatigueFeatures:
    """A labelled feature matrix for ML training / prediction.

    Attributes
    ----------
    feature_names : list[str]
        Ordered feature column names.
    values : NDArray
        Shape ``(n_samples, n_features)``.
    target : NDArray | None
        ``log10(N)`` fatigue life values, shape ``(n_samples,)``, or *None*
        if targets are unknown.
    """

    feature_names: list[str]
    values: NDArray[np.float64]
    target: NDArray[np.float64] | None = None


# ---------------------------------------------------------------------------
# Canonical feature ordering
# ---------------------------------------------------------------------------

_STANDARD_FEATURES: list[str] = [
    "stress_range",
    "r_ratio",
    "plate_thickness",
    "structural_stress_membrane",
    "structural_stress_bending",
    "scf",
    "material_uts",
    "material_yield",
    "weld_toe_angle",
    "toe_radius",
    "misalignment",
    "residual_stress_ratio",
    "hotspot_stress",
]


def standard_feature_names() -> list[str]:
    """Return the canonical ordered list of feature names."""
    return list(_STANDARD_FEATURES)


# ---------------------------------------------------------------------------
# Single-analysis feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    results: FEAResults,
    weld_line: WeldLineDefinition,
    material_props: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract ML features from one FEA analysis.

    Parameters
    ----------
    results : FEAResults
        Completed FEA result set.
    weld_line : WeldLineDefinition
        The weld line of interest (provides node IDs and plate thickness).
    material_props : dict | None
        Optional material properties with keys such as ``"uts"``,
        ``"yield"``, ``"residual_stress"``.

    Returns
    -------
    dict[str, float]
        Feature name -> value mapping.  Missing features are omitted rather
        than set to NaN so callers can decide how to handle them.
    """

    feats: dict[str, float] = {}
    material_props = material_props or {}

    # --- Stress-based features ------------------------------------------
    if results.stress is not None:
        vm = results.stress.von_mises  # all nodes

        # Restrict to weld-line nodes when possible
        weld_node_ids = weld_line.node_ids
        max_id = vm.shape[0] - 1
        valid_ids = weld_node_ids[weld_node_ids <= max_id]

        if len(valid_ids) > 0:
            weld_vm = vm[valid_ids]
        else:
            weld_vm = vm

        max_stress = float(np.max(weld_vm))
        min_stress = float(np.min(weld_vm))
        stress_range = max_stress - min_stress

        feats["stress_range"] = stress_range

        # R-ratio (min/max); guard against division by zero
        if max_stress != 0:
            feats["r_ratio"] = min_stress / max_stress
        else:
            feats["r_ratio"] = 0.0

        # SCF: max weld-line stress / nominal (mean) stress
        nominal = float(np.mean(vm))
        if nominal > 0:
            feats["scf"] = max_stress / nominal
        else:
            feats["scf"] = 1.0

        # Hotspot stress (simple: max at weld toe)
        feats["hotspot_stress"] = max_stress

    # --- Geometric features ---------------------------------------------
    feats["plate_thickness"] = weld_line.plate_thickness

    # --- Structural stress decomposition (from metadata) ----------------
    meta = results.metadata
    if "structural_stress_membrane" in meta:
        feats["structural_stress_membrane"] = float(meta["structural_stress_membrane"])
    if "structural_stress_bending" in meta:
        feats["structural_stress_bending"] = float(meta["structural_stress_bending"])

    # --- Material properties --------------------------------------------
    if "uts" in material_props:
        feats["material_uts"] = float(material_props["uts"])
    if "yield" in material_props:
        feats["material_yield"] = float(material_props["yield"])

    # Residual stress ratio
    if "residual_stress" in material_props and "yield" in material_props:
        fy = material_props["yield"]
        if fy > 0:
            feats["residual_stress_ratio"] = material_props["residual_stress"] / fy

    # --- Weld geometry from metadata ------------------------------------
    if "weld_toe_angle" in meta:
        feats["weld_toe_angle"] = float(meta["weld_toe_angle"])
    if "toe_radius" in meta:
        feats["toe_radius"] = float(meta["toe_radius"])
    if "misalignment" in meta:
        feats["misalignment"] = float(meta["misalignment"])

    return feats


# ---------------------------------------------------------------------------
# Multi-analysis feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(
    feature_dicts: list[dict[str, float]],
    target_lives: list[float] | None = None,
) -> FatigueFeatures:
    """Combine multiple feature dicts into a :class:`FatigueFeatures` matrix.

    Missing features are filled with ``NaN``.

    Parameters
    ----------
    feature_dicts : list[dict[str, float]]
        One dict per analysis, as returned by :func:`extract_features`.
    target_lives : list[float] | None
        Raw fatigue lives *N*.  Stored as ``log10(N)`` in the target array.

    Returns
    -------
    FatigueFeatures
    """

    names = standard_feature_names()
    n = len(feature_dicts)
    m = len(names)

    values = np.full((n, m), np.nan, dtype=np.float64)
    for i, fd in enumerate(feature_dicts):
        for j, name in enumerate(names):
            if name in fd:
                values[i, j] = fd[name]

    target = None
    if target_lives is not None:
        target = np.log10(np.asarray(target_lives, dtype=np.float64))

    return FatigueFeatures(feature_names=names, values=values, target=target)
