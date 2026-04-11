"""S-N curve database for IIW, DNV-RP-C203, and ASME VIII Div 2 standards.

Each curve is returned as an :class:`~feaweld.core.types.SNCurve` comprising
one or more :class:`~feaweld.core.types.SNSegment` entries.
"""

from __future__ import annotations

import math

import numpy as np

from feaweld.core.types import SNCurve, SNSegment, SNStandard

# ---------------------------------------------------------------------------
# IIW FAT classes
# ---------------------------------------------------------------------------

_IIW_FAT_CLASSES: tuple[int, ...] = (
    36, 40, 45, 50, 56, 63, 71, 80, 90, 100, 112, 125, 140, 160,
)


def iiw_fat(fat_class: int) -> SNCurve:
    """Return IIW S-N curve for the given FAT class.

    IIW formulation (two-slope):
    - Segment 1 (m = 3): N = C1 / S^3,  C1 = FAT^3 * 2e6
      valid for S >= knee_stress
    - Segment 2 (m = 5): N = C2 / S^5,  C2 computed from continuity at
      the knee point (N = 1e7).
    - Cutoff at N = 1e8 (constant-amplitude fatigue limit).

    The knee point stress is: S_knee = FAT * (2e6 / 1e7)^(1/3)
    The cutoff stress (below which infinite life) is determined at N = 1e8
    from segment 2.

    Parameters
    ----------
    fat_class : int
        FAT class number (e.g. 90 for FAT 90).

    Returns
    -------
    SNCurve
    """
    if fat_class not in _IIW_FAT_CLASSES:
        raise ValueError(
            f"Unsupported IIW FAT class: {fat_class}. "
            f"Choose from {_IIW_FAT_CLASSES}"
        )

    FAT = float(fat_class)
    C1 = FAT ** 3 * 2.0e6  # N*S^3 = C1

    # Knee point at N = 1e7
    S_knee = FAT * (2.0e6 / 1.0e7) ** (1.0 / 3.0)

    # Continuity at knee: C2 / S_knee^5 = 1e7  =>  C2 = S_knee^5 * 1e7
    C2 = S_knee ** 5 * 1.0e7

    # Cutoff stress at N = 1e8
    S_cutoff = (C2 / 1.0e8) ** (1.0 / 5.0)

    segments = [
        SNSegment(m=3.0, C=C1, stress_threshold=S_knee),
        SNSegment(m=5.0, C=C2, stress_threshold=S_cutoff),
    ]

    return SNCurve(
        name=f"IIW FAT{fat_class}",
        standard=SNStandard.IIW,
        segments=segments,
        cutoff_cycles=1e8,
    )


# ---------------------------------------------------------------------------
# DNV-RP-C203
# ---------------------------------------------------------------------------

# Published log10(a) intercepts for two-slope curves (in seawater with CP
# or in-air; these are the in-air/CP values).
_DNV_DATA: dict[str, tuple[float, float]] = {
    "B1": (15.117, 17.146),
    "B2": (14.885, 16.856),
    "C":  (12.592, 16.320),
    "C1": (12.449, 16.081),
    "C2": (12.301, 15.835),
    "D":  (12.164, 15.606),
    "E":  (12.010, 15.350),
    "F":  (11.855, 15.091),
    "F1": (11.699, 14.832),
    "F3": (11.546, 14.576),
    "G":  (11.398, 14.330),
    "W1": (11.261, 14.101),
    "W2": (11.107, 13.855),
    "W3": (10.970, 13.617),
}


def dnv_curve(category: str) -> SNCurve:
    """Return DNV-RP-C203 S-N curve for the given category.

    The DNV formulation is:
        log10(N) = log10(a) - m * log10(S)
    which gives:
        N = 10^(log_a) / S^m  =>  C = 10^(log_a)

    Two slopes: m1 = 3 for N <= 1e7 and m2 = 5 for N > 1e7.

    Parameters
    ----------
    category : str
        DNV detail category, e.g. "D", "F1", "W3".

    Returns
    -------
    SNCurve
    """
    cat = category.upper()
    if cat not in _DNV_DATA:
        raise ValueError(
            f"Unknown DNV category '{category}'. "
            f"Choose from {sorted(_DNV_DATA.keys())}"
        )

    log_a1, log_a2 = _DNV_DATA[cat]
    m1 = 3.0
    m2 = 5.0
    C1 = 10.0 ** log_a1
    C2 = 10.0 ** log_a2

    # Knee point stress at N = 1e7 using segment 1
    # 1e7 = C1 / S_knee^3  =>  S_knee = (C1 / 1e7)^(1/3)
    S_knee = (C1 / 1.0e7) ** (1.0 / 3.0)

    # Cutoff at N = 1e8
    S_cutoff = (C2 / 1.0e8) ** (1.0 / 5.0)

    segments = [
        SNSegment(m=m1, C=C1, stress_threshold=S_knee),
        SNSegment(m=m2, C=C2, stress_threshold=S_cutoff),
    ]

    return SNCurve(
        name=f"DNV {cat}",
        standard=SNStandard.DNV,
        segments=segments,
        cutoff_cycles=1e8,
    )


# ---------------------------------------------------------------------------
# ASME VIII Division 2 (simplified polynomial S-N curves)
# ---------------------------------------------------------------------------

# ASME VIII Div 2, Table 3-F.1 -- Fatigue Design Curves
# These are piecewise log-log representations.  We store several segments
# that approximate the published design curves.
#
# Ferritic steel welded joints (structural stress basis):
#   The ASME design curve for ferritic steel can be approximated by a
#   two-slope model similar to IIW/DNV with m=3.13 and m=5 above the
#   knee.  The exact polynomial coefficients from Table 3-F.3.1 define
#   N as a function of equivalent structural stress S:
#       log10(N) = a0 + a1*x + a2*x^2 + ...  where x = log10(S)
#   For simplicity we use piecewise power-law segments calibrated to
#   key points on the published curve.
#
# Austenitic steel:  Similar but with different intercept.

_ASME_CURVES: dict[str, list[tuple[float, float, float]]] = {
    # (m, C, stress_threshold_MPa)
    "ferritic": [
        # High-stress regime (N < ~1e7): m ~= 3.13
        (3.13, 1.14e14, 47.0),
        # Low-stress regime (N > 1e7): m ~= 5.0
        (5.0, 6.10e17, 0.0),
    ],
    "austenitic": [
        (3.13, 1.95e14, 55.0),
        (5.0, 1.28e18, 0.0),
    ],
}


def asme_curve(material: str) -> SNCurve:
    """Return ASME VIII Division 2 fatigue design curve.

    Parameters
    ----------
    material : str
        ``"ferritic"`` or ``"austenitic"``.

    Returns
    -------
    SNCurve
    """
    mat = material.lower()
    if mat not in _ASME_CURVES:
        raise ValueError(
            f"Unknown ASME material '{material}'. "
            "Choose from 'ferritic' or 'austenitic'."
        )

    raw = _ASME_CURVES[mat]
    segments = [
        SNSegment(m=m, C=C, stress_threshold=s_thr)
        for m, C, s_thr in raw
    ]

    return SNCurve(
        name=f"ASME VIII Div2 - {mat}",
        standard=SNStandard.ASME,
        segments=segments,
        cutoff_cycles=1e11,  # ASME does not define a strict cutoff; use 1e11
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def get_sn_curve(standard: str, name: str) -> SNCurve:
    """Return an S-N curve by standard and name/category.

    Parameters
    ----------
    standard : str
        One of ``"iiw"``, ``"dnv"``, ``"asme"`` (case-insensitive).
    name : str
        Curve identifier:
        - IIW: FAT class as string, e.g. ``"90"`` or ``"FAT90"``.
        - DNV: detail category, e.g. ``"D"``, ``"F1"``.
        - ASME: material, e.g. ``"ferritic"``.

    Returns
    -------
    SNCurve
    """
    std = standard.lower()
    if std == "iiw":
        # Accept "90" or "FAT90"
        cleaned = name.upper().replace("FAT", "").strip()
        return iiw_fat(int(cleaned))
    if std == "dnv":
        return dnv_curve(name)
    if std == "asme":
        return asme_curve(name)
    raise ValueError(f"Unknown standard '{standard}'. Choose from iiw, dnv, asme.")


def life_with_scatter(
    curve: SNCurve,
    stress_range: float,
    scatter_std_log10_N: float = 0.2,
    n_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute N as a lognormal distribution around the deterministic S-N life.

    Standards like IIW FAT curves have a fixed log-N scatter (~0.2 in
    log10-N for welds, i.e. factor-of-~1.6 per standard deviation); we
    apply it as a multiplicative lognormal on the deterministic life.
    Returns a dict with keys: mean, std, p05, p50, p95.
    """
    N_det = curve.life(stress_range)
    if not math.isfinite(N_det) or N_det <= 0:
        return {
            "mean": N_det,
            "std": 0.0,
            "p05": N_det,
            "p50": N_det,
            "p95": N_det,
        }

    rng = np.random.default_rng(seed)
    # Multiplicative lognormal: log10(N) ~ N(log10(N_det), scatter_std_log10_N)
    # Convert log10 std to natural-log std for rng.lognormal.
    sigma_ln = scatter_std_log10_N * math.log(10.0)
    mu_ln = math.log(N_det)
    samples = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_samples)

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p05": float(np.percentile(samples, 5.0)),
        "p50": float(np.percentile(samples, 50.0)),
        "p95": float(np.percentile(samples, 95.0)),
    }


def life_with_scatter_stress(
    curve: SNCurve,
    stress_mean: float,
    stress_std: float,
    scatter_std_log10_N: float = 0.2,
    n_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-source UQ: propagate both stress-range and S-N scatter into life.

    Samples stress range from a normal distribution (truncated at >0) and
    S-N life from a lognormal scatter band around the deterministic curve.
    Returns a dict with keys: mean, std, p05, p50, p95.
    """
    rng = np.random.default_rng(seed)
    stress_samples = rng.normal(stress_mean, stress_std, size=n_samples)
    # Reject non-positive stress samples and resample as needed.
    stress_samples = np.clip(stress_samples, 1e-12, None)

    N_det = np.array([curve.life(float(s)) for s in stress_samples])
    finite = np.isfinite(N_det) & (N_det > 0)
    if not np.any(finite):
        return {
            "mean": float("inf"),
            "std": 0.0,
            "p05": float("inf"),
            "p50": float("inf"),
            "p95": float("inf"),
        }

    sigma_ln = scatter_std_log10_N * math.log(10.0)
    scatter = rng.lognormal(mean=0.0, sigma=sigma_ln, size=n_samples)
    samples = N_det * scatter
    samples = samples[finite]

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p05": float(np.percentile(samples, 5.0)),
        "p50": float(np.percentile(samples, 50.0)),
        "p95": float(np.percentile(samples, 95.0)),
    }


def get_sn_curve_by_detail(detail_number: int) -> SNCurve:
    """Get an IIW S-N curve by weld detail category number.

    Looks up the FAT class from the IIW weld detail database, then
    returns the corresponding S-N curve.

    Args:
        detail_number: IIW weld detail number (e.g., 100, 211, 413).

    Returns:
        SNCurve for the FAT class associated with the detail.
    """
    from feaweld.data.sn_curves.weld_details import get_weld_detail
    detail = get_weld_detail(detail_number)
    return iiw_fat(detail.fat_class)
