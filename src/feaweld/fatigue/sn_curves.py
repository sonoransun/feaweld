"""S-N curve database for IIW, DNV-RP-C203, ASME VIII Div 2, EN 1993-1-9
(Eurocode 3), BS 7608, and AWS D1.1 standards.

Each curve is returned as an [SNCurve][feaweld.core.types.SNCurve] comprising
one or more [SNSegment][feaweld.core.types.SNSegment] entries.
"""

from __future__ import annotations

import math

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
# EN 1993-1-9 (Eurocode 3)
# ---------------------------------------------------------------------------


def ec3_curve(category: int) -> SNCurve:
    """Return EN 1993-1-9 (Eurocode 3) direct stress S-N curve.

    EC3 formulation (two-slope with constant-amplitude fatigue limit):

    - Segment 1 (m = 3): anchored at $N = 2 \\times 10^6$ where the stress
      range equals the detail category $\\Delta\\sigma_C$, valid down to
      the constant-amplitude fatigue limit at $N_D = 5 \\times 10^6$:
      $\\Delta\\sigma_D = (2/5)^{1/3} \\Delta\\sigma_C \\approx
      0.7368 \\Delta\\sigma_C$.
    - Segment 2 (m = 5): between $N_D$ and the cut-off at $N_L = 10^8$:
      $\\Delta\\sigma_L = (5/100)^{1/5} \\Delta\\sigma_D \\approx
      0.4047 \\Delta\\sigma_C$.

    Parameters
    ----------
    category : int
        Detail category (e.g. 90), the stress range in MPa giving a
        life of 2e6 cycles.

    Returns
    -------
    SNCurve
    """
    from feaweld.data.cache import get_cache

    data = get_cache().get("sn_curves/ec3")
    categories = data["categories"]
    if category not in categories:
        raise ValueError(
            f"Unsupported EC3 detail category: {category}. "
            f"Choose from {tuple(categories)}"
        )

    conv = data["conventions"]
    n_ref = float(conv["n_reference"])
    n_knee = float(conv["knee_cycles"])
    n_cutoff = float(conv["cutoff_cycles"])
    m1 = float(conv["m1"])
    m2 = float(conv["m2"])

    delta_sigma_c = float(category)
    C1 = delta_sigma_c ** m1 * n_ref

    # Constant-amplitude fatigue limit at N_D
    S_D = delta_sigma_c * (n_ref / n_knee) ** (1.0 / m1)

    # Continuity at the knee: C2 / S_D^m2 = N_D  =>  C2 = S_D^m2 * N_D
    C2 = S_D ** m2 * n_knee

    # Cut-off (variable-amplitude) limit at N_L
    S_L = (C2 / n_cutoff) ** (1.0 / m2)

    segments = [
        SNSegment(m=m1, C=C1, stress_threshold=S_D),
        SNSegment(m=m2, C=C2, stress_threshold=S_L),
    ]

    return SNCurve(
        name=f"EC3 detail category {category}",
        standard=SNStandard.EC3,
        segments=segments,
        cutoff_cycles=n_cutoff,
    )


# ---------------------------------------------------------------------------
# BS 7608
# ---------------------------------------------------------------------------


def bs7608_curve(class_name: str, *, mean_curve: bool = False) -> SNCurve:
    """Return BS 7608 S-N curve for the given weld quality class.

    BS 7608 formulation:
        log10(N) = log10(a) - m * log10(S)
    with the design curve at mean minus two standard deviations of
    log10(N).  A Haibach extension (slope m + 2, anchored by continuity)
    applies below the knee at N = 1e7; cut-off at N = 1e8.

    Parameters
    ----------
    class_name : str
        Weld quality class: B, C, D, E, F, F2, G, or W1
        (case-insensitive).
    mean_curve : bool
        If True, return the mean curve instead of the design
        (mean - 2 SD) curve.

    Returns
    -------
    SNCurve
    """
    from feaweld.data.cache import get_cache

    data = get_cache().get("sn_curves/bs7608")
    classes = data["classes"]
    cls = class_name.strip().upper()
    if cls not in classes:
        raise ValueError(
            f"Unknown BS 7608 class '{class_name}'. "
            f"Choose from {sorted(classes.keys())}"
        )

    entry = classes[cls]
    conv = data["conventions"]
    m1 = float(entry["m"])
    log_a = float(entry["log_a_mean"])
    if not mean_curve:
        log_a -= float(conv["design_sd_multiplier"]) * float(entry["sd"])

    n_knee = float(conv["knee_cycles"])
    n_cutoff = float(conv["cutoff_cycles"])

    C1 = 10.0 ** log_a

    # Knee point stress at N = 1e7 using segment 1
    S_knee = (C1 / n_knee) ** (1.0 / m1)

    # Haibach extension: slope m + 2, continuity at the knee
    m2 = m1 + 2.0
    C2 = S_knee ** m2 * n_knee

    # Cut-off at N = 1e8
    S_cutoff = (C2 / n_cutoff) ** (1.0 / m2)

    segments = [
        SNSegment(m=m1, C=C1, stress_threshold=S_knee),
        SNSegment(m=m2, C=C2, stress_threshold=S_cutoff),
    ]

    name = f"BS 7608 class {cls}"
    if mean_curve:
        name += " (mean)"

    return SNCurve(
        name=name,
        standard=SNStandard.BS7608,
        segments=segments,
        cutoff_cycles=n_cutoff,
    )


# ---------------------------------------------------------------------------
# AWS D1.1 / AISC
# ---------------------------------------------------------------------------


def aws_curve(category: str) -> SNCurve:
    """Return AWS D1.1 / AISC S-N curve for the given stress category.

    Single-slope (m = 3) curves with a constant-amplitude fatigue
    threshold F_TH per category, below which life is infinite.  The
    published SI form of the AISC/AWS equation is:
        F_SR(MPa) = (Cf * 329e8 / N)^(1/3)
    which gives C = Cf * 3.29e10 in the N * S^m = C convention.

    Parameters
    ----------
    category : str
        Stress category: A, B, B', C, D, E, or E'.  The primed
        categories are also accepted as ``"BP"`` / ``"EP"``
        (case-insensitive).

    Returns
    -------
    SNCurve
    """
    from feaweld.data.cache import get_cache

    data = get_cache().get("sn_curves/aws")
    categories = data["categories"]
    cat = category.strip().upper().replace("'", "P")
    if cat not in categories:
        raise ValueError(
            f"Unknown AWS category '{category}'. "
            f"Choose from {sorted(categories.keys())}"
        )

    conv = data["conventions"]
    m = float(conv["m"])
    entry = categories[cat]
    C = float(entry["Cf_ksi"]) * float(conv["si_factor"])
    f_th = float(entry["f_th_mpa"])

    segments = [SNSegment(m=m, C=C, stress_threshold=f_th)]

    display = cat[:-1] + "'" if cat.endswith("P") else cat
    return SNCurve(
        name=f"AWS D1.1 category {display}",
        standard=SNStandard.AWS,
        segments=segments,
        # Threshold cycle count: life at F_TH (infinite life below F_TH)
        cutoff_cycles=C / f_th ** m,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def get_sn_curve(standard: str, name: str) -> SNCurve:
    """Return an S-N curve by standard and name/category.

    Parameters
    ----------
    standard : str
        One of ``"iiw"``, ``"dnv"``, ``"asme"``, ``"ec3"`` (or
        ``"eurocode3"``), ``"bs7608"`` (or ``"bs"``), ``"aws"``
        (case-insensitive).
    name : str
        Curve identifier:
        - IIW: FAT class as string, e.g. ``"90"`` or ``"FAT90"``.
        - DNV: detail category, e.g. ``"D"``, ``"F1"``.
        - ASME: material, e.g. ``"ferritic"``.
        - EC3: detail category, e.g. ``"90"``.
        - BS 7608: weld class, e.g. ``"D"``, ``"F2"``.
        - AWS: stress category, e.g. ``"C"``, ``"BP"``.

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
    if std in ("ec3", "eurocode3"):
        return ec3_curve(int(name.strip()))
    if std in ("bs7608", "bs"):
        return bs7608_curve(name)
    if std == "aws":
        return aws_curve(name)
    raise ValueError(
        f"Unknown standard '{standard}'. "
        "Choose from iiw, dnv, asme, ec3, bs7608, aws."
    )


def parse_sn_spec(spec: str) -> SNCurve:
    """Parse a combined S-N curve spec string like ``"IIW_FAT90"``.

    Accepts ``"<standard>_<name>"`` (e.g. ``"IIW_FAT90"``, ``"DNV_D"``,
    ``"ASME_ferritic"``, ``"EC3_90"``, ``"BS7608_F2"``, ``"AWS_C"``) or a
    bare IIW FAT class (``"FAT90"`` / ``"90"``).

    Parameters
    ----------
    spec : str
        Combined curve specification.

    Returns
    -------
    SNCurve
    """
    if "_" in spec:
        standard, name = spec.split("_", 1)
        return get_sn_curve(standard.lower(), name)
    return get_sn_curve("iiw", spec)


def get_sn_curve_by_detail(detail_number: int) -> SNCurve:
    """Get an IIW S-N curve by weld detail category number.

    Looks up the FAT class from the IIW weld detail database, then
    returns the corresponding S-N curve.

    Parameters
    ----------
    detail_number : int
        IIW weld detail number (e.g., 100, 211, 413).

    Returns
    -------
    SNCurve
        SNCurve for the FAT class associated with the detail.
    """
    from feaweld.data.sn_curves.weld_details import get_weld_detail
    detail = get_weld_detail(detail_number)
    return iiw_fat(detail.fat_class)
