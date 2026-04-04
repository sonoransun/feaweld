"""Blodgett's "Design of Welded Structures" weld-as-a-line calculations.

Provides weld group section properties, stress calculations, AISC capacity
checks, and the Instantaneous Center of Rotation (ICR) method for
eccentrically loaded weld groups.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from feaweld.core.types import WeldGroupProperties, WeldGroupShape


# ---------------------------------------------------------------------------
# Weld group section properties
# ---------------------------------------------------------------------------


def weld_group_properties(
    shape: WeldGroupShape,
    d: float,
    b: float = 0.0,
) -> WeldGroupProperties:
    """Compute section properties for a standard weld group treated as a line.

    Parameters
    ----------
    shape : WeldGroupShape
        The standard weld-group configuration.
    d : float
        Primary dimension -- depth / length / diameter depending on *shape*.
    b : float, optional
        Secondary dimension -- width / flange width / spacing (default 0).

    Returns
    -------
    WeldGroupProperties
        Computed section properties.
    """
    if shape == WeldGroupShape.LINE:
        return _line(d)
    if shape == WeldGroupShape.PARALLEL:
        return _parallel(d, b)
    if shape == WeldGroupShape.C_SHAPE:
        return _c_shape(d, b)
    if shape == WeldGroupShape.L_SHAPE:
        return _l_shape(d, b)
    if shape == WeldGroupShape.BOX:
        return _box(d, b)
    if shape == WeldGroupShape.CIRCULAR:
        return _circular(d)
    if shape == WeldGroupShape.I_SHAPE:
        return _i_shape(d, b)
    if shape == WeldGroupShape.T_SHAPE:
        return _t_shape(d, b)
    if shape == WeldGroupShape.U_SHAPE:
        return _u_shape(d, b)
    raise ValueError(f"Unknown weld group shape: {shape}")


# ---- Individual shape helpers ------------------------------------------------


def _line(d: float) -> WeldGroupProperties:
    """Single straight line of length *d* along the y-axis."""
    A_w = d
    I_x = d ** 3 / 12.0
    S_x = d ** 2 / 6.0
    I_y = 0.0
    S_y = 0.0
    J_w = d ** 3 / 12.0
    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=0.0, y_bar=d / 2.0,
    )


def _parallel(d: float, b: float) -> WeldGroupProperties:
    """Two parallel lines of length *d* separated by spacing *b*."""
    A_w = 2.0 * d
    I_x = d * b ** 2 / 2.0
    S_x = d * b
    I_y = d ** 3 / 6.0
    S_y = d ** 2 / 3.0
    J_w = d ** 3 / 6.0 + d * b ** 2 / 2.0
    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=b / 2.0, y_bar=d / 2.0,
    )


def _c_shape(d: float, b: float) -> WeldGroupProperties:
    """Channel shape: two flanges of width *b* + web of depth *d*.

    Open side faces right; web on the left.
    Flanges run along x at y = 0 and y = d.
    """
    A_w = 2.0 * b + d

    # Centroid x: flanges each contribute b*(b/2), web contributes 0
    x_bar = b ** 2 / (2.0 * b + d)
    y_bar = d / 2.0

    # I_x about centroidal x-axis
    # Web: d^3/12 (line along y centred at y_bar)
    # Flanges: each at distance d/2 from centroid, length b -> b*(d/2)^2
    I_x = d ** 3 / 12.0 + 2.0 * b * (d / 2.0) ** 2
    # simplify: d^3/12 + b*d^2/2
    S_x = I_x / (d / 2.0)

    # I_y about centroidal y-axis
    # Web: line along y at x=0 -> contribution = d * x_bar^2
    # Top flange: line along x from 0 to b at y=d
    #   about centroid: integral of (x - x_bar)^2 dx from 0 to b
    #   = b^3/3 - b^2*x_bar + b*x_bar^2 = b^3/12 + b*(b/2 - x_bar)^2
    flange_Iy = b ** 3 / 12.0 + b * (b / 2.0 - x_bar) ** 2
    web_Iy = d * x_bar ** 2
    I_y = 2.0 * flange_Iy + web_Iy
    S_y_pos = I_y / (b - x_bar) if b > x_bar else 0.0
    S_y_neg = I_y / x_bar if x_bar > 0 else 0.0
    S_y = min(S_y_pos, S_y_neg) if S_y_pos > 0 and S_y_neg > 0 else max(S_y_pos, S_y_neg)

    J_w = I_x + I_y

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=x_bar, y_bar=y_bar,
    )


def _l_shape(d: float, b: float) -> WeldGroupProperties:
    """Angle shape: vertical leg *d* + horizontal leg *b*.

    Corner at origin; web goes up (y) and flange goes right (x).
    """
    A_w = b + d

    x_bar = b ** 2 / (2.0 * (b + d))
    y_bar = d ** 2 / (2.0 * (b + d))

    # I_x about centroidal x-axis
    # Vertical leg (length d along y from 0 to d):
    vert_Ix = d ** 3 / 12.0 + d * (d / 2.0 - y_bar) ** 2
    # Horizontal leg (length b along x at y = 0):
    horiz_Ix = b * y_bar ** 2
    I_x = vert_Ix + horiz_Ix
    S_x = I_x / max(y_bar, d - y_bar)

    # I_y about centroidal y-axis
    vert_Iy = d * x_bar ** 2
    horiz_Iy = b ** 3 / 12.0 + b * (b / 2.0 - x_bar) ** 2
    I_y = vert_Iy + horiz_Iy
    S_y = I_y / max(x_bar, b - x_bar)

    J_w = I_x + I_y

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=x_bar, y_bar=y_bar,
    )


def _box(d: float, b: float) -> WeldGroupProperties:
    """Rectangular (closed) weld group -- width *b*, depth *d*."""
    A_w = 2.0 * b + 2.0 * d

    I_x = d ** 2 * (3.0 * b + d) / 6.0
    S_x = I_x / (d / 2.0)  # = d*(3b+d)/3

    I_y = b ** 2 * (3.0 * d + b) / 6.0
    S_y = I_y / (b / 2.0)

    J_w = (b + d) ** 3 / 6.0

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=b / 2.0, y_bar=d / 2.0,
    )


def _circular(d: float) -> WeldGroupProperties:
    """Circular weld group of diameter *d*."""
    A_w = math.pi * d
    I_x = math.pi * d ** 3 / 8.0
    S_x = math.pi * d ** 2 / 4.0
    I_y = I_x
    S_y = S_x
    J_w = math.pi * d ** 3 / 4.0

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=d / 2.0, y_bar=d / 2.0,
    )


def _i_shape(d: float, b: float) -> WeldGroupProperties:
    """I-shape: two horizontal flanges of width *b* separated by depth *d*."""
    A_w = 2.0 * b
    I_x = b * d ** 2 / 2.0
    S_x = b * d
    I_y = b ** 3 / 6.0
    S_y = b ** 2 / 3.0
    J_w = I_x + I_y

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=b / 2.0, y_bar=d / 2.0,
    )


def _t_shape(d: float, b: float) -> WeldGroupProperties:
    """T-shape: horizontal flange *b* at bottom + vertical web *d*."""
    A_w = b + d

    x_bar = b / 2.0  # symmetric about x
    y_bar = d ** 2 / (2.0 * (b + d))

    # I_x
    # Flange (length b along x at y=0): b * y_bar^2
    # Web (length d along y from 0 to d): d^3/12 + d*(d/2 - y_bar)^2
    flange_Ix = b * y_bar ** 2
    web_Ix = d ** 3 / 12.0 + d * (d / 2.0 - y_bar) ** 2
    I_x = flange_Ix + web_Ix
    S_x = I_x / max(y_bar, d - y_bar)

    # I_y
    # Flange: b^3/12 (centred on x_bar = b/2)
    # Web: at x = b/2 (x_bar) -> 0 contribution
    I_y = b ** 3 / 12.0
    S_y = I_y / (b / 2.0)

    J_w = I_x + I_y

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=x_bar, y_bar=y_bar,
    )


def _u_shape(d: float, b: float) -> WeldGroupProperties:
    """U-shape (open top): bottom flange *b* + two webs of height *d*.

    Same geometry as C-shape but rotated -- open at top instead of right.
    """
    A_w = b + 2.0 * d

    x_bar = b / 2.0
    y_bar = d ** 2 / (b + 2.0 * d)

    # I_x about centroidal axis
    # Bottom flange at y=0: b * y_bar^2
    # Left web (length d along y): d^3/12 + d*(d/2 - y_bar)^2
    # Right web: same as left
    flange_Ix = b * y_bar ** 2
    web_Ix = d ** 3 / 12.0 + d * (d / 2.0 - y_bar) ** 2
    I_x = flange_Ix + 2.0 * web_Ix
    S_x = I_x / max(y_bar, d - y_bar)

    # I_y about centroidal axis
    flange_Iy = b ** 3 / 12.0
    web_Iy = d * (b / 2.0) ** 2
    I_y = flange_Iy + 2.0 * web_Iy
    S_y = I_y / (b / 2.0)

    J_w = I_x + I_y

    return WeldGroupProperties(
        A_w=A_w, S_x=S_x, S_y=S_y, J_w=J_w,
        I_x=I_x, I_y=I_y, x_bar=x_bar, y_bar=y_bar,
    )


# ---------------------------------------------------------------------------
# Weld stress calculations
# ---------------------------------------------------------------------------


def weld_stress(
    props: WeldGroupProperties,
    throat: float,
    P: float = 0.0,
    V: float = 0.0,
    M: float = 0.0,
    T: float = 0.0,
) -> dict[str, float]:
    """Compute weld line stresses using Blodgett's method.

    Parameters
    ----------
    props : WeldGroupProperties
        Section properties of the weld group.
    throat : float
        Effective weld throat thickness (mm).
    P : float
        Axial force (N) -- direct normal stress.
    V : float
        Shear force (N) -- direct shear stress.
    M : float
        Bending moment (N-mm).
    T : float
        Torsional moment (N-mm).

    Returns
    -------
    dict with keys: f_a, f_v, f_b, f_t, f_n, f_r, von_mises
    """
    f_a = P / (props.A_w * throat) if props.A_w * throat != 0 else 0.0
    f_v = V / (props.A_w * throat) if props.A_w * throat != 0 else 0.0
    f_b = M / (props.S_x * throat) if props.S_x * throat != 0 else 0.0

    # Torsion: maximum shear stress at outermost fibre
    # c = max distance from centroid to any point in the weld group
    # For a conservative estimate, use sqrt((x_bar_max)^2 + (y_bar_max)^2)
    # where the max distances are half-dimensions for symmetric shapes.
    # Since we already know I_x and I_y contribute to J_w, we can compute
    # c from I_x, I_y relationship.  For the general case:
    #   c_x = max distance in x from centroid
    #   c_y = max distance in y from centroid
    #   tau_torsion = T*c / J_w where c = sqrt(c_x^2 + c_y^2)
    # A safe general estimate: c = sqrt(S_x_dist^2 + S_y_dist^2) where
    # S_x_dist = I_x/S_x (= the extreme fibre distance used for S_x).
    if props.J_w > 0 and throat > 0:
        c_y = props.I_x / props.S_x if props.S_x > 0 else 0.0
        c_x = props.I_y / props.S_y if props.S_y > 0 else 0.0
        c = math.sqrt(c_x ** 2 + c_y ** 2)
        f_t = T * c / (props.J_w * throat)
    else:
        f_t = 0.0

    f_n = f_a + f_b  # normal stresses are additive
    f_r = math.sqrt(f_n ** 2 + (f_v + f_t) ** 2)

    # Von Mises: sigma_vm = sqrt(sigma_normal^2 + 3*tau^2)
    tau_total = f_v + f_t
    von_mises = math.sqrt(f_n ** 2 + 3.0 * tau_total ** 2)

    return {
        "f_a": f_a,
        "f_v": f_v,
        "f_b": f_b,
        "f_t": f_t,
        "f_n": f_n,
        "f_r": f_r,
        "von_mises": von_mises,
    }


# ---------------------------------------------------------------------------
# AISC capacity checks (LRFD and ASD)
# ---------------------------------------------------------------------------


def lrfd_capacity(
    throat: float,
    A_w: float,
    F_EXX: float = 483.0,
    filler: str | None = None,
) -> float:
    """LRFD design strength per AISC for fillet welds.

    phi * R_n = 0.75 * 0.60 * F_EXX * A_w * throat

    Parameters
    ----------
    throat : float
        Effective weld throat (mm).
    A_w : float
        Weld group length (mm).
    F_EXX : float
        Electrode tensile strength (MPa). Default 483 MPa = E70XX (70 ksi).
        Ignored if *filler* is provided.
    filler : str, optional
        AWS filler metal classification (e.g., "E7018", "E8018-C1").
        When provided, *F_EXX* is looked up from the filler metal database.

    Returns
    -------
    float
        LRFD design capacity (N).
    """
    if filler is not None:
        F_EXX = _resolve_filler_strength(filler)
    return 0.75 * 0.60 * F_EXX * A_w * throat


def asd_capacity(
    throat: float,
    A_w: float,
    F_EXX: float = 483.0,
    filler: str | None = None,
) -> float:
    """ASD allowable strength per AISC for fillet welds.

    R_n / Omega = 0.60 * F_EXX * A_w * throat / 2.0

    Parameters
    ----------
    throat : float
        Effective weld throat (mm).
    A_w : float
        Weld group length (mm).
    F_EXX : float
        Electrode tensile strength (MPa). Default 483 MPa = E70XX (70 ksi).
        Ignored if *filler* is provided.
    filler : str, optional
        AWS filler metal classification. See :func:`lrfd_capacity`.

    Returns
    -------
    float
        ASD allowable capacity (N).
    """
    if filler is not None:
        F_EXX = _resolve_filler_strength(filler)
    return 0.60 * F_EXX * A_w * throat / 2.0


def _resolve_filler_strength(classification: str) -> float:
    """Look up filler metal tensile strength from the database."""
    from feaweld.data.filler_metals import get_filler_metal
    fm = get_filler_metal(classification)
    return fm.tensile_mpa


# ---------------------------------------------------------------------------
# Instantaneous Center of Rotation (ICR) method
# ---------------------------------------------------------------------------


def icr_analysis(
    segments: list[dict[str, Any]],
    loads: dict[str, float],
    F_EXX: float = 483.0,
) -> dict[str, Any]:
    """Elastic Instantaneous Center of Rotation analysis for eccentric loads.

    Each weld segment is a short element. Under an eccentric load the weld
    group rotates about an instantaneous center.  This routine finds that
    centre by enforcing equilibrium and returns the coefficient *C* such that
    phi * R_n = C * throat * D_weld * l  (per AISC Table 8-4 style).

    Parameters
    ----------
    segments : list[dict]
        Each dict has keys ``x``, ``y`` (centroid coordinates, mm) and
        ``length`` (mm).  Optionally ``throat`` (mm); defaults to 1.
    loads : dict
        ``Px`` (N), ``Py`` (N), ``Mx`` (N-mm), ``My`` (N-mm), ``T`` (N-mm).
        The eccentricity can also be specified as ``ex``, ``ey`` (mm).
    F_EXX : float
        Electrode tensile strength (MPa).

    Returns
    -------
    dict
        ``icr_x``, ``icr_y`` -- location of instantaneous centre.
        ``forces`` -- list of per-segment force magnitudes.
        ``max_force`` -- maximum element force.
        ``C_coefficient`` -- AISC-style coefficient.
    """
    # Build arrays
    n = len(segments)
    xs = np.array([s["x"] for s in segments], dtype=np.float64)
    ys = np.array([s["y"] for s in segments], dtype=np.float64)
    lengths = np.array([s["length"] for s in segments], dtype=np.float64)
    throats = np.array([s.get("throat", 1.0) for s in segments], dtype=np.float64)

    # Total weld length
    total_length = float(np.sum(lengths))

    # Centroid of weld group
    cg_x = float(np.sum(xs * lengths) / total_length)
    cg_y = float(np.sum(ys * lengths) / total_length)

    # Applied loads (at weld group centroid frame)
    Px = loads.get("Px", 0.0)
    Py = loads.get("Py", 0.0)

    # Torsion about weld group centroid
    ex = loads.get("ex", 0.0)
    ey = loads.get("ey", 0.0)
    T_ext = loads.get("T", 0.0)
    # If eccentricities given, compute moment from forces
    T_ext += Py * ex - Px * ey

    # ---------------------------------------------------------------
    # Elastic ICR: for the elastic method the IC is located such that
    # the deformation of each element is proportional to its distance
    # from the IC.
    #
    # The equilibrium equations:
    #   sum(F_xi) + Px = 0
    #   sum(F_yi) + Py = 0
    #   sum(F_xi*(yi - y0) - F_yi*(xi - x0)) + T = 0
    #
    # where the force on element i is proportional to its distance from
    # the IC and directed perpendicular to the radius vector.
    # ---------------------------------------------------------------

    def _residuals(params: NDArray) -> NDArray:
        x0, y0 = params
        dx = xs - x0
        dy = ys - y0
        ri = np.sqrt(dx ** 2 + dy ** 2)
        ri_max = np.max(ri)
        if ri_max < 1e-12:
            return np.array([1e12, 1e12])

        # Force proportional to distance * length * throat
        fi = ri * lengths * throats  # relative force magnitudes
        # Direction: perpendicular to radius, i.e. (-dy, dx) / r
        with np.errstate(divide="ignore", invalid="ignore"):
            fx_i = np.where(ri > 1e-12, -fi * dy / ri, 0.0)
            fy_i = np.where(ri > 1e-12, fi * dx / ri, 0.0)

        # We need a scaling factor k so that equilibrium is satisfied.
        # Three equations, but we have only 2 unknowns (x0, y0) plus k.
        # Use moments to eliminate k.
        # sum(F_xi) = k * sum(fx_i)  etc.
        sum_fx = np.sum(fx_i)
        sum_fy = np.sum(fy_i)

        # Moment of element forces about the IC
        # M_i = F_xi*(yi - y0) - F_yi*(xi - x0) = fi/ri * (dx^2 + dy^2) = fi*ri
        sum_m = np.sum(fi * ri)  # always positive

        # We set k from moment equilibrium: k * sum_m = -T_ext
        # => k = -T_ext / sum_m  (T_ext positive = CCW)
        if abs(sum_m) < 1e-12:
            return np.array([1e12, 1e12])
        k = -T_ext / sum_m if abs(T_ext) > 1e-12 else 1.0

        # Force equilibrium residuals
        res_x = k * sum_fx + Px
        res_y = k * sum_fy + Py
        return np.array([res_x, res_y])

    # Initial guess: offset from centroid
    if abs(T_ext) > 1e-6:
        # Start opposite to the resultant force direction
        P_mag = math.sqrt(Px ** 2 + Py ** 2) + 1e-12
        guess_x = cg_x - Py / P_mag * total_length * 0.1
        guess_y = cg_y + Px / P_mag * total_length * 0.1
    else:
        guess_x = cg_x + 1.0
        guess_y = cg_y + 1.0

    sol = fsolve(_residuals, np.array([guess_x, guess_y]), full_output=True)
    x0, y0 = sol[0]

    # Compute final element forces
    dx = xs - x0
    dy = ys - y0
    ri = np.sqrt(dx ** 2 + dy ** 2)
    ri_max = np.max(ri) if np.max(ri) > 1e-12 else 1.0
    fi = ri * lengths * throats
    sum_m = float(np.sum(fi * ri))

    if abs(sum_m) > 1e-12 and abs(T_ext) > 1e-12:
        k = -T_ext / sum_m
    else:
        # Pure force, no eccentricity -- use direct distribution
        k = math.sqrt(Px ** 2 + Py ** 2) / (float(np.sum(fi)) + 1e-30)

    forces = np.abs(k) * fi
    max_force = float(np.max(forces))

    # C coefficient: phi*Rn = C * D * l   where D = weld size (sixteenths), l = length
    # We report C as: max_force / (total_length)
    C_coeff = max_force / total_length if total_length > 0 else 0.0

    return {
        "icr_x": float(x0),
        "icr_y": float(y0),
        "forces": forces.tolist(),
        "max_force": max_force,
        "C_coefficient": C_coeff,
    }
