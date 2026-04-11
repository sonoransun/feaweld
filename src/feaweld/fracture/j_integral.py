"""Domain-form J-integral and interaction integral for 2D TRI3 meshes (Track F3).

Implements the area-integral (equivalent domain integral) form of Rice's
J-integral with a q-function weighting, plus the Stern-Becker interaction
integral against Williams-series auxiliary fields for K_I / K_II separation.

MVP scope
---------
* TRI3 plane-strain or plane-stress only.
* q-function evaluated at element centroids; domain contribution uses the
  element area and centroid gradient (``A_e * (...) * dq/dx_j``).
* Displacement gradients are reconstructed with standard CST shape
  function derivatives from the nodal displacement field carried on
  :class:`feaweld.core.types.FEAResults`.
* Stresses are taken from ``FEAResults.stress`` (nodal) and averaged over
  each element's three corners; strains are taken from
  ``FEAResults.strain`` when present, otherwise recomputed from the CST
  displacement gradient.

The interaction integral uses the unit-K_I or unit-K_II Williams near-tip
field in polar coordinates centred at the user-supplied ``crack_tip`` with
the crack pointing along the +x axis. If the physical crack is not
aligned with +x, rotate the mesh or the tip definition upstream — MVP does
not auto-detect the crack direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import ElementType, FEAResults, FEMesh


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class JResult:
    """Container for a single J-integral / interaction-integral evaluation."""

    J_value: float
    K_I: float
    K_II: float
    K_III: float
    T_stress: float = 0.0
    path_independence_residual: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# q-function helper
# ---------------------------------------------------------------------------


def _q_function(
    distances: NDArray, inner_r: float, outer_r: float
) -> NDArray:
    """Pyramid weighting function: 1 inside ``inner_r``, 0 outside ``outer_r``.

    Linear decay in the annulus. Matches the standard "plateau" shape
    used in the equivalent-domain-integral J formulation.
    """
    distances = np.asarray(distances, dtype=np.float64)
    q = np.ones_like(distances)
    if outer_r <= inner_r:
        # Degenerate: step function.
        q[distances > outer_r] = 0.0
        return q
    mask_decay = (distances >= inner_r) & (distances <= outer_r)
    mask_outside = distances > outer_r
    q[mask_decay] = (outer_r - distances[mask_decay]) / (outer_r - inner_r)
    q[mask_outside] = 0.0
    return q


# ---------------------------------------------------------------------------
# CST (TRI3) helpers
# ---------------------------------------------------------------------------


def _cst_area_and_dN(
    coords: NDArray,
) -> tuple[float, NDArray]:
    """Return (area, dN/dx) for a single TRI3 with node coordinates (3, 2).

    dN is shape (3, 2) giving ∂N_i/∂x and ∂N_i/∂y for the three nodes.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    # Signed area ×2
    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * twoA
    # Standard CST shape function derivatives (b_i, c_i)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    dN = np.stack([b, c], axis=1) / twoA
    return float(area), dN


def _collect_tri3_elements(mesh: FEMesh) -> NDArray:
    """Return the (n_elem, 3) connectivity for a TRI3 mesh, validating the type."""
    from feaweld.core.types import ElementType

    if mesh.element_type != ElementType.TRI3:
        raise NotImplementedError(
            "j_integral_2d currently supports only TRI3 meshes"
        )
    if mesh.elements.shape[1] != 3:
        raise ValueError(
            f"TRI3 connectivity must be (n, 3), got {mesh.elements.shape}"
        )
    return mesh.elements


# ---------------------------------------------------------------------------
# Elastic helpers
# ---------------------------------------------------------------------------


def compute_k_from_j_elastic(
    J: float, E: float, nu: float, plane_strain: bool = True
) -> float:
    """Return ``K = sqrt(E' * J)`` with the appropriate effective modulus.

    Plane strain: ``E' = E / (1 - nu^2)``.
    Plane stress: ``E' = E``.
    """
    E_eff = E / (1.0 - nu**2) if plane_strain else E
    return float(np.sqrt(max(J, 0.0) * E_eff))


# ---------------------------------------------------------------------------
# Williams-series auxiliary fields
# ---------------------------------------------------------------------------


def _williams_auxiliary_fields(
    r: NDArray,
    theta: NDArray,
    mode: str,
    E: float,
    nu: float,
    plane_strain: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Unit-K Williams near-tip field (displacement, stress, strain).

    Returned arrays:
        u_aux       : (n, 2)  -- (u_x, u_y)
        sigma_aux   : (n, 3)  -- (sigma_xx, sigma_yy, tau_xy)
        epsilon_aux : (n, 3)  -- (eps_xx, eps_yy, gamma_xy)

    The field is normalised so that the stress intensity factor of the
    returned field is exactly 1 (K_aux = 1). Plane strain by default.
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    # Guard against r = 0 (avoid NaNs at the tip itself).
    r_safe = np.where(r > 0.0, r, 1e-30)

    mu = E / (2.0 * (1.0 + nu))
    if plane_strain:
        kappa = 3.0 - 4.0 * nu
    else:
        kappa = (3.0 - nu) / (1.0 + nu)

    sqrt_r = np.sqrt(r_safe)
    ct2 = np.cos(theta / 2.0)
    st2 = np.sin(theta / 2.0)
    ct = np.cos(theta)
    st = np.sin(theta)

    if mode == "I":
        # Displacements (K_I = 1)
        ux = (1.0 / (2.0 * mu)) * sqrt_r / np.sqrt(2.0 * np.pi) * ct2 * (
            kappa - 1.0 + 2.0 * st2**2
        )
        uy = (1.0 / (2.0 * mu)) * sqrt_r / np.sqrt(2.0 * np.pi) * st2 * (
            kappa + 1.0 - 2.0 * ct2**2
        )
        # Stresses
        pref = 1.0 / np.sqrt(2.0 * np.pi * r_safe)
        sxx = pref * ct2 * (1.0 - st2 * np.sin(3.0 * theta / 2.0))
        syy = pref * ct2 * (1.0 + st2 * np.sin(3.0 * theta / 2.0))
        sxy = pref * ct2 * st2 * np.cos(3.0 * theta / 2.0)
    elif mode == "II":
        ux = (1.0 / (2.0 * mu)) * sqrt_r / np.sqrt(2.0 * np.pi) * st2 * (
            kappa + 1.0 + 2.0 * ct2**2
        )
        uy = -(1.0 / (2.0 * mu)) * sqrt_r / np.sqrt(2.0 * np.pi) * ct2 * (
            kappa - 1.0 - 2.0 * st2**2
        )
        pref = 1.0 / np.sqrt(2.0 * np.pi * r_safe)
        sxx = -pref * st2 * (2.0 + ct2 * np.cos(3.0 * theta / 2.0))
        syy = pref * st2 * ct2 * np.cos(3.0 * theta / 2.0)
        sxy = pref * ct2 * (1.0 - st2 * np.sin(3.0 * theta / 2.0))
    elif mode == "III":
        # Anti-plane shear — not supported in 2D plane solver; return zeros.
        ux = np.zeros_like(r_safe)
        uy = np.zeros_like(r_safe)
        sxx = np.zeros_like(r_safe)
        syy = np.zeros_like(r_safe)
        sxy = np.zeros_like(r_safe)
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'I', 'II', or 'III')")

    u = np.stack([ux, uy], axis=1)
    sigma = np.stack([sxx, syy, sxy], axis=1)

    # Plane compliance -> strains (2D)
    if plane_strain:
        eps_xx = ((1.0 - nu**2) / E) * sxx - (nu * (1.0 + nu) / E) * syy
        eps_yy = ((1.0 - nu**2) / E) * syy - (nu * (1.0 + nu) / E) * sxx
    else:
        eps_xx = (1.0 / E) * (sxx - nu * syy)
        eps_yy = (1.0 / E) * (syy - nu * sxx)
    gamma_xy = sxy / mu
    epsilon = np.stack([eps_xx, eps_yy, gamma_xy], axis=1)

    return u, sigma, epsilon


# ---------------------------------------------------------------------------
# Per-element kinematic / kinetic reconstruction
# ---------------------------------------------------------------------------


def _element_state_tri3(
    fea_results: FEAResults,
) -> dict:
    """Pre-compute per-element centroids, areas, dN, grad(u), stress, strain.

    Returns dict with keys:
        centroids     : (n_el, 2)
        areas         : (n_el,)
        dN            : (n_el, 3, 2)
        grad_u        : (n_el, 2, 2)  -- ∂u_i/∂x_j
        sigma         : (n_el, 3)     -- (σ_xx, σ_yy, τ_xy) at centroid
        strain        : (n_el, 3)     -- (ε_xx, ε_yy, γ_xy) at centroid
        W             : (n_el,)       -- strain-energy density
    """
    mesh = fea_results.mesh
    elements = _collect_tri3_elements(mesh)
    n_el = elements.shape[0]
    nodes_xy = mesh.nodes[:, :2]

    disp = fea_results.displacement
    if disp is None:
        raise ValueError("FEAResults.displacement is required for j_integral_2d")
    u_xy = disp[:, :2]

    stress_nodal = None
    if fea_results.stress is not None:
        stress_nodal = fea_results.stress.values  # (n_nodes, 6)

    strain_nodal = fea_results.strain  # (n_nodes, 6) or None

    centroids = np.zeros((n_el, 2))
    areas = np.zeros(n_el)
    dN_all = np.zeros((n_el, 3, 2))
    grad_u = np.zeros((n_el, 2, 2))
    sigma_e = np.zeros((n_el, 3))
    eps_e = np.zeros((n_el, 3))

    for e in range(n_el):
        conn = elements[e]
        xy = nodes_xy[conn]
        area, dN = _cst_area_and_dN(xy)
        areas[e] = area
        dN_all[e] = dN
        centroids[e] = xy.mean(axis=0)

        ue = u_xy[conn]  # (3, 2)
        # grad_u[i, j] = sum_a dN_a/dx_j * u_a_i
        grad_u[e] = ue.T @ dN  # (2, 2)

        if stress_nodal is not None:
            s_avg = stress_nodal[conn].mean(axis=0)  # 6 components Voigt
            sigma_e[e, 0] = s_avg[0]  # σ_xx
            sigma_e[e, 1] = s_avg[1]  # σ_yy
            sigma_e[e, 2] = s_avg[3]  # τ_xy (Voigt index 3)

        if strain_nodal is not None:
            eps_avg = strain_nodal[conn].mean(axis=0)
            eps_e[e, 0] = eps_avg[0]
            eps_e[e, 1] = eps_avg[1]
            eps_e[e, 2] = eps_avg[3]  # engineering γ_xy
        else:
            # Small-strain symmetric part of grad u.
            eps_e[e, 0] = grad_u[e, 0, 0]
            eps_e[e, 1] = grad_u[e, 1, 1]
            eps_e[e, 2] = grad_u[e, 0, 1] + grad_u[e, 1, 0]

    W = 0.5 * (
        sigma_e[:, 0] * eps_e[:, 0]
        + sigma_e[:, 1] * eps_e[:, 1]
        + sigma_e[:, 2] * eps_e[:, 2]
    )

    return dict(
        centroids=centroids,
        areas=areas,
        dN=dN_all,
        grad_u=grad_u,
        sigma=sigma_e,
        strain=eps_e,
        W=W,
    )


# ---------------------------------------------------------------------------
# q-function gradient at element level
# ---------------------------------------------------------------------------


def _element_q_and_grad(
    centroids: NDArray,
    dN: NDArray,
    elements: NDArray,
    nodes_xy: NDArray,
    crack_tip: NDArray,
    inner_r: float,
    outer_r: float,
) -> tuple[NDArray, NDArray]:
    """Compute q at centroid and dq/dx at each element.

    q is defined per node from its distance to the crack tip; ∂q/∂x_j is
    reconstructed on each element as sum_a dN_a/dx_j * q_a (linear CST
    interpolation of the nodal q field). Returns:
        q_centroid : (n_el,)
        grad_q     : (n_el, 2)
    """
    node_r = np.linalg.norm(nodes_xy - crack_tip[None, :2], axis=1)
    q_node = _q_function(node_r, inner_r, outer_r)

    n_el = elements.shape[0]
    grad_q = np.zeros((n_el, 2))
    q_centroid = np.zeros(n_el)
    for e in range(n_el):
        conn = elements[e]
        q_e = q_node[conn]
        # Linear CST interpolation: q at centroid = mean of nodal q.
        q_centroid[e] = q_e.mean()
        grad_q[e] = q_e @ dN[e]  # (2,)
    return q_centroid, grad_q


# ---------------------------------------------------------------------------
# J-integral (area/domain form)
# ---------------------------------------------------------------------------


def j_integral_2d(
    fea_results: FEAResults,
    crack_tip: NDArray,
    q_function_radius: float,
    mode: Literal["elastic_plane_strain", "elastic_plane_stress"] = "elastic_plane_strain",
    E: float = 210000.0,
    nu: float = 0.3,
) -> JResult:
    """Compute J via the domain (equivalent-area) integral on a 2D TRI3 mesh.

    The integral

        J = ∫_A ( σ_ij  ∂u_i/∂x_1  -  W δ_{1j} ) ∂q/∂x_j  dA

    is approximated as a sum over elements using the centroid stress /
    strain-energy density and the linear q-function reconstructed from
    nodal distances to the crack tip. ``q_function_radius`` is the outer
    radius of the pyramid; the inner radius (plateau) is set to 40 %% of
    the outer by default to give a smooth annulus of contribution.

    Returns a :class:`JResult` with J, K_I from ``sqrt(E' J)``, and
    ``K_II = K_III = 0``. Use :func:`interaction_integral` to split
    mixed-mode contributions.
    """
    crack_tip = np.asarray(crack_tip, dtype=np.float64)
    if crack_tip.ndim != 1 or crack_tip.shape[0] not in (2, 3):
        raise ValueError("crack_tip must be shape (2,) or (3,)")
    crack_tip = np.array([crack_tip[0], crack_tip[1]], dtype=np.float64)

    plane_strain = mode == "elastic_plane_strain"

    state = _element_state_tri3(fea_results)
    mesh = fea_results.mesh
    elements = mesh.elements
    nodes_xy = mesh.nodes[:, :2]

    outer_r = float(q_function_radius)
    inner_r = 0.4 * outer_r

    q_centroid, grad_q = _element_q_and_grad(
        state["centroids"],
        state["dN"],
        elements,
        nodes_xy,
        crack_tip,
        inner_r,
        outer_r,
    )

    # Build the integrand at each element centroid:
    # integrand_j = σ_ij ∂u_i/∂x_1  - W δ_{1j}
    sigma = state["sigma"]  # (n_el, 3)
    grad_u = state["grad_u"]  # (n_el, 2, 2)
    W = state["W"]
    areas = state["areas"]

    # σ as (n_el, 2, 2)
    sig2 = np.zeros((sigma.shape[0], 2, 2))
    sig2[:, 0, 0] = sigma[:, 0]
    sig2[:, 1, 1] = sigma[:, 1]
    sig2[:, 0, 1] = sigma[:, 2]
    sig2[:, 1, 0] = sigma[:, 2]

    # t_j = σ_ij ∂u_i/∂x_1 summed over i
    # (n_el, 2) where second axis is j
    du_dx1 = grad_u[:, :, 0]  # (n_el, 2)  (∂u_0/∂x_1, ∂u_1/∂x_1)
    t_j = np.einsum("eij,ei->ej", sig2, du_dx1)
    # Subtract W δ_{1j} — only j=0 component (x direction) sees W.
    t_j[:, 0] -= W

    integrand = np.einsum("ej,ej->e", t_j, grad_q)  # (n_el,)
    J = float(np.sum(areas * integrand))

    K_I = compute_k_from_j_elastic(J, E, nu, plane_strain=plane_strain)

    return JResult(
        J_value=J,
        K_I=K_I,
        K_II=0.0,
        K_III=0.0,
        T_stress=0.0,
        path_independence_residual=0.0,
        metadata={
            "inner_radius": inner_r,
            "outer_radius": outer_r,
            "n_elements_in_domain": int(np.sum(q_centroid > 0.0)),
            "mode": mode,
        },
    )


# ---------------------------------------------------------------------------
# Interaction integral
# ---------------------------------------------------------------------------


def interaction_integral(
    fea_results: FEAResults,
    crack_tip: NDArray,
    q_function_radius: float,
    auxiliary_mode: Literal["I", "II", "III"] = "I",
    E: float = 210000.0,
    nu: float = 0.3,
) -> JResult:
    """Stern-Becker interaction integral for K separation.

    Uses the Williams-series auxiliary field for the requested mode and
    extracts the corresponding physical K. The auxiliary field is
    normalised to K_aux = 1 so that

        I = (2 / E') * K_physical * K_aux   =>   K_phys = 0.5 * E' * I.
    """
    crack_tip = np.asarray(crack_tip, dtype=np.float64)
    if crack_tip.ndim != 1 or crack_tip.shape[0] not in (2, 3):
        raise ValueError("crack_tip must be shape (2,) or (3,)")
    crack_tip = np.array([crack_tip[0], crack_tip[1]], dtype=np.float64)

    plane_strain = True  # MVP: plane-strain interaction integral
    E_eff = E / (1.0 - nu**2) if plane_strain else E

    state = _element_state_tri3(fea_results)
    mesh = fea_results.mesh
    elements = mesh.elements
    nodes_xy = mesh.nodes[:, :2]

    outer_r = float(q_function_radius)
    inner_r = 0.4 * outer_r

    _, grad_q = _element_q_and_grad(
        state["centroids"],
        state["dN"],
        elements,
        nodes_xy,
        crack_tip,
        inner_r,
        outer_r,
    )

    # Evaluate the auxiliary field at element centroids in polar coords
    # relative to the crack tip with +x axis = crack line.
    rel = state["centroids"] - crack_tip[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    u_aux, sig_aux, eps_aux = _williams_auxiliary_fields(
        r, theta, auxiliary_mode, E, nu, plane_strain=plane_strain
    )

    # We need grad(u_aux) with respect to spatial (x, y). Use a finite
    # difference in polar-converted cartesian coordinates: bump each
    # element centroid by +/- h in x and y and re-evaluate. Since the
    # Williams field is analytic, central differencing is accurate.
    h = max(0.002 * outer_r, 1e-6)
    grad_u_aux = np.zeros((rel.shape[0], 2, 2))  # grad_u_aux[e, i, j] = ∂u_i/∂x_j
    for j_dir in range(2):
        delta = np.zeros(2)
        delta[j_dir] = h
        rp = rel + delta[None, :]
        rm = rel - delta[None, :]
        r_p = np.linalg.norm(rp, axis=1)
        t_p = np.arctan2(rp[:, 1], rp[:, 0])
        r_m = np.linalg.norm(rm, axis=1)
        t_m = np.arctan2(rm[:, 1], rm[:, 0])
        u_p, _, _ = _williams_auxiliary_fields(
            r_p, t_p, auxiliary_mode, E, nu, plane_strain=plane_strain
        )
        u_m, _, _ = _williams_auxiliary_fields(
            r_m, t_m, auxiliary_mode, E, nu, plane_strain=plane_strain
        )
        grad_u_aux[:, :, j_dir] = (u_p - u_m) / (2.0 * h)

    # FE fields at centroid
    sigma_fe = state["sigma"]  # (n_el, 3)
    eps_fe = state["strain"]   # (n_el, 3)  -- γ_xy engineering
    grad_u_fe = state["grad_u"]  # (n_el, 2, 2)
    areas = state["areas"]

    # Build σ_fe (n_el, 2, 2) and σ_aux (n_el, 2, 2)
    def _voigt_to_tensor(v: NDArray) -> NDArray:
        t = np.zeros((v.shape[0], 2, 2))
        t[:, 0, 0] = v[:, 0]
        t[:, 1, 1] = v[:, 1]
        t[:, 0, 1] = v[:, 2]
        t[:, 1, 0] = v[:, 2]
        return t

    sig_fe_t = _voigt_to_tensor(sigma_fe)
    sig_aux_t = _voigt_to_tensor(sig_aux)

    # W_int = σ_ij^fe ε_ij^aux
    # eps_fe and eps_aux store (ε_xx, ε_yy, γ_xy); tensor component is ε_xy = γ_xy / 2.
    W_int = (
        sigma_fe[:, 0] * eps_aux[:, 0]
        + sigma_fe[:, 1] * eps_aux[:, 1]
        + sigma_fe[:, 2] * eps_aux[:, 2]  # τ_xy * γ_xy == 2 * τ_xy * ε_xy
    )

    # Integrand per element (vector in j):
    #   (σ_ij^fe  ∂u_i^aux/∂x_1  +  σ_ij^aux  ∂u_i^fe/∂x_1  -  W_int δ_{1j}) ∂q/∂x_j
    du_aux_dx1 = grad_u_aux[:, :, 0]  # (n_el, 2)
    du_fe_dx1 = grad_u_fe[:, :, 0]    # (n_el, 2)

    term1 = np.einsum("eij,ei->ej", sig_fe_t, du_aux_dx1)
    term2 = np.einsum("eij,ei->ej", sig_aux_t, du_fe_dx1)
    t_j = term1 + term2
    t_j[:, 0] -= W_int

    integrand = np.einsum("ej,ej->e", t_j, grad_q)
    I_value = float(np.sum(areas * integrand))

    if auxiliary_mode == "III":
        K_mode = 0.0
    else:
        K_mode = 0.5 * E_eff * I_value

    K_I = K_mode if auxiliary_mode == "I" else 0.0
    K_II = K_mode if auxiliary_mode == "II" else 0.0
    K_III = K_mode if auxiliary_mode == "III" else 0.0

    # Equivalent J from the two K's that we do know (just this one):
    J_equiv = (K_mode**2) / E_eff if auxiliary_mode in ("I", "II") else 0.0

    return JResult(
        J_value=J_equiv,
        K_I=K_I,
        K_II=K_II,
        K_III=K_III,
        T_stress=0.0,
        path_independence_residual=0.0,
        metadata={
            "I_value": I_value,
            "auxiliary_mode": auxiliary_mode,
            "inner_radius": inner_r,
            "outer_radius": outer_r,
        },
    )


# ---------------------------------------------------------------------------
# 3D J-integral along a crack front (Track F5)
# ---------------------------------------------------------------------------


def _frenet_frame_from_tangent(
    tangent: NDArray, plate_normal_hint: NDArray | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """Build a right-handed orthonormal triad (T, N, B) from a tangent.

    ``N`` is chosen as the component of ``plate_normal_hint`` perpendicular
    to ``T``. If the hint is unavailable or parallel to ``T`` an arbitrary
    stable perpendicular is chosen.
    """
    t = np.asarray(tangent, dtype=np.float64)
    n_t = float(np.linalg.norm(t))
    if n_t < 1e-15:
        raise ValueError("tangent vector has zero length")
    t = t / n_t

    if plate_normal_hint is not None:
        h = np.asarray(plate_normal_hint, dtype=np.float64)
        n_h = float(np.linalg.norm(h))
        if n_h > 1e-15:
            h = h / n_h
            proj = h - np.dot(h, t) * t
            if float(np.linalg.norm(proj)) > 1e-10:
                n = proj / float(np.linalg.norm(proj))
                b = np.cross(t, n)
                b = b / float(np.linalg.norm(b))
                return t, n, b

    # Fallback: pick the axis most orthogonal to t, then Gram-Schmidt.
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n = ref - np.dot(ref, t) * t
    n = n / float(np.linalg.norm(n))
    b = np.cross(t, n)
    b = b / float(np.linalg.norm(b))
    return t, n, b


def _rotate_voigt_stress(voigt: NDArray, R: NDArray) -> NDArray:
    """Rotate a 6-component Voigt stress (σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz) by R.

    ``R`` is a (3, 3) rotation that expresses local-frame axes as columns
    of the global frame (i.e. local = R.T @ global). Accepts (n, 6) or (6,).
    """
    v = np.atleast_2d(voigt)
    n = v.shape[0]
    out = np.zeros((n, 6))
    for i in range(n):
        s = np.array(
            [
                [v[i, 0], v[i, 3], v[i, 5]],
                [v[i, 3], v[i, 1], v[i, 4]],
                [v[i, 5], v[i, 4], v[i, 2]],
            ]
        )
        s_local = R.T @ s @ R
        out[i, 0] = s_local[0, 0]
        out[i, 1] = s_local[1, 1]
        out[i, 2] = s_local[2, 2]
        out[i, 3] = s_local[0, 1]
        out[i, 4] = s_local[1, 2]
        out[i, 5] = s_local[0, 2]
    if voigt.ndim == 1:
        return out[0]
    return out


def _rotate_voigt_strain(voigt: NDArray, R: NDArray) -> NDArray:
    """Rotate a 6-component Voigt strain (ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz) by R."""
    v = np.atleast_2d(voigt)
    n = v.shape[0]
    out = np.zeros((n, 6))
    for i in range(n):
        e = np.array(
            [
                [v[i, 0], 0.5 * v[i, 3], 0.5 * v[i, 5]],
                [0.5 * v[i, 3], v[i, 1], 0.5 * v[i, 4]],
                [0.5 * v[i, 5], 0.5 * v[i, 4], v[i, 2]],
            ]
        )
        e_local = R.T @ e @ R
        out[i, 0] = e_local[0, 0]
        out[i, 1] = e_local[1, 1]
        out[i, 2] = e_local[2, 2]
        out[i, 3] = 2.0 * e_local[0, 1]
        out[i, 4] = 2.0 * e_local[1, 2]
        out[i, 5] = 2.0 * e_local[0, 2]
    if voigt.ndim == 1:
        return out[0]
    return out


def _j_domain_2d_slice(
    coords_xy: NDArray,
    u_xy: NDArray,
    sigma_xy_voigt: NDArray,
    strain_xy_voigt: NDArray | None,
    q_node: NDArray,
    R_out: float,
) -> float:
    """Approximate a 2D equivalent-domain J-integral on a 2D point cloud slice.

    The slice is a scattered set of points living in a local (x, y) frame
    (x along the crack line, y perpendicular). Since we don't have a
    connectivity on the slice we use a simple quadrature:

        J ≈ Σ_k ( σ_ij^k  ∂u_i/∂x_1^k  -  W^k δ_{1j} ) ∂q/∂x_j^k  dA_k

    where ``∂u/∂x_1`` and ``∂q/∂x_j`` are reconstructed by a local
    least-squares plane fit over each point's k nearest neighbours, and
    ``dA_k`` is 1/density estimated from the neighbourhood radius.

    Returns the scalar J for the slice.
    """
    n = coords_xy.shape[0]
    if n < 6:
        return 0.0

    from scipy.spatial import cKDTree

    tree = cKDTree(coords_xy)
    # Use k nearest neighbours (including self) for local least-squares.
    k_nn = min(n, 8)
    dists, idxs = tree.query(coords_xy, k=k_nn)

    J = 0.0

    # Effective area per point: pi * R_out^2 / n_inside (very coarse),
    # but actually we use a Voronoi-like "area per point" approximation:
    # area_k ≈ pi * r_k^2 where r_k is the mean neighbour distance / 2.
    mean_nb_radius = dists[:, -1].mean() if k_nn > 1 else 1.0

    for p in range(n):
        nb = idxs[p]
        xy_nb = coords_xy[nb]
        # Local coordinates centered at point p.
        dx = xy_nb - coords_xy[p]
        # Ax = b for each scalar quantity, where x = [value, dval/dx, dval/dy].
        A = np.column_stack([np.ones(len(nb)), dx[:, 0], dx[:, 1]])
        # Guard against singular systems (points collinear).
        try:
            pinv = np.linalg.pinv(A)
        except np.linalg.LinAlgError:
            continue

        # Gradients of u_x, u_y
        grad_ux = pinv @ u_xy[nb, 0]  # [u_x, du_x/dx, du_x/dy]
        grad_uy = pinv @ u_xy[nb, 1]
        dux_dx = grad_ux[1]
        dux_dy = grad_ux[2]
        duy_dx = grad_uy[1]
        duy_dy = grad_uy[2]

        # Gradient of q
        grad_q_fit = pinv @ q_node[nb]
        dq_dx = grad_q_fit[1]
        dq_dy = grad_q_fit[2]

        # Stress at point p (already in the local frame, 2D component slice)
        sxx = sigma_xy_voigt[p, 0]
        syy = sigma_xy_voigt[p, 1]
        sxy = sigma_xy_voigt[p, 3]

        # Strain-energy density
        if strain_xy_voigt is not None:
            exx = strain_xy_voigt[p, 0]
            eyy = strain_xy_voigt[p, 1]
            gxy = strain_xy_voigt[p, 3]
        else:
            exx = dux_dx
            eyy = duy_dy
            gxy = dux_dy + duy_dx

        W = 0.5 * (sxx * exx + syy * eyy + sxy * gxy)

        # t_j = σ_ij ∂u_i/∂x_1 (2D slice, i, j ∈ {0,1}; ∂/∂x_1 is along
        # the crack line which we take as local x).
        t_x = sxx * dux_dx + sxy * duy_dx
        t_y = sxy * dux_dx + syy * duy_dx
        # Subtract W δ_{1j}
        t_x -= W

        integrand = t_x * dq_dx + t_y * dq_dy

        # Local "area" element — nearest-neighbour Voronoi estimate.
        r_local = max(dists[p, 1], mean_nb_radius * 0.25) if k_nn > 1 else mean_nb_radius
        dA = np.pi * (r_local**2)

        J += integrand * dA

    return float(J)


def j_integral_3d(
    fea_results: FEAResults,
    crack_front: "WeldPath",  # noqa: F821  (forward reference to avoid circular import)
    q_function_radius: float,
    n_front_samples: int = 20,
    E: float = 210000.0,
    nu: float = 0.3,
) -> list[JResult]:
    """Compute J along a 3D crack front.

    The crack front is parameterized by arc length; at each of
    ``n_front_samples`` points we:

    1. Establish a local Frenet frame (tangent along the front, normal
       into the plate, binormal along the crack face).
    2. Evaluate an equivalent-domain J-integral in a plane perpendicular
       to the front tangent using mesh nodes within ``q_function_radius``
       of the sample point.
    3. Store a :class:`JResult` per sample.

    MVP simplifications:

    * Restricted to ``TET4`` / ``TET10`` meshes.
    * No true 3D q-function. At each sample we take the in-range nodes,
      project them and their stresses/strains/displacements into the
      local 2D crack-plane frame, and run a scattered-point domain
      integral on that slice.
    * Aggregation (max, average along front) is the caller's job.
    """
    mesh = fea_results.mesh
    if mesh.element_type not in (ElementType.TET4, ElementType.TET10):
        raise NotImplementedError(
            "j_integral_3d currently supports only TET4 / TET10 meshes, "
            f"got {mesh.element_type}"
        )

    if fea_results.displacement is None:
        raise ValueError("FEAResults.displacement is required for j_integral_3d")
    if fea_results.stress is None:
        raise ValueError("FEAResults.stress is required for j_integral_3d")

    nodes = mesh.nodes
    if nodes.shape[1] != 3:
        raise ValueError("j_integral_3d requires 3D node coordinates")

    disp = fea_results.displacement[:, :3]
    stress_voigt = fea_results.stress.values  # (n_nodes, 6)
    strain_voigt = fea_results.strain  # (n_nodes, 6) or None

    from scipy.spatial import cKDTree

    tree = cKDTree(nodes)

    total_s = float(crack_front.arc_length())
    if total_s <= 0.0:
        raise ValueError("crack_front has zero arc length")

    # Sample positions along the front (exclude the absolute ends slightly
    # to avoid boundary effects).
    if n_front_samples < 1:
        raise ValueError("n_front_samples must be >= 1")
    if n_front_samples == 1:
        s_values = np.array([0.5 * total_s])
    else:
        s_values = np.linspace(0.0, total_s, n_front_samples)

    outer_r = float(q_function_radius)
    inner_r = 0.4 * outer_r

    results: list[JResult] = []

    # Plate-normal hint: pick the sampled binormal from the crack front
    # (curvature-dependent); fall back to +z if degenerate.
    for s in s_values:
        pt = np.asarray(crack_front.evaluate_s(float(s)), dtype=np.float64)
        if pt.ndim > 1:
            pt = pt.reshape(-1)

        # Tangent at this point via a finite difference on arc length.
        eps_s = max(1e-6, 1e-4 * total_s)
        s_plus = min(total_s, float(s) + eps_s)
        s_minus = max(0.0, float(s) - eps_s)
        p_plus = np.asarray(crack_front.evaluate_s(s_plus), dtype=np.float64).reshape(-1)
        p_minus = np.asarray(crack_front.evaluate_s(s_minus), dtype=np.float64).reshape(-1)
        tangent_vec = p_plus - p_minus
        if float(np.linalg.norm(tangent_vec)) < 1e-15:
            tangent_vec = np.array([1.0, 0.0, 0.0])

        # Plate normal hint: use +z (typical plate orientation). The
        # result is insensitive to the exact in-plane rotation.
        T_vec, N_vec, B_vec = _frenet_frame_from_tangent(
            tangent_vec, plate_normal_hint=np.array([0.0, 0.0, 1.0])
        )
        # Local frame convention for the 2D slice:
        #   local x  = N_vec  (crack-line direction, +x = into material)
        #   local y  = B_vec  (crack-face normal)
        #   local z  = T_vec  (along the front; slice normal)
        R_local = np.column_stack([N_vec, B_vec, T_vec])  # (3, 3)

        # Find all nodes within q_function_radius of the sample point.
        idx = tree.query_ball_point(pt, r=outer_r)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size < 8:
            # Not enough nodes for a meaningful domain integral.
            results.append(
                JResult(
                    J_value=0.0,
                    K_I=0.0,
                    K_II=0.0,
                    K_III=0.0,
                    T_stress=0.0,
                    path_independence_residual=0.0,
                    metadata={
                        "s": float(s),
                        "position": pt.tolist(),
                        "n_nodes_in_slice": int(idx.size),
                        "note": "insufficient_nodes",
                    },
                )
            )
            continue

        rel = nodes[idx] - pt[None, :]
        coords_local = rel @ R_local  # (n_k, 3), columns = (x_loc, y_loc, z_loc)
        # Thin slab along the front: keep only nodes whose projection on
        # the tangent is small compared to the in-plane distance.
        in_plane_r = np.linalg.norm(coords_local[:, :2], axis=1)
        max_slab = 0.6 * outer_r
        slab_mask = np.abs(coords_local[:, 2]) <= max_slab
        idx = idx[slab_mask]
        coords_local = coords_local[slab_mask]
        in_plane_r = in_plane_r[slab_mask]
        if idx.size < 8:
            results.append(
                JResult(
                    J_value=0.0,
                    K_I=0.0,
                    K_II=0.0,
                    K_III=0.0,
                    T_stress=0.0,
                    path_independence_residual=0.0,
                    metadata={
                        "s": float(s),
                        "position": pt.tolist(),
                        "n_nodes_in_slice": int(idx.size),
                        "note": "insufficient_nodes_after_slab",
                    },
                )
            )
            continue

        # Local displacements.
        disp_local = disp[idx] @ R_local  # (n_k, 3)
        u_xy = disp_local[:, :2]

        # Rotate Voigt stresses / strains into the local frame.
        sigma_local = _rotate_voigt_stress(stress_voigt[idx], R_local)
        if strain_voigt is not None:
            strain_local = _rotate_voigt_strain(strain_voigt[idx], R_local)
        else:
            strain_local = None

        # Pyramid q-function based on the in-plane distance to the front
        # sample point (origin in local coords).
        q_node = _q_function(in_plane_r, inner_r, outer_r)

        J = _j_domain_2d_slice(
            coords_xy=coords_local[:, :2],
            u_xy=u_xy,
            sigma_xy_voigt=sigma_local,
            strain_xy_voigt=strain_local,
            q_node=q_node,
            R_out=outer_r,
        )
        # Ensure non-negative (small numerical noise can make J slightly negative).
        J_clipped = float(max(J, 0.0))

        K_I = compute_k_from_j_elastic(J_clipped, E, nu, plane_strain=True)

        results.append(
            JResult(
                J_value=J_clipped,
                K_I=K_I,
                K_II=0.0,
                K_III=0.0,
                T_stress=0.0,
                path_independence_residual=0.0,
                metadata={
                    "s": float(s),
                    "position": pt.tolist(),
                    "tangent": T_vec.tolist(),
                    "inner_radius": inner_r,
                    "outer_radius": outer_r,
                    "n_nodes_in_slice": int(idx.size),
                },
            )
        )

    return results
