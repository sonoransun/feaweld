#!/usr/bin/env python3
"""Generate high-resolution concept images for feaweld advanced features.

Produces 23 figures under ``docs/images/`` as SVG + PNG@300dpi pairs.
Each generator calls into live feaweld code where possible so the
images stay consistent with the current implementation.

Usage:
    python scripts/generate_docs_concept_images.py
    python scripts/generate_docs_concept_images.py --only jax_backend_flow j2_radial_return
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon as MplPolygon,
    Rectangle,
)
from matplotlib.lines import Line2D

from feaweld.visualization import theme as _theme

IMAGES_DIR = _ROOT / "docs" / "images"


BLUE = _theme.FEAWELD_BLUE
RED = _theme.FEAWELD_RED
ORANGE = _theme.FEAWELD_ORANGE
GREEN = _theme.FEAWELD_GREEN
DARK = _theme.FEAWELD_DARK
GRAY = _theme.FEAWELD_GRAY


def _prepare():
    _theme.apply_feaweld_style()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, stem: str) -> tuple[Path, Path]:
    svg_path = IMAGES_DIR / f"{stem}.svg"
    png_path = IMAGES_DIR / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


# ---------------------------------------------------------------------------
# I01 — JAX differentiable backend data flow
# ---------------------------------------------------------------------------

def i01_jax_backend_flow():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 2.0, 2.0, 1.0, "AnalysisCase\n(pydantic)", BLUE),
        (2.8, 2.0, 2.0, 1.0, "JAXBackend\n.solve_static", DARK),
        (5.3, 2.0, 2.3, 1.0, "JAXConstitutiveModel\n(batched stress/tangent)", ORANGE),
        (8.1, 2.0, 1.8, 1.0, "Assemble K\n+ penalty BC", BLUE),
        (10.2, 2.0, 1.6, 1.0, "solve K u = f\n(jax.numpy)", GREEN),
    ]
    for x, y, w, h, label, color in boxes:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", lw=2,
            edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, weight="bold")

    # Forward arrows
    for (x1, _, w1, _, *_), (x2, *_) in zip(boxes[:-1], boxes[1:]):
        arrow = FancyArrowPatch(
            (x1 + w1, 2.5), (x2, 2.5),
            arrowstyle="-|>", mutation_scale=18, lw=1.6, color=DARK,
        )
        ax.add_patch(arrow)

    # Gradient backward flow (red, on top)
    back_boxes = boxes[::-1]
    for (x1, _, w1, _, *_), (x2, _, w2, *_) in zip(back_boxes[:-1], back_boxes[1:]):
        arrow = FancyArrowPatch(
            (x1 + 0.2, 3.4), (x2 + w2 - 0.2, 3.4),
            arrowstyle="-|>", mutation_scale=16, lw=1.4,
            color=RED, linestyle="--",
        )
        ax.add_patch(arrow)

    ax.text(6, 4.1, "jax.grad / jax.vjp — gradient flow through constitutive model",
            ha="center", fontsize=11, color=RED, weight="bold")
    ax.text(6, 1.5, "Forward FE assembly + linear solve",
            ha="center", fontsize=11, color=DARK)

    # Scalar QoI block
    qoi = FancyBboxPatch((10.4, 0.3), 1.4, 0.9, boxstyle="round,pad=0.05",
                        lw=2, edgecolor=RED, facecolor="#fff0ef")
    ax.add_patch(qoi)
    ax.text(11.1, 0.75, "Scalar QoI\n(compliance)",
            ha="center", va="center", fontsize=9, weight="bold")
    arrow_down = FancyArrowPatch((11.0, 2.0), (11.1, 1.2),
                                arrowstyle="-|>", mutation_scale=16,
                                color=DARK, lw=1.4)
    ax.add_patch(arrow_down)

    ax.set_title("JAX differentiable backend — gradients thread through solve")
    return _save(fig, "jax_backend_flow")


# ---------------------------------------------------------------------------
# I02 — J2 plasticity radial return (π-plane)
# ---------------------------------------------------------------------------

def i02_j2_radial_return():
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Yield surface (circle of radius sigma_y in π-plane)
    R = 2.0
    yld = Circle((0, 0), R, fill=False, color=DARK, lw=2.2, label="yield surface")
    ax.add_patch(yld)
    ax.text(0, -R - 0.35, r"$\sigma_y$", ha="center", fontsize=12, color=DARK)

    # Origin axes (pi-plane projections)
    for ang, lbl in zip([0, 2*np.pi/3, 4*np.pi/3], [r"$s_1$", r"$s_2$", r"$s_3$"]):
        xe = 3.0 * np.cos(ang)
        ye = 3.0 * np.sin(ang)
        ax.plot([0, xe], [0, ye], color=GRAY, lw=1.0, ls="--")
        ax.text(1.08 * xe, 1.08 * ye, lbl, ha="center", va="center",
                fontsize=11, color=GRAY)

    # Trial stress outside yield surface
    trial = np.array([2.6, 1.4])
    ax.plot(*trial, "o", color=RED, markersize=10, zorder=5)
    ax.annotate(r"$\sigma^{trial}$", trial, (trial[0] + 0.2, trial[1] + 0.25),
                fontsize=12, color=RED, weight="bold")

    # Return-mapped stress on yield surface (unit direction to origin from trial deviator)
    s_dir = trial / np.linalg.norm(trial)
    returned = s_dir * R
    ax.plot(*returned, "o", color=GREEN, markersize=10, zorder=5)
    ax.annotate(r"$\sigma^{n+1}$", returned, (returned[0] - 0.6, returned[1] + 0.3),
                fontsize=12, color=GREEN, weight="bold")

    # Return arrow
    arrow = FancyArrowPatch(trial, returned, arrowstyle="-|>",
                            mutation_scale=18, color=RED, lw=2.0)
    ax.add_patch(arrow)
    ax.text(0.5 * (trial[0] + returned[0]) + 0.1,
            0.5 * (trial[1] + returned[1]) - 0.35,
            r"$-2\mu \Delta\gamma\, \hat n$", fontsize=11, color=RED)

    # Normal direction from origin to yield-surface point
    ax.plot([0, returned[0] * 1.4], [0, returned[1] * 1.4],
            color=BLUE, lw=1.2, ls=":")
    ax.text(returned[0] * 1.45, returned[1] * 1.45, r"$\hat n$",
            fontsize=12, color=BLUE)

    ax.set_title("J2 plasticity radial return\n(deviatoric π-plane)")
    return _save(fig, "j2_radial_return")


# ---------------------------------------------------------------------------
# I03 — FCC crystal plasticity slip systems
# ---------------------------------------------------------------------------

def i03_crystal_plasticity_fcc():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from feaweld.solver.jax_crystal_plasticity import (
        FCC_SLIP_DIRECTIONS, FCC_SLIP_NORMALS,
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Unit cube wireframe
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=GRAY, lw=0.8)

    # Each of the 4 {111} slip planes — translucent
    plane_normals_set = []
    for n in FCC_SLIP_NORMALS:
        sig = tuple(np.round(n, 4))
        if sig not in plane_normals_set and tuple(-np.array(sig)) not in plane_normals_set:
            plane_normals_set.append(sig)

    plane_colors = [BLUE, ORANGE, GREEN, RED]

    def _clip_plane_in_cube(normal, c=0.5):
        # Vertices of cube edges intersected by plane n·x = c
        verts = []
        eps = 1e-9
        for (a, b) in edges:
            pa, pb = corners[a], corners[b]
            da = np.dot(normal, pa) - c
            db = np.dot(normal, pb) - c
            if da * db < -eps:
                t = da / (da - db)
                verts.append(pa + t * (pb - pa))
            elif abs(da) < eps:
                verts.append(pa)
        if len(verts) < 3:
            return None
        verts = np.array(verts)
        centroid = verts.mean(axis=0)
        # Sort by angle around centroid in plane
        ref = verts[0] - centroid
        ref /= np.linalg.norm(ref) + 1e-12
        ortho = np.cross(normal, ref)
        angs = np.arctan2(
            np.dot(verts - centroid, ortho),
            np.dot(verts - centroid, ref),
        )
        order = np.argsort(angs)
        return verts[order]

    for plane_n, color in zip(plane_normals_set[:4], plane_colors):
        verts = _clip_plane_in_cube(np.array(plane_n), c=np.dot(plane_n, [0.5, 0.5, 0.5]))
        if verts is None:
            continue
        poly = Poly3DCollection([verts], alpha=0.18, facecolor=color, edgecolor=color)
        ax.add_collection3d(poly)

    # Plot all 12 slip directions as small arrows from the cube center
    center = np.array([0.5, 0.5, 0.5])
    for n, s in zip(FCC_SLIP_NORMALS, FCC_SLIP_DIRECTIONS):
        end = center + 0.28 * s
        ax.plot([center[0], end[0]], [center[1], end[1]], [center[2], end[2]],
                color=DARK, lw=1.2)
        ax.scatter(*end, color=DARK, s=12)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("FCC crystal plasticity — 4 {111} planes × 3 <110> directions = 12 slip systems")
    return _save(fig, "crystal_plasticity_fcc")


# ---------------------------------------------------------------------------
# I04 — Phase-field fracture schematic
# ---------------------------------------------------------------------------

def i04_phase_field_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: sharp crack vs diffuse crack (damage field)
    ax = axes[0]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.axis("off")
    ax.set_title("Sharp crack  →  AT2 regularization (length l₀)")

    # Plate outline
    plate = Rectangle((-5, -2), 10, 4, fill=False, ec=GRAY, lw=1.5)
    ax.add_patch(plate)

    # Diffuse damage field d(x,y) rendered via imshow (vector-friendly)
    l0 = 0.6
    x = np.linspace(-5, 5, 120)
    y = np.linspace(-1.95, 1.95, 80)
    X, Y = np.meshgrid(x, y)
    D = np.exp(-np.abs(X) / l0) * np.exp(-np.abs(Y) / 2.0)
    ax.imshow(D, extent=(-5, 5, -1.95, 1.95), origin="lower",
              cmap="YlOrRd", aspect="auto", alpha=0.85,
              interpolation="bilinear")
    # Sharp crack line
    ax.plot([-5, 0], [0, 0], color="black", lw=2)
    ax.annotate("pre-crack (d=1)", (-4, 0.25), fontsize=10, color="black")
    ax.annotate("diffuse\ncrack tip\nd → 0",
                (2.4, 0.0), (2.8, -1.5),
                arrowprops=dict(arrowstyle="->", color=DARK),
                fontsize=10, color=DARK)
    ax.text(0, 1.7, f"l₀ = {l0}", ha="center", fontsize=11, color=DARK)

    # Right: AT2 functional breakdown
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("AT2 energy functional")

    ax.text(5, 8.5, r"$\Psi(u, d) = \int_\Omega g(d)\, \psi_{\rm el}(\varepsilon(u))\,dV$ "
                    r"$\;+\;G_c\int_\Omega \left(\dfrac{d^2}{2 l_0}+\dfrac{l_0}{2}|\nabla d|^2\right)dV$",
            ha="center", fontsize=13)
    ax.text(2.5, 5.0, "degraded elastic\nenergy",
            ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(7.5, 5.0, "crack surface\nenergy",
            ha="center", fontsize=11, color=RED, weight="bold")
    ax.text(5, 2.5, r"$g(d) = (1-d)^2 + \eta$",
            ha="center", fontsize=12, color=DARK)
    ax.text(5, 1.0, "staggered solve: u-step (K scales with g(d)) → d-step (history H = max ψ_el)",
            ha="center", fontsize=10, color=GRAY, style="italic")

    return _save(fig, "phase_field_schematic")


# ---------------------------------------------------------------------------
# I05 — DeepONet architecture
# ---------------------------------------------------------------------------

def i05_deeponet_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Branch network
    ax.add_patch(Rectangle((0.5, 5.0), 2.5, 2.0, fc=BLUE + "22", ec=BLUE, lw=2))
    ax.text(1.75, 6.0, "load params\n(F, M, T, ...)", ha="center", fontsize=10)
    for i, (x0, title) in enumerate([(3.5, "Dense 64"), (5.0, "Dense 64"), (6.5, "Dense 64")]):
        ax.add_patch(Rectangle((x0, 5.0), 1.3, 2.0, fc=DARK + "22", ec=DARK, lw=1.5))
        ax.text(x0 + 0.65, 6.0, title, ha="center", fontsize=9)
    ax.text(5.15, 7.4, "Branch network", ha="center", fontsize=12, weight="bold", color=BLUE)

    # Trunk network
    ax.add_patch(Rectangle((0.5, 1.0), 2.5, 2.0, fc=GREEN + "22", ec=GREEN, lw=2))
    ax.text(1.75, 2.0, "trunk coords\n(x, y, z)", ha="center", fontsize=10)
    for i, (x0, title) in enumerate([(3.5, "Dense 64"), (5.0, "Dense 64"), (6.5, "Dense 64")]):
        ax.add_patch(Rectangle((x0, 1.0), 1.3, 2.0, fc=DARK + "22", ec=DARK, lw=1.5))
        ax.text(x0 + 0.65, 2.0, title, ha="center", fontsize=9)
    ax.text(5.15, 0.4, "Trunk network", ha="center", fontsize=12, weight="bold", color=GREEN)

    # Latent dot product
    ax.add_patch(Circle((9.0, 4.0), 0.6, fc=ORANGE + "55", ec=ORANGE, lw=2))
    ax.text(9.0, 4.0, "⟨·,·⟩", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(9.0, 5.0, "latent inner product", ha="center", fontsize=10, color=ORANGE)

    # Output
    ax.add_patch(Rectangle((11.2, 3.0), 2.5, 2.0, fc=RED + "22", ec=RED, lw=2))
    ax.text(12.45, 4.0, "predicted\ndisplacement\nfield", ha="center", fontsize=11, weight="bold")

    # Arrows
    for y in (6.0, 2.0):
        ax.add_patch(FancyArrowPatch((3.0, y), (3.5, y),
                                    arrowstyle="-|>", mutation_scale=14, lw=1.5, color=GRAY))
        ax.add_patch(FancyArrowPatch((4.8, y), (5.0, y),
                                    arrowstyle="-|>", mutation_scale=14, lw=1.5, color=GRAY))
        ax.add_patch(FancyArrowPatch((6.3, y), (6.5, y),
                                    arrowstyle="-|>", mutation_scale=14, lw=1.5, color=GRAY))
    ax.add_patch(FancyArrowPatch((7.8, 6.0), (8.7, 4.4),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.8, color=BLUE))
    ax.add_patch(FancyArrowPatch((7.8, 2.0), (8.7, 3.6),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.8, color=GREEN))
    ax.add_patch(FancyArrowPatch((9.6, 4.0), (11.2, 4.0),
                                 arrowstyle="-|>", mutation_scale=16, lw=2.0, color=ORANGE))

    ax.set_title("DeepONet neural operator surrogate\n(feaweld.solver.neural_backend)")
    return _save(fig, "deeponet_architecture")


# ---------------------------------------------------------------------------
# I06 — Bayesian ensemble uncertainty bands (synthetic)
# ---------------------------------------------------------------------------

def i06_bayesian_ensemble_uq():
    rng = np.random.default_rng(0)
    x_train = np.linspace(0, 2 * np.pi, 40)
    y_train = np.sin(x_train) + 0.1 * rng.normal(size=x_train.size)
    x_test = np.linspace(-np.pi / 3, 3 * np.pi, 300)

    # Simulate 5 ensemble members with stochastic offsets in the OOD region
    members = []
    for i in range(5):
        noise_scale = 0.05 + 0.02 * i
        scale = 1.0 + 0.03 * (i - 2)
        shift = 0.02 * (i - 2)
        y = scale * np.sin(x_test + shift)
        y += noise_scale * rng.normal(size=x_test.size)
        # Add growing disagreement outside [0, 2pi]
        ood_mask = (x_test < 0) | (x_test > 2 * np.pi)
        ood_dist = np.where(ood_mask, np.minimum(np.abs(x_test), np.abs(x_test - 2*np.pi)), 0.0)
        y += 0.4 * (i - 2) * ood_dist
        members.append(y)
    members = np.array(members)

    mean = members.mean(axis=0)
    epistemic = members.std(axis=0)
    aleatoric = 0.1 * np.ones_like(mean)
    total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Aleatoric (inner) band
    ax.fill_between(x_test, mean - 2 * aleatoric, mean + 2 * aleatoric,
                    color=BLUE, alpha=0.2, label=r"aleatoric 2$\sigma$")
    # Total (outer) band
    ax.fill_between(x_test, mean - 2 * total, mean + 2 * total,
                    color=ORANGE, alpha=0.25, label=r"epistemic + aleatoric 2$\sigma$")
    # Members
    for i, y in enumerate(members):
        ax.plot(x_test, y, color=GRAY, lw=0.8, alpha=0.6,
                label="ensemble members" if i == 0 else None)
    ax.plot(x_test, mean, color=DARK, lw=2.2, label="ensemble mean")
    ax.scatter(x_train, y_train, color=RED, s=22, zorder=5, label="training data")

    ax.axvspan(-np.pi / 3, 0, color=GRAY, alpha=0.1)
    ax.axvspan(2 * np.pi, 3 * np.pi, color=GRAY, alpha=0.1)
    ax.text(-0.15, -1.9, "OOD", fontsize=10, ha="right", color=GRAY)
    ax.text(2 * np.pi + 0.15, -1.9, "OOD", fontsize=10, ha="left", color=GRAY)

    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title("Bayesian deep ensemble — epistemic uncertainty widens out-of-distribution")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return _save(fig, "bayesian_ensemble_uq")


# ---------------------------------------------------------------------------
# I07 — Active learning acquisition heatmap
# ---------------------------------------------------------------------------

def i07_active_learning_acquisition():
    rng = np.random.default_rng(1)
    nx, ny = 60, 60
    x = np.linspace(0, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)
    # Target QoI (analytic)
    Z = np.sin(0.6 * X) * np.cos(0.4 * Y) + 0.08 * X

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: true QoI + initial random samples
    ax = axes[0]
    im0 = ax.imshow(Z, extent=(0, 10, -5, 5), origin="lower",
                    cmap=_theme.get_cmap("stress"), aspect="auto",
                    interpolation="bilinear")
    init = rng.uniform([0, -5], [10, 5], size=(8, 2))
    ax.scatter(init[:, 0], init[:, 1], color="white", edgecolor="black",
               s=60, zorder=5, label="initial LHS")
    fig.colorbar(im0, ax=ax, label="true QoI")
    ax.set_title("Initial Latin hypercube (n=8)")
    ax.set_xlabel("F (axial force)")
    ax.set_ylabel("M (bending)")
    ax.legend(loc="upper right", fontsize=9)

    # Right: surrogate mean + acquisition-picked candidates
    ax = axes[1]
    # Simple synthetic surrogate std: large far from training points
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists = np.min(np.linalg.norm(coords[:, None, :] - init[None, :, :], axis=-1), axis=1)
    uncert = dists.reshape(X.shape)
    uncert = uncert / uncert.max()
    im1 = ax.imshow(uncert, extent=(0, 10, -5, 5), origin="lower",
                    cmap="magma", aspect="auto", interpolation="bilinear")
    fig.colorbar(im1, ax=ax, label="epistemic σ (max-variance acquisition)")
    # Top-k picked candidates
    picks_idx = np.argpartition(uncert.ravel(), -8)[-8:]
    picks = np.stack([X.ravel()[picks_idx], Y.ravel()[picks_idx]], axis=1)
    ax.scatter(init[:, 0], init[:, 1], color="white", edgecolor="black",
               s=50, zorder=5, label="training set")
    ax.scatter(picks[:, 0], picks[:, 1], color=RED, marker="*",
               s=200, edgecolor="white", zorder=6, label="acquisition picks")
    ax.set_title("Max-variance acquisition picks next cases")
    ax.set_xlabel("F (axial force)")
    ax.set_ylabel("M (bending)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Active learning loop — acquisition focuses exploration on uncertain regions",
                 fontsize=13)
    return _save(fig, "active_learning_acquisition")


# ---------------------------------------------------------------------------
# I08 — FFT homogenization RVE
# ---------------------------------------------------------------------------

def i08_fft_homogenization_rve():
    from feaweld.multiscale.fft_homogenization import (
        fft_homogenize,
        isotropic_stiffness,
        make_sphere_rve,
    )

    def _E_nu_to_K_G(E, nu):
        K = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        return K, G

    N = 32
    phase_map = make_sphere_rve(N, radius_frac=0.3)
    K_m, G_m = _E_nu_to_K_G(70e3, 0.33)     # aluminum-ish
    K_i, G_i = _E_nu_to_K_G(420e3, 0.25)    # alumina-ish
    C_matrix = isotropic_stiffness(bulk=K_m, shear=G_m)
    C_inclusion = isotropic_stiffness(bulk=K_i, shear=G_i)
    phase_stiffness = {0: C_matrix, 1: C_inclusion}

    C_eff = fft_homogenize(phase_map, phase_stiffness, tol=1e-4, max_iter=150)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: cross-section of the RVE
    ax = axes[0]
    slc = phase_map[:, :, N // 2]
    ax.imshow(slc, cmap="coolwarm", origin="lower", interpolation="nearest")
    ax.set_title(f"Voxel RVE ({N}³) — sphere inclusion in matrix")
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")

    # Right: effective stiffness matrix
    ax = axes[1]
    im = ax.imshow(C_eff, cmap=_theme.get_cmap("stress"))
    ax.set_title("Effective stiffness C_eff (Voigt)")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{zz}$",
              r"$\tau_{xy}$", r"$\tau_{yz}$", r"$\tau_{xz}$"]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(6):
        ax.text(i, i, f"{C_eff[i, i] / 1e3:.1f}", ha="center", va="center",
                fontsize=9, color="white", weight="bold")
    fig.colorbar(im, ax=ax, label="stiffness (MPa)")

    fig.suptitle("FFT (Moulinec–Suquet) homogenization", fontsize=13)
    return _save(fig, "fft_homogenization_rve")


# ---------------------------------------------------------------------------
# I09 — EnKF state-space
# ---------------------------------------------------------------------------

def i09_enkf_state_space():
    from feaweld.digital_twin.assimilation import (
        CrackEnKF,
        ParisLawModel,
        paris_law_sif,
    )

    rng = np.random.default_rng(0)
    dK = paris_law_sif(stress_range=100.0, geometry_factor=1.12)
    truth_model = ParisLawModel(C=1e-11, m=3.0, dK_fn=dK)
    # Simulate the true crack evolution
    n_blocks = 100
    dn = 10.0
    truth = [0.5]
    for _ in range(n_blocks):
        truth.append(truth_model.step(truth[-1], dn))
    truth = np.array(truth)
    obs_times = np.arange(0, n_blocks + 1, 10)
    observations = truth[obs_times] + rng.normal(0, 0.01, size=obs_times.size)

    filt = CrackEnKF(
        model=truth_model, n_ensemble=60,
        initial_mean=0.3, initial_std=0.1,
        process_noise_std=0.005, seed=1,
    )
    means, stds = [filt.mean], [filt.std]
    for i in range(1, n_blocks + 1):
        filt.predict(dn)
        if i in obs_times:
            obs_idx = int(np.where(obs_times == i)[0][0])
            filt.update(observations[obs_idx], obs_std=0.02)
        means.append(filt.mean)
        stds.append(filt.std)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    t = np.arange(n_blocks + 1)
    ax.fill_between(t, means - 2 * stds, means + 2 * stds,
                    color=ORANGE, alpha=0.25, label=r"ensemble 2$\sigma$")
    ax.plot(t, truth, color=DARK, lw=2, label="truth (Paris law)")
    ax.plot(t, means, color=BLUE, lw=2, label="EnKF mean")
    ax.scatter(obs_times, observations, color=RED, s=45, zorder=5, label="observations")
    ax.set_xlabel("cycle block")
    ax.set_ylabel("crack length a (mm)")
    ax.set_title("EnKF crack-length assimilation from noisy strain gauges")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, "enkf_state_space")


# ---------------------------------------------------------------------------
# I10 — Normalizing flow density transform
# ---------------------------------------------------------------------------

def i10_normalizing_flow_density():
    rng = np.random.default_rng(0)
    # Synthetic base (standard normal) and target (log-normal)
    base = rng.normal(size=5000)
    target = np.exp(0.5 * rng.normal(size=5000) + 0.3)  # log-normal

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, samples, title, color in [
        (axes[0], base, "Base: N(0, 1)", BLUE),
        (axes[1], np.exp(0.5 * base + 0.3), "RealNVP mapping", ORANGE),
        (axes[2], target, r"Target: $\log N_f$ posterior", RED),
    ]:
        ax.hist(samples, bins=50, color=color, alpha=0.7, density=True)
        ax.set_title(title, color=color, fontsize=12)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)

    # Arrows between panels
    for i in range(2):
        fig.text(0.34 + 0.33 * i, 0.5, "→", ha="center", va="center",
                 fontsize=32, color=GRAY)

    fig.suptitle("RealNVP normalizing flow — base → target density transform",
                 fontsize=13)
    return _save(fig, "normalizing_flow_density")


# ---------------------------------------------------------------------------
# I11 — Spline weld path with Frenet frame
# ---------------------------------------------------------------------------

def i11_spline_weld_path():
    from feaweld.geometry.weld_path import WeldPath
    from feaweld.core.types import Point3D

    # Helical control points
    t = np.linspace(0, 1, 10)
    ctrl = [
        Point3D(x=float(10 * np.cos(2 * np.pi * u)),
                y=float(10 * np.sin(2 * np.pi * u)),
                z=float(5 * u))
        for u in t
    ]
    path = WeldPath(control_points=ctrl, mode="bspline", degree=3)

    u_dense = np.linspace(0, 1, 200)
    pts = np.array([path.evaluate_u(u) for u in u_dense])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=BLUE, lw=2.5, label="B-spline path")
    cp = np.array([[p.x, p.y, p.z] for p in ctrl])
    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], color=RED, s=60, zorder=5,
               label="control points")
    ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=GRAY, lw=1.0, ls="--",
            label="control polygon")

    # Frenet frame at a few samples
    for u in [0.15, 0.4, 0.65, 0.9]:
        try:
            T, N, B = path.frenet_frame(u)
            p = path.evaluate_u(u)
            scale = 3.0
            for vec, color, name in [(T, GREEN, "T"), (N, ORANGE, "N"), (B, DARK, "B")]:
                end = p + scale * vec
                ax.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]],
                        color=color, lw=1.6)
        except Exception:
            pass

    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Spline weld path — control points + Frenet frame (T, N, B)")
    return _save(fig, "spline_weld_path")


# ---------------------------------------------------------------------------
# I12 — Groove profile gallery
# ---------------------------------------------------------------------------

def i12_groove_profile_gallery():
    from feaweld.geometry.groove import (
        JGroove, KGroove, UGroove, VGroove, XGroove,
    )

    grooves = [
        ("V", VGroove(plate_thickness=20, angle=60, root_gap=2, root_face=2)),
        ("U", UGroove(plate_thickness=20, root_radius=4, bevel_angle=10, root_gap=2, root_face=2)),
        ("J", JGroove(plate_thickness=20, bevel_angle=30, root_radius=4, root_gap=2, root_face=2)),
        ("X", XGroove(plate_thickness=20, angle_top=60, angle_bottom=60, root_gap=2, root_face=3)),
        ("K", KGroove(plate_thickness=20, angle=45, root_gap=2, root_face=2)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    for ax, (name, groove) in zip(axes, grooves):
        poly = groove.cross_section_polygon()
        # Plate background (30 mm wide to show context)
        plate = Rectangle((-15, 0), 30, 20, fc="#dcdcdc", ec=GRAY, lw=1)
        ax.add_patch(plate)
        # Groove weld metal polygon
        ax.add_patch(MplPolygon(poly, fc=ORANGE + "99", ec=ORANGE, lw=2))
        ax.set_xlim(-15, 15)
        ax.set_ylim(-1, 21)
        ax.set_aspect("equal")
        ax.set_title(f"{name}-groove\narea = {groove.area():.1f} mm²",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Weld groove preparation gallery (plate thickness = 20 mm)",
                 fontsize=13)
    return _save(fig, "groove_profile_gallery")


# ---------------------------------------------------------------------------
# I13 — Volumetric joint render (matplotlib 3D wireframe fallback)
# ---------------------------------------------------------------------------

def i13_volumetric_joint_render():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    def _box(origin, size, facecolor, alpha=0.4):
        ox, oy, oz = origin
        sx, sy, sz = size
        verts = [
            [(ox, oy, oz), (ox + sx, oy, oz), (ox + sx, oy + sy, oz), (ox, oy + sy, oz)],
            [(ox, oy, oz + sz), (ox + sx, oy, oz + sz), (ox + sx, oy + sy, oz + sz), (ox, oy + sy, oz + sz)],
            [(ox, oy, oz), (ox + sx, oy, oz), (ox + sx, oy, oz + sz), (ox, oy, oz + sz)],
            [(ox, oy + sy, oz), (ox + sx, oy + sy, oz), (ox + sx, oy + sy, oz + sz), (ox, oy + sy, oz + sz)],
            [(ox, oy, oz), (ox, oy + sy, oz), (ox, oy + sy, oz + sz), (ox, oy, oz + sz)],
            [(ox + sx, oy, oz), (ox + sx, oy + sy, oz), (ox + sx, oy + sy, oz + sz), (ox + sx, oy, oz + sz)],
        ]
        coll = Poly3DCollection(verts, facecolors=facecolor, edgecolors=DARK, linewidths=0.8, alpha=alpha)
        ax.add_collection3d(coll)

    # Plate left
    _box((-20, -5, 0), (19, 10, 20), BLUE, alpha=0.25)
    # Plate right
    _box((1, -5, 0), (19, 10, 20), BLUE, alpha=0.25)
    # Weld metal (V-groove prism)
    weld_verts = [
        [(-1, -5, 0), (1, -5, 0), (3, -5, 20), (-3, -5, 20)],
        [(-1, 5, 0), (1, 5, 0), (3, 5, 20), (-3, 5, 20)],
        [(-1, -5, 0), (-1, 5, 0), (-3, 5, 20), (-3, -5, 20)],
        [(1, -5, 0), (1, 5, 0), (3, 5, 20), (3, -5, 20)],
        [(-1, -5, 0), (1, -5, 0), (1, 5, 0), (-1, 5, 0)],
        [(-3, -5, 20), (3, -5, 20), (3, 5, 20), (-3, 5, 20)],
    ]
    weld = Poly3DCollection(weld_verts, facecolors=ORANGE, edgecolors=RED,
                            linewidths=1.4, alpha=0.85)
    ax.add_collection3d(weld)

    ax.set_xlim(-22, 22)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-1, 22)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Volumetric butt joint with V-groove weld metal")
    return _save(fig, "volumetric_joint_render")


# ---------------------------------------------------------------------------
# I14 — Fastener welds gallery (2x2 3D wireframes)
# ---------------------------------------------------------------------------

def i14_fastener_welds_gallery():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    titles = ["Plug weld", "Slot weld", "Stud weld", "Spot weld"]

    def _add_plate(ax, z0, thickness=3.0, half=10.0):
        x = [-half, half, half, -half]
        y = [-half, -half, half, half]
        top = [(x[i], y[i], z0 + thickness) for i in range(4)]
        bot = [(x[i], y[i], z0) for i in range(4)]
        verts = [top, bot,
                 [top[0], top[1], bot[1], bot[0]],
                 [top[1], top[2], bot[2], bot[1]],
                 [top[2], top[3], bot[3], bot[2]],
                 [top[3], top[0], bot[0], bot[3]]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=BLUE,
                                             edgecolors=DARK, alpha=0.25, linewidths=0.6))

    def _add_cylinder(ax, radius, z0, z1, color, alpha=0.9, n=24):
        theta = np.linspace(0, 2 * np.pi, n + 1)
        verts = []
        for i in range(n):
            quad = [
                (radius * np.cos(theta[i]), radius * np.sin(theta[i]), z0),
                (radius * np.cos(theta[i+1]), radius * np.sin(theta[i+1]), z0),
                (radius * np.cos(theta[i+1]), radius * np.sin(theta[i+1]), z1),
                (radius * np.cos(theta[i]), radius * np.sin(theta[i]), z1),
            ]
            verts.append(quad)
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color,
                                             edgecolors=DARK, alpha=alpha, linewidths=0.5))

    def _add_sphere(ax, radius, center, color, n=16):
        u = np.linspace(0, 2 * np.pi, n + 1)
        v = np.linspace(0, np.pi, n + 1)
        xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs * 0.3 + center[2] * 0.7, color=color,
                        alpha=0.9, linewidth=0)

    for i, title in enumerate(titles):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _add_plate(ax, 0, thickness=3.0)
        if title == "Plug weld":
            _add_cylinder(ax, 2.0, 0, 3.0, ORANGE)
        elif title == "Slot weld":
            # Approximate as a capsule: two half-cylinders at ends of a box
            ax.add_collection3d(Poly3DCollection([
                [(-3, -1.5, 0), (3, -1.5, 0), (3, 1.5, 0), (-3, 1.5, 0)],
                [(-3, -1.5, 3), (3, -1.5, 3), (3, 1.5, 3), (-3, 1.5, 3)],
                [(-3, -1.5, 0), (3, -1.5, 0), (3, -1.5, 3), (-3, -1.5, 3)],
                [(-3, 1.5, 0), (3, 1.5, 0), (3, 1.5, 3), (-3, 1.5, 3)],
            ], facecolors=ORANGE, edgecolors=DARK, alpha=0.9, linewidths=0.6))
        elif title == "Stud weld":
            _add_cylinder(ax, 1.5, 3.0, 10.0, ORANGE)
        elif title == "Spot weld":
            _add_plate(ax, 3.0, thickness=3.0)
            theta = np.linspace(0, 2 * np.pi, 24)
            # A lenticular nugget at z ~ 3 mm
            nugget = np.column_stack([
                1.5 * np.cos(theta),
                1.5 * np.sin(theta),
                3.0 + 0.2 * np.sin(theta),
            ])
            ax.plot(nugget[:, 0], nugget[:, 1], nugget[:, 2], color=RED, lw=2)

        ax.set_title(title, fontsize=12, color=DARK)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_zlim(-1, 12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle("Fastener weld types (geometry.fastener_welds)", fontsize=14)
    return _save(fig, "fastener_welds_gallery")


# ---------------------------------------------------------------------------
# I15 — Defect type gallery
# ---------------------------------------------------------------------------

def i15_defect_type_gallery():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    titles = ["Pore", "Slag inclusion", "Undercut",
              "Lack of fusion", "Root gap", "Surface crack"]

    for ax, title in zip(axes.flat, titles):
        # Plate background
        ax.add_patch(Rectangle((0, 0), 20, 10, fc="#e0e0e0", ec=GRAY, lw=1))
        # Weld metal (lens in the middle)
        weld = MplPolygon([(6, 0), (14, 0), (12, 10), (8, 10)],
                         fc=BLUE + "55", ec=BLUE, lw=1.2)
        ax.add_patch(weld)

        if title == "Pore":
            ax.add_patch(Circle((10, 5), 0.8, fc="white", ec=RED, lw=2))
        elif title == "Slag inclusion":
            ax.add_patch(Ellipse((10, 5), 2.5, 0.8, fc=DARK, ec=DARK, alpha=0.8))
        elif title == "Undercut":
            ax.add_patch(MplPolygon([(11, 10), (13, 10), (12, 9.3)],
                                   fc=RED, ec=RED))
            ax.plot([6, 8], [10, 10], color=GRAY, lw=2)
        elif title == "Lack of fusion":
            ax.plot([6.5, 7.5], [2, 8], color=RED, lw=2.5)
        elif title == "Root gap":
            ax.add_patch(Rectangle((9.6, 0), 0.8, 1.5, fc="white", ec=RED, lw=2))
        elif title == "Surface crack":
            crack_x = np.linspace(9, 11, 20)
            crack_y = 10 - 0.5 * (crack_x - 10) ** 2
            ax.plot(crack_x, crack_y, color=RED, lw=2)
            ax.plot([9, 11], [10, 10], color=GRAY, lw=1.5)

        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 11)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Weld defect types (feaweld.defects)", fontsize=14)
    return _save(fig, "defect_type_gallery")


# ---------------------------------------------------------------------------
# I16 — FAT downgrade curves
# ---------------------------------------------------------------------------

def i16_fat_downgrade_curves():
    from feaweld.defects.knockdown import (
        lof_fat_downgrade,
        porosity_fat_downgrade,
        undercut_fat_downgrade,
    )

    base_fat = 100.0
    t = 20.0

    # Pore: vary d/t from 0 to 0.25
    d_over_t = np.linspace(0.001, 0.25, 100)
    pore_fat = [porosity_fat_downgrade(dot * t, t, base_fat).downgraded_fat
                for dot in d_over_t]

    # Undercut: vary undercut depth from 0 to 0.2*t
    uc = np.linspace(0.001, 0.2 * t, 100)
    uc_fat = [undercut_fat_downgrade(u, t, base_fat).downgraded_fat for u in uc]

    # LoF: constant floor at 36
    lof = np.linspace(0.01, 2.0, 10)
    lof_fat = [lof_fat_downgrade(d, base_fat).downgraded_fat for d in lof]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d_over_t, pore_fat, color=BLUE, lw=2.5,
            label="Pore (BS 7910 Annex F)")
    ax.plot(uc / t, uc_fat, color=ORANGE, lw=2.5,
            label="Undercut (IIW)")
    ax.axhline(36, color=RED, lw=2.0, ls="--", label="LoF floor (IIW → FAT 36)")
    ax.axhline(base_fat, color=DARK, lw=1.2, ls=":", label=f"Base FAT = {base_fat:.0f}")

    ax.set_xlabel("defect characteristic / plate thickness")
    ax.set_ylabel("effective FAT class (MPa)")
    ax.set_title("Defect-based FAT downgrade per BS 7910 + IIW")
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    return _save(fig, "fat_downgrade_curves")


# ---------------------------------------------------------------------------
# I17 — ISO 5817 population sample
# ---------------------------------------------------------------------------

def i17_iso5817_population_sample():
    from feaweld.defects.population import sample_iso5817_population

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    level_colors = {"B": BLUE, "C": ORANGE, "D": RED}
    for ax, level in zip(axes, "BCD"):
        defects = sample_iso5817_population(
            level=level, weld_length=1000.0,
            weld_width=10.0, plate_thickness=20.0,
            seed=42,
            pore_rate_per_mm=0.05,
            undercut_rate_per_mm=0.02,
            slag_rate_per_mm=0.01,
        )
        # Plate background (1m long, 10 mm wide)
        ax.add_patch(Rectangle((0, -5), 1000, 10, fc="#f0f0f0", ec=GRAY, lw=0.5))
        counts = {"pore": 0, "undercut": 0, "slag": 0}
        for d in defects:
            if d.defect_type == "pore":
                counts["pore"] += 1
                ax.add_patch(Circle((d.center.x, d.center.y), d.diameter / 2,
                                    fc="white", ec=RED, lw=1.2))
            elif d.defect_type == "undercut":
                counts["undercut"] += 1
                ax.plot([d.start.x, d.end.x], [d.start.y, d.end.y],
                        color=ORANGE, lw=3)
            elif d.defect_type == "slag":
                counts["slag"] += 1
                ax.add_patch(Ellipse((d.center.x, d.center.y),
                                    2 * d.semi_axes[0], 2 * d.semi_axes[1],
                                    fc=DARK, alpha=0.7))
        ax.set_ylim(-6, 6)
        ax.set_ylabel(f"Level {level}\nwidth (mm)", color=level_colors[level])
        total = sum(counts.values())
        ax.set_title(f"ISO 5817 Level {level}  —  {total} defects "
                     f"(pore={counts['pore']}, undercut={counts['undercut']}, "
                     f"slag={counts['slag']})",
                     fontsize=11, color=level_colors[level])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("weld length (mm)")
    fig.suptitle("ISO 5817 stochastic defect populations (1 m weld, same seed)",
                 fontsize=13)
    fig.tight_layout()
    return _save(fig, "iso5817_population_sample")


# ---------------------------------------------------------------------------
# I18 — Multiaxial critical-plane search (Fibonacci sphere)
# ---------------------------------------------------------------------------

def i18_multiaxial_critical_plane():
    from feaweld.postprocess.multiaxial import fibonacci_sphere_grid

    grid = fibonacci_sphere_grid(200)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Colour by hypothetical Findley damage surface
    dmg = np.abs(grid[:, 2]) + 0.5 * np.abs(grid[:, 0])
    sc = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2],
                    c=dmg, cmap="plasma", s=35)

    # Highlight the Findley-optimal plane
    idx = int(np.argmax(dmg))
    ax.scatter(*grid[idx], color=RED, s=220, edgecolor="black",
               marker="*", zorder=10, label="Findley optimum")

    # Draw the plane normal as an arrow from origin
    n = grid[idx]
    ax.plot([0, n[0]], [0, n[1]], [0, n[2]], color=RED, lw=2.5)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-0.05, 1.2)
    ax.set_xlabel("n_x")
    ax.set_ylabel("n_y")
    ax.set_zlabel("n_z")
    ax.set_title("Multiaxial fatigue — Fibonacci sphere plane-search grid\n"
                 "(upper hemisphere, Findley-optimal plane highlighted)")
    fig.colorbar(sc, ax=ax, label="damage parameter", shrink=0.6)
    ax.legend(loc="upper left", fontsize=10)
    return _save(fig, "multiaxial_critical_plane")


# ---------------------------------------------------------------------------
# I19 — J-integral q-function weighting
# ---------------------------------------------------------------------------

def i19_j_integral_qfunction():
    from feaweld.fracture.j_integral import _q_function

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    q = _q_function(R, inner_r=1.0, outer_r=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: 2D q-function field (imshow keeps SVG small)
    ax = axes[0]
    im = ax.imshow(q, extent=(-5, 5, -5, 5), origin="lower",
                   cmap="magma", aspect="equal", interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1.0, fc="none", ec="white", lw=1.5, ls="--"))
    ax.add_patch(Circle((0, 0), 3.0, fc="none", ec="white", lw=1.5, ls="--"))
    ax.plot(0, 0, "r*", markersize=18, zorder=5, label="crack tip")
    ax.plot([-5, 0], [0, 0], color="white", lw=2.5, label="crack")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("q-function weighting field")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="q(r)")

    # Right: radial profile
    ax = axes[1]
    r_line = np.linspace(0, 5, 300)
    q_line = _q_function(r_line, inner_r=1.0, outer_r=3.0)
    ax.plot(r_line, q_line, color=BLUE, lw=2.5)
    ax.axvline(1.0, color=GREEN, ls="--", label="inner radius (plateau)")
    ax.axvline(3.0, color=RED, ls="--", label="outer radius (cutoff)")
    ax.set_xlabel("r (distance from crack tip)")
    ax.set_ylabel("q(r)")
    ax.set_title("Radial profile")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("J-integral domain-form weighting (feaweld.fracture.j_integral_2d)",
                 fontsize=13)
    return _save(fig, "j_integral_qfunction")


# ---------------------------------------------------------------------------
# I20 — CTOD extrapolation
# ---------------------------------------------------------------------------

def i20_ctod_extrapolation():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    # Undeformed plate outline
    ax.add_patch(Rectangle((-8, -2.5), 16, 5, fc="#f0f0f0", ec=GRAY, lw=0.8))
    # Crack axis (undeformed)
    ax.plot([-8, 0], [0, 0], color="black", lw=2, ls="--", label="undeformed crack line")

    # Deformed crack flanks (opening monotonically behind the tip)
    x_flank = np.linspace(-8, 0, 200)
    opening = 0.8 * np.sqrt(np.maximum(-x_flank, 0))
    upper = opening
    lower = -opening
    ax.plot(x_flank, upper, color=BLUE, lw=2.5, label="upper flank (deformed)")
    ax.plot(x_flank, lower, color=GREEN, lw=2.5, label="lower flank (deformed)")

    # Crack tip marker
    ax.plot(0, 0, "r*", markersize=18, zorder=6, label="crack tip")

    # Displacement extrapolation points at offset -2
    x_ext = -2.0
    y_u = 0.8 * np.sqrt(2)
    y_l = -y_u
    ax.plot(x_ext, y_u, "o", color=BLUE, markersize=10, zorder=6)
    ax.plot(x_ext, y_l, "o", color=GREEN, markersize=10, zorder=6)
    ax.annotate("", xy=(x_ext + 0.2, y_l), xytext=(x_ext + 0.2, y_u),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2))
    ax.text(x_ext + 0.35, 0, "CTOD\n(extrapolation)",
            fontsize=11, color=RED, weight="bold")

    # 90-degree intercept construction
    ax.plot([-6, 0], [6, 0], color=ORANGE, lw=1.4, ls=":")
    ax.plot([-6, 0], [-6, 0], color=ORANGE, lw=1.4, ls=":")
    ax.text(-5.5, 2.4, "45° rays from tip\n(90° intercept method)",
            fontsize=9, color=ORANGE)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (distance from crack tip)")
    ax.set_ylabel("y (crack opening)")
    ax.set_title("CTOD extraction methods (feaweld.fracture.ctod)")
    ax.grid(True, alpha=0.3)
    return _save(fig, "ctod_extrapolation")


# ---------------------------------------------------------------------------
# I21 — Volumetric SED control volume over a notch
# ---------------------------------------------------------------------------

def i21_volumetric_sed_control():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Notched plate outline (simple box with a triangular notch)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    box_verts = [
        [(-10, -5, 0), (10, -5, 0), (10, 5, 0), (-10, 5, 0)],
        [(-10, -5, 10), (10, -5, 10), (10, 5, 10), (-10, 5, 10)],
        [(-10, -5, 0), (-10, -5, 10), (-10, 5, 10), (-10, 5, 0)],
        [(10, -5, 0), (10, -5, 10), (10, 5, 10), (10, 5, 0)],
        [(-10, -5, 0), (10, -5, 0), (10, -5, 10), (-10, -5, 10)],
    ]
    ax.add_collection3d(Poly3DCollection(box_verts, fc=BLUE,
                                         ec=DARK, alpha=0.15, linewidths=0.6))

    # Control volume (cylinder at the notch tip)
    theta = np.linspace(0, 2 * np.pi, 32)
    for z_level in np.linspace(4, 6, 8):
        xs = 1.5 * np.cos(theta)
        ys = 1.5 * np.sin(theta)
        ax.plot(xs, ys, z_level, color=ORANGE, lw=0.7)

    # MC sample points colored by SED
    n_samples = 300
    pts = rng.uniform([-1.5, -1.5, 4], [1.5, 1.5, 6], size=(n_samples, 3))
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.5 ** 2
    pts = pts[inside]
    sed = 50 * np.exp(-np.linalg.norm(pts - np.array([0, 0, 5]), axis=1))
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sed, cmap="YlOrRd", s=10)

    # Notch tip marker
    ax.scatter(0, 0, 5, color=RED, marker="*", s=180, zorder=6, label="notch tip")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Volumetric SED — cylindrical control volume around a notch\n"
                 "(MC sampling via averaged_sed_over_volume)")
    fig.colorbar(sc, ax=ax, label="SED (N·mm/mm³)", shrink=0.7)
    ax.legend(loc="upper left", fontsize=10)
    return _save(fig, "volumetric_sed_control")


# ---------------------------------------------------------------------------
# I22 — Multipass bead stacking
# ---------------------------------------------------------------------------

def i22_multipass_bead_stacking():
    fig, ax = plt.subplots(figsize=(10, 6))

    # V-groove outline
    groove = MplPolygon([(-6, 0), (6, 0), (10, 20), (-10, 20)],
                       fc="#e8e8e8", ec=GRAY, lw=1.5)
    ax.add_patch(groove)
    # Also plate outline
    ax.add_patch(Rectangle((-20, 0), 40, 20, fc="#f5f5f5", ec=GRAY, lw=1,
                           zorder=-1))

    # 4 beads: root, fill1, fill2, cap
    beads = [
        ("Root", -5, 5, 0, 5, BLUE, 1),
        ("Fill 1", -6, 6, 5, 11, GREEN, 2),
        ("Fill 2", -7, 7, 11, 16, ORANGE, 3),
        ("Cap", -9, 9, 16, 20, RED, 4),
    ]
    for name, x0, x1, y0, y1, color, order in beads:
        # Trapezoidal bead matching groove taper
        frac0 = y0 / 20.0
        frac1 = y1 / 20.0
        xl0 = -6 - 4 * frac0
        xl1 = -6 - 4 * frac1
        xr0 = 6 + 4 * frac0
        xr1 = 6 + 4 * frac1
        bead = MplPolygon([(xl0, y0), (xr0, y0), (xr1, y1), (xl1, y1)],
                         fc=color, ec=DARK, lw=1.5, alpha=0.85)
        ax.add_patch(bead)
        ax.text(0, 0.5 * (y0 + y1), f"{order}: {name}",
                ha="center", va="center", fontsize=11, color="white",
                weight="bold")

    ax.set_xlim(-22, 22)
    ax.set_ylim(-1, 22)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Multipass bead stacking (V-groove: root → fills → cap)")
    ax.grid(True, alpha=0.25)
    return _save(fig, "multipass_bead_stacking")


# ---------------------------------------------------------------------------
# I23 — Rainflow multiaxial projection
# ---------------------------------------------------------------------------

def i23_rainflow_multiaxial_projection():
    rng = np.random.default_rng(0)
    n_t = 400
    t = np.arange(n_t)

    # Synthetic variable-amplitude signed-principal stress history
    sigma = (100 * np.sin(2 * np.pi * t / 50)
             + 50 * np.sin(2 * np.pi * t / 17)
             + 20 * rng.normal(size=n_t))

    from feaweld.fatigue.rainflow import rainflow_count

    cycles = rainflow_count(sigma)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: history with extracted peaks
    ax = axes[0]
    ax.plot(t, sigma, color=BLUE, lw=1.2)
    ax.set_xlabel("time step")
    ax.set_ylabel("signed max principal stress (MPa)")
    ax.set_title("Signed-principal history")
    ax.grid(True, alpha=0.3)

    # Right: extracted cycles as range vs mean
    ax = axes[1]
    ranges = [c[0] for c in cycles]
    means = [c[1] for c in cycles]
    counts = [c[2] for c in cycles]
    sc = ax.scatter(means, ranges, c=counts, cmap=_theme.get_cmap("damage"),
                    s=80, edgecolor="black")
    ax.set_xlabel("mean stress (MPa)")
    ax.set_ylabel("stress range (MPa)")
    ax.set_title(f"Rainflow cycles extracted ({len(cycles)} cycles)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="cycle count (1.0 or 0.5)")

    fig.suptitle("Rainflow multiaxial: signed-principal projection + ASTM E1049 extraction",
                 fontsize=13)
    return _save(fig, "rainflow_multiaxial_projection")


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Callable] = {
    "jax_backend_flow": i01_jax_backend_flow,
    "j2_radial_return": i02_j2_radial_return,
    "crystal_plasticity_fcc": i03_crystal_plasticity_fcc,
    "phase_field_schematic": i04_phase_field_schematic,
    "deeponet_architecture": i05_deeponet_architecture,
    "bayesian_ensemble_uq": i06_bayesian_ensemble_uq,
    "active_learning_acquisition": i07_active_learning_acquisition,
    "fft_homogenization_rve": i08_fft_homogenization_rve,
    "enkf_state_space": i09_enkf_state_space,
    "normalizing_flow_density": i10_normalizing_flow_density,
    "spline_weld_path": i11_spline_weld_path,
    "groove_profile_gallery": i12_groove_profile_gallery,
    "volumetric_joint_render": i13_volumetric_joint_render,
    "fastener_welds_gallery": i14_fastener_welds_gallery,
    "defect_type_gallery": i15_defect_type_gallery,
    "fat_downgrade_curves": i16_fat_downgrade_curves,
    "iso5817_population_sample": i17_iso5817_population_sample,
    "multiaxial_critical_plane": i18_multiaxial_critical_plane,
    "j_integral_qfunction": i19_j_integral_qfunction,
    "ctod_extrapolation": i20_ctod_extrapolation,
    "volumetric_sed_control": i21_volumetric_sed_control,
    "multipass_bead_stacking": i22_multipass_bead_stacking,
    "rainflow_multiaxial_projection": i23_rainflow_multiaxial_projection,
}


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Subset of figure stems to generate")
    args = parser.parse_args(argv)

    _prepare()

    targets = _GENERATORS.items()
    if args.only:
        missing = [n for n in args.only if n not in _GENERATORS]
        if missing:
            raise SystemExit(f"Unknown generators: {missing}")
        targets = [(k, v) for k, v in _GENERATORS.items() if k in args.only]

    paths: list[Path] = []
    for name, gen in targets:
        try:
            svg_path, png_path = gen()
            size_kb = svg_path.stat().st_size / 1024
            print(f"✓ {name}: {size_kb:.1f} KB SVG, "
                  f"{png_path.stat().st_size / 1024:.1f} KB PNG")
            paths.extend([svg_path, png_path])
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")
            raise
    return paths


if __name__ == "__main__":
    main()
