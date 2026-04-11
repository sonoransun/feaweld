#!/usr/bin/env python3
"""Generate documentation animations (GIF + MP4) for feaweld concepts.

Produces 9 animations under ``docs/animations/``. Each is a GIF +
MP4 pair. GIFs use palette quantization through pillow; MP4s use
imageio-ffmpeg's bundled binary.

Usage:
    python scripts/generate_docs_animations.py
    python scripts/generate_docs_animations.py --only phase_field_crack_propagation
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
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

import imageio.v3 as iio

from feaweld.visualization import theme as _theme

ANIM_DIR = _ROOT / "docs" / "animations"

BLUE = _theme.FEAWELD_BLUE
RED = _theme.FEAWELD_RED
ORANGE = _theme.FEAWELD_ORANGE
GREEN = _theme.FEAWELD_GREEN
DARK = _theme.FEAWELD_DARK
GRAY = _theme.FEAWELD_GRAY


def _prepare():
    _theme.apply_feaweld_style()
    ANIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Frame renderer → GIF + MP4
# ---------------------------------------------------------------------------

def _fig_to_frame(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy frame."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf[..., :3].reshape(h, w, 3)


def _write_animation(
    frames: list[np.ndarray],
    stem: str,
    fps: int = 10,
) -> tuple[Path, Path]:
    """Write an animation to both GIF (palette quantized) and MP4."""
    gif_path = ANIM_DIR / f"{stem}.gif"
    mp4_path = ANIM_DIR / f"{stem}.mp4"

    # GIF via imageio (applies pillow palette quantization by default)
    iio.imwrite(gif_path, frames, duration=int(1000 / fps), loop=0)

    # MP4 via imageio (imageio-ffmpeg bundled binary)
    iio.imwrite(
        mp4_path, frames,
        extension=".mp4",
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
    )
    return gif_path, mp4_path


# ---------------------------------------------------------------------------
# A01 — Phase-field crack propagation
# ---------------------------------------------------------------------------

def a01_phase_field_crack_propagation():
    # Analytic 2D diffuse damage field evolving with load step
    nx, ny = 120, 60
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)

    frames = []
    n_frames = 20
    for i in range(n_frames):
        load = (i + 1) / n_frames
        l0 = 0.5
        # Crack tip position advances with load
        tip = -4.0 + 7.0 * load
        dist = np.sqrt((X - tip) ** 2 + Y ** 2)
        # Damage field: plateau at 1 behind the tip, decays ahead
        d = np.where(X < tip, 1.0, np.exp(-(X - tip) / l0) * np.exp(-np.abs(Y) / (2 * l0)))
        d = np.clip(d, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(d, extent=(-5, 5, -2.5, 2.5), origin="lower",
                  cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Phase-field crack propagation  —  load step {i + 1}/{n_frames}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.text(4.5, 2.0, f"load = {load:.2f}", fontsize=11,
                ha="right", color=DARK, weight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=DARK))
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "phase_field_crack_propagation", fps=5)


# ---------------------------------------------------------------------------
# A02 — Multipass thermal cycle
# ---------------------------------------------------------------------------

def a02_multipass_thermal_cycle():
    nx, ny = 60, 60
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)

    # 3 passes with Gaussian peaks moving in +x, shifted vertically by pass
    n_frames = 60
    pass_bounds = [(0, 20, 0.0), (22, 42, 2.0), (44, 60, 4.0)]  # (start, end, y_center)

    frames = []
    for i in range(n_frames):
        T = 20.0 * np.ones_like(X)  # ambient
        # Residual heat from earlier passes (exponential decay)
        for (start, end, y_c) in pass_bounds:
            if i < start:
                continue
            progress = min((i - start) / (end - start), 1.0)
            x_c = -8.0 + 16.0 * progress
            if i < end:
                # Active heat source
                T += 1200.0 * np.exp(-((X - x_c) ** 2 + (Y - y_c) ** 2) / 3.0)
            else:
                # Residual cooling heat
                residual_age = i - end
                decay = np.exp(-residual_age / 15.0)
                T += 400.0 * decay * np.exp(-((X - 5.0) ** 2 + (Y - y_c) ** 2) / 5.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(T, extent=(-10, 10, -5, 5), origin="lower",
                       cmap=_theme.get_cmap("temperature"), aspect="auto",
                       vmin=20, vmax=1500)
        # V-groove outline
        ax.plot([-6, 6], [5, 5], color="white", lw=1.2)
        ax.plot([-6, -2], [5, -5], color="white", lw=1.2)
        ax.plot([2, 6], [-5, 5], color="white", lw=1.2)
        # Bead outlines
        for (start, end, y_c) in pass_bounds:
            if i >= start:
                ax.add_patch(Rectangle((-8, y_c - 0.8), 16, 1.6,
                                       fill=False, ec="white", lw=0.8, ls="--"))
        active_pass = next((k + 1 for k, (s, e, _) in enumerate(pass_bounds)
                            if s <= i < e), None)
        label = f"Pass {active_pass} active" if active_pass else "Cooling"
        ax.set_title(f"Multipass thermal cycle  —  t = {i}s  ({label})")
        ax.set_xlabel("x along weld (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="T (°C)")
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "multipass_thermal_cycle", fps=10)


# ---------------------------------------------------------------------------
# A03 — Goldak heat source sweep
# ---------------------------------------------------------------------------

def a03_goldak_heat_source_sweep():
    from feaweld.solver.thermal import GoldakHeatSource

    src = GoldakHeatSource(
        power=2000.0,
        travel_speed=5.0,
        a_f=4.0,
        a_r=8.0,
        b=5.0,
        c=5.0,
        start_position=np.array([-12.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    nx, ny = 120, 60
    x = np.linspace(-15, 15, nx)
    y = np.linspace(-6, 6, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    n_frames = 40
    t_range = np.linspace(0, 5.0, n_frames)
    frames = []
    for t in t_range:
        q = src.evaluate(X, Y, Z, t)
        q = np.array(q, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(q, extent=(-15, 15, -6, 6), origin="lower",
                       cmap=_theme.get_cmap("temperature"), aspect="auto")
        ax.set_title(f"Goldak double-ellipsoid heat source  —  t = {t:.2f} s")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="q (W/mm³)")
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "goldak_heat_source_sweep", fps=10)


# ---------------------------------------------------------------------------
# A04 — Active learning convergence
# ---------------------------------------------------------------------------

def a04_active_learning_convergence():
    rng = np.random.default_rng(0)

    def true_qoi(x, y):
        return np.sin(0.6 * x) * np.cos(0.4 * y) + 0.08 * x

    # Grid for background
    nx, ny = 60, 60
    gx = np.linspace(0, 10, nx)
    gy = np.linspace(-5, 5, ny)
    GX, GY = np.meshgrid(gx, gy)
    Z = true_qoi(GX, GY)

    # Initial LHS
    init = rng.uniform([0, -5], [10, 5], size=(6, 2))
    train = list(init)
    y_train = [true_qoi(x[0], x[1]) for x in train]

    n_iters = 20
    rmse_history = []
    test_x = rng.uniform([0, -5], [10, 5], size=(100, 2))
    test_y = np.array([true_qoi(*p) for p in test_x])

    frames = []
    for it in range(n_iters):
        # Simple surrogate: RBF-weighted average of training points
        dists = np.linalg.norm(
            np.stack([GX.ravel(), GY.ravel()], axis=1)[:, None, :]
            - np.array(train)[None, :, :], axis=-1
        )
        weights = np.exp(-dists / 2.0)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12
        mean = (weights @ np.array(y_train)).reshape(nx, ny)
        # Epistemic sigma proxy: min distance to any training point
        sigma = np.min(dists, axis=1).reshape(nx, ny)
        sigma /= sigma.max() + 1e-12

        # Test RMSE
        test_dists = np.linalg.norm(
            test_x[:, None, :] - np.array(train)[None, :, :], axis=-1
        )
        test_w = np.exp(-test_dists / 2.0)
        test_w /= test_w.sum(axis=1, keepdims=True) + 1e-12
        test_pred = test_w @ np.array(y_train)
        rmse = float(np.sqrt(np.mean((test_pred - test_y) ** 2)))
        rmse_history.append(rmse)

        # Pick next candidate by max-variance
        pick_idx = int(np.argmax(sigma.ravel()))
        pick = np.array([GX.ravel()[pick_idx], GY.ravel()[pick_idx]])
        train.append(pick)
        y_train.append(true_qoi(*pick))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        im0 = axes[0].imshow(mean, extent=(0, 10, -5, 5), origin="lower",
                             cmap=_theme.get_cmap("stress"), aspect="auto",
                             vmin=Z.min(), vmax=Z.max())
        tr = np.array(train[:-1])
        axes[0].scatter(tr[:, 0], tr[:, 1], c="white", edgecolor="black",
                        s=40, zorder=5)
        axes[0].scatter(*pick, c=RED, marker="*", s=200, edgecolor="white",
                        zorder=6)
        axes[0].set_title(f"Surrogate mean — iter {it + 1}/{n_iters}")
        axes[0].set_xlabel("F (axial)")
        axes[0].set_ylabel("M (bending)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        axes[1].plot(range(1, it + 2), rmse_history, color=BLUE, lw=2)
        axes[1].scatter(range(1, it + 2), rmse_history, color=DARK, s=30)
        axes[1].set_xlim(0, n_iters + 1)
        axes[1].set_ylim(0, max(rmse_history + [0.5]) * 1.1)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("test RMSE")
        axes[1].set_title("Convergence")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Active learning — max-variance acquisition", fontsize=12)
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "active_learning_convergence", fps=4)


# ---------------------------------------------------------------------------
# A05 — Monte Carlo convergence
# ---------------------------------------------------------------------------

def a05_monte_carlo_convergence():
    rng = np.random.default_rng(0)

    # Target: lognormal fatigue life draws
    n_total = 5000
    samples = rng.lognormal(mean=14.0, sigma=0.4, size=n_total)
    mu_true = float(np.mean(samples))

    n_frames = 50
    checkpoints = np.linspace(20, n_total, n_frames, dtype=int)

    frames = []
    running_mean = []
    for i, n in enumerate(checkpoints):
        subset = samples[:n]
        running_mean.append(float(np.mean(subset)))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        # Left: histogram
        axes[0].hist(subset, bins=40, color=BLUE, alpha=0.75, density=True)
        axes[0].axvline(mu_true, color=RED, lw=2, ls="--", label="true mean")
        axes[0].axvline(running_mean[-1], color=GREEN, lw=2,
                        label=f"running mean (n={n})")
        axes[0].set_xlabel("fatigue life N (log-scale)")
        axes[0].set_ylabel("density")
        axes[0].set_title(f"Sample histogram  —  n = {n}")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].set_xscale("log")
        axes[0].grid(True, alpha=0.3)

        # Right: running mean vs n
        axes[1].plot(checkpoints[: i + 1], running_mean, color=BLUE, lw=2)
        axes[1].axhline(mu_true, color=RED, lw=1.5, ls="--")
        axes[1].fill_between(
            checkpoints[: i + 1],
            np.array(running_mean) - 0.1 * mu_true,
            np.array(running_mean) + 0.1 * mu_true,
            color=BLUE, alpha=0.2,
        )
        axes[1].set_xlim(0, n_total)
        axes[1].set_xlabel("sample count n")
        axes[1].set_ylabel("running mean")
        axes[1].set_title("Monte Carlo convergence")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Monte Carlo fatigue-life sampling",
                     fontsize=13)
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "monte_carlo_convergence", fps=10)


# ---------------------------------------------------------------------------
# A06 — EnKF crack tracking
# ---------------------------------------------------------------------------

def a06_enkf_crack_tracking():
    from feaweld.digital_twin.assimilation import (
        CrackEnKF, ParisLawModel, paris_law_sif,
    )

    rng = np.random.default_rng(0)
    dK = paris_law_sif(stress_range=100.0, geometry_factor=1.12)
    model = ParisLawModel(C=1e-11, m=3.0, dK_fn=dK)
    dn = 10.0
    n_blocks = 100

    truth = [0.5]
    for _ in range(n_blocks):
        truth.append(model.step(truth[-1], dn))
    truth = np.array(truth)
    obs_times = np.arange(0, n_blocks + 1, 10)
    observations = truth[obs_times] + rng.normal(0, 0.01, size=obs_times.size)

    filt = CrackEnKF(
        model=model, n_ensemble=50,
        initial_mean=0.3, initial_std=0.12,
        process_noise_std=0.005, seed=1,
    )

    means, stds = [filt.mean], [filt.std]
    frames = []

    for step in range(1, n_blocks + 1):
        filt.predict(dn)
        if step in obs_times:
            obs_idx = int(np.where(obs_times == step)[0][0])
            filt.update(observations[obs_idx], obs_std=0.02)
        means.append(filt.mean)
        stds.append(filt.std)

        if step % 2 == 0:  # one frame every 2 blocks for 50 frames total
            fig, ax = plt.subplots(figsize=(10, 5.5))
            t = np.arange(len(means))
            ax.fill_between(t, np.array(means) - 2 * np.array(stds),
                            np.array(means) + 2 * np.array(stds),
                            color=ORANGE, alpha=0.25, label=r"2$\sigma$")
            ax.plot(np.arange(n_blocks + 1), truth, color=DARK, lw=2,
                    label="truth")
            ax.plot(t, means, color=BLUE, lw=2, label="EnKF mean")
            obs_seen = [i for i in obs_times if i <= step]
            if obs_seen:
                obs_vals = [observations[int(np.where(obs_times == i)[0][0])]
                            for i in obs_seen]
                ax.scatter(obs_seen, obs_vals, color=RED, s=50, zorder=5,
                           label="observations")
            ax.set_xlim(0, n_blocks + 1)
            ax.set_ylim(0, max(truth) * 1.1)
            ax.set_xlabel("cycle block")
            ax.set_ylabel("crack length a (mm)")
            ax.set_title(f"EnKF crack-length tracking  —  block {step}/{n_blocks}")
            ax.legend(loc="upper left", fontsize=10)
            ax.grid(True, alpha=0.3)
            frames.append(_fig_to_frame(fig))
            plt.close(fig)

    return _write_animation(frames, "enkf_crack_tracking", fps=10)


# ---------------------------------------------------------------------------
# A07 — Bayesian posterior update
# ---------------------------------------------------------------------------

def a07_bayesian_posterior_update():
    rng = np.random.default_rng(0)

    # 2D Gaussian conjugate update over (mu_x, mu_y) with known unit variance
    true_mean = np.array([1.5, -0.5])
    nx = ny = 80
    x = np.linspace(-3, 4, nx)
    y = np.linspace(-4, 3, ny)
    X, Y = np.meshgrid(x, y)

    def gauss2d(mx, my, sx, sy):
        return np.exp(-0.5 * ((X - mx) ** 2 / sx ** 2 + (Y - my) ** 2 / sy ** 2))

    n_obs = 30
    observations = rng.normal(loc=true_mean, scale=1.0, size=(n_obs, 2))

    # Prior: N(0, 2^2 I)
    prior_mean = np.array([0.0, 0.0])
    prior_std = 2.0

    frames = []
    post_mean = prior_mean.copy()
    post_precision = np.ones(2) / prior_std ** 2
    for i in range(n_obs):
        obs_precision = 1.0
        post_precision = post_precision + obs_precision
        post_mean = (post_mean * (post_precision - obs_precision)
                     + observations[i] * obs_precision) / post_precision
        post_std = 1.0 / np.sqrt(post_precision)

        fig, ax = plt.subplots(figsize=(7, 6))
        prior_Z = gauss2d(prior_mean[0], prior_mean[1], prior_std, prior_std)
        post_Z = gauss2d(post_mean[0], post_mean[1], post_std[0], post_std[1])

        ax.contour(X, Y, prior_Z, levels=5, colors=[GRAY], linewidths=1,
                   linestyles="--")
        ax.contourf(X, Y, post_Z, levels=20, cmap="Blues", alpha=0.7)
        ax.scatter(observations[: i + 1, 0], observations[: i + 1, 1],
                   color=RED, s=40, edgecolor="black", label="observations")
        ax.plot(*true_mean, "g*", markersize=22, markeredgecolor="black",
                label="true mean")
        ax.plot(*post_mean, "o", color=ORANGE, markersize=12,
                markeredgecolor="black", label="posterior mean")
        ax.set_xlim(-3, 4)
        ax.set_ylim(-4, 3)
        ax.set_xlabel(r"$\mu_x$")
        ax.set_ylabel(r"$\mu_y$")
        ax.set_title(f"Bayesian update  —  obs {i + 1}/{n_obs}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "bayesian_posterior_update", fps=8)


# ---------------------------------------------------------------------------
# A08 — Rainflow cycle counting
# ---------------------------------------------------------------------------

def a08_rainflow_cycle_counting():
    rng = np.random.default_rng(0)
    n_t = 120
    t = np.arange(n_t)
    sigma = (100 * np.sin(2 * np.pi * t / 30)
             + 50 * np.sin(2 * np.pi * t / 13)
             + 20 * rng.normal(size=n_t))

    from feaweld.fatigue.rainflow import rainflow_count

    # Progressive extraction: build up frame-by-frame by feeding prefixes
    frames = []
    n_frames = 40
    checkpoints = np.linspace(20, n_t, n_frames, dtype=int)
    for i, k in enumerate(checkpoints):
        cycles = rainflow_count(sigma[:k])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                                 gridspec_kw={"height_ratios": [2, 1]},
                                 sharex=False)
        # Top: running history
        ax = axes[0]
        ax.plot(t[:k], sigma[:k], color=BLUE, lw=1.2)
        ax.plot(t[k:], sigma[k:], color=GRAY, lw=0.8, alpha=0.3)
        ax.axvline(k, color=RED, lw=1.2, ls="--")
        ax.set_xlim(0, n_t)
        ax.set_ylim(sigma.min() - 20, sigma.max() + 20)
        ax.set_ylabel("stress (MPa)")
        ax.set_title(f"Stress history  —  processed {k}/{n_t} points")
        ax.grid(True, alpha=0.3)

        # Bottom: extracted range distribution so far
        ax = axes[1]
        if cycles:
            ranges = [c[0] for c in cycles]
            ax.hist(ranges, bins=15, color=ORANGE,
                    edgecolor=DARK, alpha=0.85)
        ax.set_xlabel("stress range (MPa)")
        ax.set_ylabel("cycle count")
        ax.set_title(f"Rainflow-extracted cycles ({len(cycles)})")
        ax.set_xlim(0, 400)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "rainflow_cycle_counting", fps=8)


# ---------------------------------------------------------------------------
# A09 — Cyclic stress field
# ---------------------------------------------------------------------------

def a09_cyclic_stress_field():
    # Synthetic 2D stress field on a fillet-T outline oscillating under load
    nx, ny = 100, 60
    x = np.linspace(0, 20, nx)
    y = np.linspace(0, 12, ny)
    X, Y = np.meshgrid(x, y)
    # Hot spot near (10, 8) — fillet toe
    def sigma_field(load):
        return load * 200.0 * np.exp(-((X - 10) ** 2 / 4.0 + (Y - 8) ** 2 / 3.0))

    n_frames = 30
    frames = []
    for i in range(n_frames):
        load = np.sin(2 * np.pi * i / n_frames)
        sig = sigma_field(load)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        im = ax.imshow(sig, extent=(0, 20, 0, 12), origin="lower",
                       cmap=_theme.get_cmap("diverging"),
                       vmin=-200, vmax=200, aspect="equal")
        # Fillet T outline
        ax.plot([0, 20, 20, 12, 12, 0, 0], [0, 0, 6, 6, 12, 12, 0],
                color="black", lw=1.5)
        ax.plot([8, 12], [6, 6], color="black", lw=1.5)
        ax.set_title(f"Von Mises stress, t = {i}/{n_frames}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="σ (MPa)")

        # Right: load vs time
        ax = axes[1]
        ts = np.arange(n_frames)
        loads = np.sin(2 * np.pi * ts / n_frames)
        ax.plot(ts, loads, color=GRAY, lw=1.5)
        ax.plot(ts[: i + 1], loads[: i + 1], color=BLUE, lw=2.5)
        ax.scatter(i, load, color=RED, s=60, zorder=5)
        ax.set_xlim(0, n_frames)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("time step")
        ax.set_ylabel("load amplitude")
        ax.set_title("Load history")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Cyclic stress field on fillet-T", fontsize=13)
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    return _write_animation(frames, "cyclic_stress_field", fps=8)


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Callable] = {
    "phase_field_crack_propagation": a01_phase_field_crack_propagation,
    "multipass_thermal_cycle": a02_multipass_thermal_cycle,
    "goldak_heat_source_sweep": a03_goldak_heat_source_sweep,
    "active_learning_convergence": a04_active_learning_convergence,
    "monte_carlo_convergence": a05_monte_carlo_convergence,
    "enkf_crack_tracking": a06_enkf_crack_tracking,
    "bayesian_posterior_update": a07_bayesian_posterior_update,
    "rainflow_cycle_counting": a08_rainflow_cycle_counting,
    "cyclic_stress_field": a09_cyclic_stress_field,
}


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Subset of animation stems")
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
            gif_path, mp4_path = gen()
            gif_kb = gif_path.stat().st_size / 1024
            mp4_kb = mp4_path.stat().st_size / 1024
            print(f"✓ {name}: {gif_kb:.0f} KB GIF, {mp4_kb:.0f} KB MP4")
            paths.extend([gif_path, mp4_path])
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")
            raise
    return paths


if __name__ == "__main__":
    main()
