"""Matplotlib plots for metallurgical and material-model data.

Renders CCT diagrams, residual-stress profiles, Norton-Bailey creep curves,
and Hall-Petch grain-size strengthening relationships used across the
`feaweld.multiscale` and `feaweld.data` modules.

All matplotlib imports are deferred via
`feaweld.visualization.plots_2d._require_matplotlib` so that the rest of
feaweld remains usable without the optional matplotlib dependency.  Every
public function follows the house signature style::

    (*, title=None, show=True, ax=None) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


_EQ_BOX = dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", alpha=0.9, edgecolor="#bdc3c7")


# ---------------------------------------------------------------------------
# 1. CCT diagram
# ---------------------------------------------------------------------------

def plot_cct_diagram(
    diagram: Any,
    *,
    grade: str | None = None,
    cooling_rate: float | None = None,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Stackplot of phase fractions versus cooling rate for a CCT diagram.

    Parameters
    ----------
    diagram : feaweld.multiscale.meso.CCTDiagram or str
        A CCT diagram, or a steel-grade string which is resolved via
        [feaweld.multiscale.meso.cct_for_grade][].
    grade : str, optional
        Grade label for the title (auto-filled when *diagram* is a string).
    cooling_rate : float, optional
        If given, a vertical marker is drawn at this cooling rate (C/s) and
        annotated with the predicted phase mix from ``diagram.predict_phases``.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes
    from feaweld.visualization.theme import (
        FEAWELD_BLUE, FEAWELD_GREEN, FEAWELD_ORANGE, FEAWELD_RED,
    )

    if isinstance(diagram, str):
        from feaweld.multiscale.meso import cct_for_grade
        grade = grade or diagram
        diagram = cct_for_grade(diagram)

    plt = _require_matplotlib()
    resolved_title = title or (f"CCT Diagram: {grade}" if grade else "CCT Diagram")
    fig, ax = _prepare_axes(plt, ax, resolved_title)

    rates = np.asarray(diagram.cooling_rates, dtype=np.float64)

    def _phase(arr: Any) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        return a if a.size == rates.size else np.zeros_like(rates)

    ferrite = _phase(diagram.ferrite_fraction)
    pearlite = _phase(diagram.pearlite_fraction)
    bainite = _phase(diagram.bainite_fraction)
    martensite = _phase(diagram.martensite_fraction)

    ax.stackplot(
        rates, ferrite, pearlite, bainite, martensite,
        labels=["Ferrite", "Pearlite", "Bainite", "Martensite"],
        colors=[FEAWELD_BLUE, FEAWELD_GREEN, FEAWELD_ORANGE, FEAWELD_RED],
        alpha=0.85,
    )

    ax.set_xscale("log")
    if rates.size:
        ax.set_xlim(float(rates.min()), float(rates.max()))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cooling rate (°C/s)")
    ax.set_ylabel("Phase fraction")
    ax.legend(loc="center left", fontsize="small")

    # Transformation-temperature text box.
    ax.text(
        0.98, 0.98,
        f"Ac1 = {diagram.Ac1:.0f} °C\nAc3 = {diagram.Ac3:.0f} °C\nMs = {diagram.Ms:.0f} °C",
        transform=ax.transAxes, fontsize=8, ha="right", va="top", bbox=_EQ_BOX,
    )

    # Optional cooling-rate marker with predicted phase mix.
    if cooling_rate is not None:
        ax.axvline(cooling_rate, color="black", linestyle="--", linewidth=1.2)
        phases = diagram.predict_phases(cooling_rate)
        mix = (
            f"@ {cooling_rate:g} °C/s\n"
            f"F {phases.ferrite:.0%}  P {phases.pearlite:.0%}\n"
            f"B {phases.bainite:.0%}  M {phases.martensite:.0%}"
        )
        ax.annotate(
            mix,
            xy=(cooling_rate, 1.0),
            xytext=(8, -8), textcoords="offset points",
            fontsize=7, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"),
        )

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Residual-stress profile
# ---------------------------------------------------------------------------

def plot_residual_stress_profile(
    profile: Any,
    *,
    yield_strength: float | None = None,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Through-thickness residual-stress profile.

    Stress is plotted on the x-axis against normalised through-thickness
    position ``z/t`` on the y-axis (matching
    [feaweld.visualization.plots_2d.plot_through_thickness][]).

    Parameters
    ----------
    profile : feaweld.data.residual_stress.ResidualStressProfile or str
        A residual-stress profile, or a profile-name string resolved via
        `feaweld.data.residual_stress.get_residual_profile`.
    yield_strength : float, optional
        Material yield strength (MPa).  When given the profile is shown in
        MPa; otherwise it is shown normalised as ``sigma / sigma_y``.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes
    from feaweld.visualization.theme import FEAWELD_BLUE

    if isinstance(profile, str):
        from feaweld.data.residual_stress import get_residual_profile
        profile = get_residual_profile(profile)

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or f"Residual Stress Profile: {profile.name}")

    z = np.asarray(profile.z_over_t, dtype=np.float64)
    s_norm = np.asarray(profile.stress_over_sy, dtype=np.float64)

    if yield_strength is not None:
        stress = s_norm * yield_strength
        xlabel = "Residual stress (MPa)"
    else:
        stress = s_norm
        xlabel = r"Residual stress $\sigma / \sigma_y$"

    ax.plot(stress, z, color=FEAWELD_BLUE, linewidth=1.6, marker="o", markersize=4,
            label=profile.name)

    # Zero-stress reference line.
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised through-thickness $z/t$")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Caption from the profile's standard / weld type.
    ax.text(
        0.02, 0.02, f"{profile.standard} — {profile.weld_type}",
        transform=ax.transAxes, fontsize=8, ha="left", va="bottom", bbox=_EQ_BOX,
    )

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. Norton-Bailey creep curve
# ---------------------------------------------------------------------------

def plot_creep_curve(
    A: float,
    n: float,
    m: float,
    *,
    stress_levels: Sequence[float],
    t_end: float,
    n_points: int = 200,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Norton-Bailey creep-strain curves at several constant stress levels.

    The time-hardening rate law ``eps_dot = A * sigma**n * t**m`` integrates
    at constant stress to::

        eps(t) = A * sigma**n * t**(m + 1) / (m + 1)

    Parameters
    ----------
    A : float
        Creep pre-factor.
    n : float
        Stress exponent.
    m : float
        Time exponent.
    stress_levels : sequence of float
        Constant stress levels (MPa), one curve each.
    t_end : float
        End time (s) of the curves.
    n_points : int
        Number of log-spaced time samples per curve.
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or "Norton-Bailey Creep Curve")

    t_end = float(t_end)
    t0 = max(t_end / 1e4, 1e-9)
    t = np.logspace(np.log10(t0), np.log10(t_end), n_points)

    exp = m + 1.0
    for s in stress_levels:
        eps = A * (float(s) ** n) * t ** exp / exp
        ax.loglog(t, eps, linewidth=1.5, label=f"σ = {float(s):g} MPa")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Creep strain $\\varepsilon$")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 4. Hall-Petch relationship
# ---------------------------------------------------------------------------

def plot_hall_petch(
    params: Any,
    *,
    grain_size_range: tuple[float, float] = (1.0, 100.0),
    markers: Mapping[str, float] | None = None,
    title: str | None = None,
    show: bool = True,
    ax: Any = None,
) -> Any:
    """Yield strength versus grain size via the Hall-Petch relation.

    Parameters
    ----------
    params : feaweld.multiscale.micro.HallPetchParams or dict
        A single Hall-Petch parameter set, or a mapping ``{label: params}``
        to overlay several curves.
    grain_size_range : tuple of float
        ``(min, max)`` grain size (μm) for the log-spaced x-axis.
    markers : dict, optional
        Mapping ``{label: grain_size_um}`` placing a labelled point on the
        matching curve (or on the single curve when *params* is one object).
    title : str, optional
        Plot title.
    show : bool
        Call ``plt.show()`` when *True*.
    ax : matplotlib.axes.Axes, optional
        Reuse an existing Axes; a new Figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from feaweld.visualization.plots_2d import _require_matplotlib, _prepare_axes

    plt = _require_matplotlib()
    fig, ax = _prepare_axes(plt, ax, title or "Hall-Petch Relationship")

    d = np.logspace(
        np.log10(grain_size_range[0]),
        np.log10(grain_size_range[1]),
        200,
    )

    if isinstance(params, Mapping):
        curves = dict(params)
        single = False
    else:
        curves = {"Hall-Petch": params}
        single = True

    for label, p in curves.items():
        sigma = np.asarray(p.yield_strength(d), dtype=np.float64)
        line, = ax.plot(d, sigma, linewidth=1.6, label=label)

        # Marker(s) on this curve.
        curve_markers: dict[str, float] = {}
        if markers:
            if label in markers:
                curve_markers = {label: markers[label]}
            elif single:
                curve_markers = dict(markers)
        for mlabel, gm in curve_markers.items():
            ym = float(p.yield_strength(gm))
            ax.plot(gm, ym, "o", color=line.get_color(), markersize=8, zorder=5)
            ax.annotate(
                f"{mlabel}\n{gm:g} μm, {ym:.0f} MPa",
                xy=(gm, ym), xytext=(6, 6), textcoords="offset points",
                fontsize=7, color=line.get_color(),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Grain size d (μm)")
    ax.set_ylabel("Yield strength $\\sigma_y$ (MPa)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
