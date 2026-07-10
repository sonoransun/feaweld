"""Spectrum fatigue assessment engine.

Pure functions that turn a cyclic-loading definition into a cycle set,
scale it to stress space, and assess it against an S-N curve with
optional mean-stress correction and strength knockdowns.

Cycle sets are lists of ``(stress_range, mean_stress, count)`` triples
living in one of two spaces:

- **factor space** -- ranges and means are fractions of a reference
  stress (typically the stress solved at maximum load); multiply by the
  reference via [scale_cycles][feaweld.fatigue.assessment.scale_cycles]
  before assessing.
- **absolute space** -- ranges and means are already in MPa and are
  passed to [assess_spectrum][feaweld.fatigue.assessment.assess_spectrum]
  directly, without scaling.

[build_cycle_set][feaweld.fatigue.assessment.build_cycle_set] reports
which space its output is in via the ``is_absolute`` element of its
return tuple.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import SNCurve
from feaweld.fatigue.knockdown import gerber_correction, goodman_correction
from feaweld.fatigue.miner import fatigue_life_from_damage, miner_damage
from feaweld.fatigue.rainflow import rainflow_count

_FACTOR_RANGE_KEY = "range_factor"
_FACTOR_MEAN_KEY = "mean_factor"
_ABSOLUTE_RANGE_KEY = "stress_range"
_ABSOLUTE_MEAN_KEY = "mean_stress"

_MEAN_CORRECTIONS = ("none", "goodman", "gerber")
_HISTORY_UNITS = ("load_factor", "stress")


def _load_history_file(
    history_file: str | Path,
    base_dir: str | Path | None,
) -> NDArray[np.float64]:
    """Load a 1-D load history from a text file.

    Accepts comma- or whitespace-delimited files; multi-column files
    contribute only their first column.  Relative paths are resolved
    against *base_dir* when given.
    """
    path = Path(history_file)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    try:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        data = np.loadtxt(path, ndmin=2)
    return data[:, 0].astype(np.float64)


def _blocks_to_cycles(
    blocks: Sequence[Mapping[str, float]],
) -> tuple[list[tuple[float, float, float]], bool]:
    """Convert block mappings to triples, detecting factor vs. absolute keys."""
    triples: list[tuple[float, float, float]] = []
    is_absolute: bool | None = None
    for i, block in enumerate(blocks):
        has_factor = _FACTOR_RANGE_KEY in block or _FACTOR_MEAN_KEY in block
        has_abs = _ABSOLUTE_RANGE_KEY in block or _ABSOLUTE_MEAN_KEY in block
        if has_factor and has_abs:
            raise ValueError(
                f"Spectrum block {i} mixes factor keys "
                f"('{_FACTOR_RANGE_KEY}'/'{_FACTOR_MEAN_KEY}') with absolute keys "
                f"('{_ABSOLUTE_RANGE_KEY}'/'{_ABSOLUTE_MEAN_KEY}'). "
                "Use one style per block."
            )
        if not has_factor and not has_abs:
            raise ValueError(
                f"Spectrum block {i} needs '{_FACTOR_RANGE_KEY}' or "
                f"'{_ABSOLUTE_RANGE_KEY}'."
            )
        if is_absolute is None:
            is_absolute = has_abs
        elif is_absolute != has_abs:
            raise ValueError(
                "All spectrum blocks must use the same key style "
                "(factor or absolute); mixing styles across blocks is ambiguous."
            )
        range_key = _ABSOLUTE_RANGE_KEY if has_abs else _FACTOR_RANGE_KEY
        mean_key = _ABSOLUTE_MEAN_KEY if has_abs else _FACTOR_MEAN_KEY
        if range_key not in block:
            raise ValueError(f"Spectrum block {i} is missing '{range_key}'.")
        stress_range = float(block[range_key])
        mean = float(block.get(mean_key, 0.0))
        count = float(block.get("cycles", 1.0))
        triples.append((stress_range, mean, count))
    return triples, bool(is_absolute)


def build_cycle_set(
    *,
    r_ratio: float | None = None,
    cycles: float | None = None,
    blocks: Sequence[Mapping[str, float]] = (),
    history: Sequence[float] | NDArray[np.float64] | None = None,
    history_file: str | Path | None = None,
    history_units: str = "load_factor",
    base_dir: str | Path | None = None,
) -> tuple[list[tuple[float, float, float]], str, bool] | None:
    """Build a fatigue cycle set from exactly one cyclic-loading style.

    Exactly one of *r_ratio*, *blocks*, *history*, or *history_file* may
    be given.  If none is given the function returns ``None`` (no cyclic
    definition).

    Parameters
    ----------
    r_ratio : float, optional
        Stress ratio $R = \\sigma_{min} / \\sigma_{max}$ (must be < 1).
        Produces one constant-amplitude cycle in factor space with
        range $1 - R$ and mean $(1 + R)/2$.
    cycles : float, optional
        Cycle count for the *r_ratio* style (default 1.0).  Ignored for
        other styles.
    blocks : sequence of mapping, optional
        Spectrum blocks.  Each block uses either factor keys
        (``range_factor``, ``mean_factor``) or absolute keys
        (``stress_range``, ``mean_stress`` in MPa), plus ``cycles``
        (default 1.0).  Mixing key styles within a block or across
        blocks raises ``ValueError``.  Missing mean defaults to 0.0.
    history : array_like, optional
        Inline 1-D load signal; rainflow-counted into cycles.
    history_file : str or Path, optional
        Path to a comma- or whitespace-delimited text file containing
        the load signal (first column of multi-column files).  Relative
        paths are resolved against *base_dir*.
    history_units : str
        ``"load_factor"`` (default) marks history values as fractions
        of a reference stress; ``"stress"`` marks them as absolute MPa.
        Applies to *history* and *history_file* only.
    base_dir : str or Path, optional
        Base directory for resolving a relative *history_file*.

    Returns
    -------
    tuple[list[tuple[float, float, float]], str, bool] or None
        ``(triples, kind, is_absolute)`` where each triple is
        ``(stress_range, mean_stress, count)``, *kind* is one of
        ``"constant_amplitude"``, ``"blocks"``, or ``"history"``, and
        *is_absolute* is True when the triples are already in MPa
        (skip [scale_cycles][feaweld.fatigue.assessment.scale_cycles])
        and False when they are in factor space (scale by the reference
        stress before assessing).  ``None`` when no style is given.
    """
    has_history = history is not None and np.size(history) > 0
    given = [
        name
        for name, present in (
            ("r_ratio", r_ratio is not None),
            ("blocks", bool(blocks)),
            ("history", has_history),
            ("history_file", history_file is not None),
        )
        if present
    ]
    if not given:
        return None
    if len(given) > 1:
        raise ValueError(
            "Only one cyclic-loading style may be given; got "
            + " and ".join(given)
            + "."
        )

    if r_ratio is not None:
        if r_ratio >= 1.0:
            raise ValueError(
                f"r_ratio must be < 1 (R = sigma_min / sigma_max); got {r_ratio}."
            )
        count = float(cycles) if cycles is not None else 1.0
        triples = [(1.0 - r_ratio, (1.0 + r_ratio) / 2.0, count)]
        return triples, "constant_amplitude", False

    if blocks:
        triples, is_absolute = _blocks_to_cycles(blocks)
        return triples, "blocks", is_absolute

    if history_units not in _HISTORY_UNITS:
        raise ValueError(
            f"Unknown history_units '{history_units}'. "
            f"Choose from {_HISTORY_UNITS}."
        )
    if history_file is not None:
        signal = _load_history_file(history_file, base_dir)
    else:
        signal = np.atleast_1d(np.asarray(history, dtype=np.float64))
        if signal.ndim > 1:
            signal = signal[:, 0]
    triples = rainflow_count(signal)
    return triples, "history", history_units == "stress"


def scale_cycles(
    cycles: Sequence[tuple[float, float, float]],
    reference_stress: float,
) -> list[tuple[float, float, float]]:
    """Scale factor-space cycles to absolute stress space.

    Parameters
    ----------
    cycles : sequence of tuple
        ``(range, mean, count)`` triples in factor space.
    reference_stress : float
        Reference stress (MPa) multiplying each range and mean.

    Returns
    -------
    list[tuple[float, float, float]]
        Triples with range and mean in MPa; counts unchanged.
    """
    return [
        (rng * reference_stress, mean * reference_stress, count)
        for rng, mean, count in cycles
    ]


def assess_spectrum(
    cycles_mpa: Sequence[tuple[float, float, float]],
    curve: SNCurve,
    *,
    mean_correction: str = "none",
    sigma_u: float | None = None,
    residual_mean: float = 0.0,
    strength_factor: float = 1.0,
) -> dict[str, float]:
    """Assess a stress-space cycle set against an S-N curve.

    For each cycle the amplitude is half the range and *residual_mean*
    is added to the mean.  With ``mean_correction="goodman"`` or
    ``"gerber"`` the (amplitude, mean) pair is converted to an
    equivalent fully-reversed amplitude via
    [goodman_correction][feaweld.fatigue.knockdown.goodman_correction] /
    [gerber_correction][feaweld.fatigue.knockdown.gerber_correction];
    with ``"none"`` the means -- including *residual_mean* -- are not
    used at all.  The corrected range is then divided by
    *strength_factor* (a factor below 1 inflates the applied stress)
    before Miner damage summation on *curve*.  Cycles below the curve's
    cutoff contribute zero damage.

    Parameters
    ----------
    cycles_mpa : sequence of tuple
        ``(stress_range, mean_stress, count)`` triples in MPa.
    curve : SNCurve
        S-N curve for the detail category.
    mean_correction : str
        ``"none"`` (default), ``"goodman"``, or ``"gerber"``.
    sigma_u : float, optional
        Ultimate tensile strength (MPa); required for Goodman/Gerber.
    residual_mean : float
        Residual mean stress (MPa) added to every cycle mean before the
        correction.  Has no effect when *mean_correction* is ``"none"``.
    strength_factor : float
        Combined strength-reduction factor (thickness, surface,
        environment, ...); effective range = corrected range / factor.

    Returns
    -------
    dict
        ``damage`` -- Miner damage per repeat of the cycle set.
        ``life_repeats`` -- repeats of the set to failure (1 / damage).
        ``life_cycles`` -- ``life_repeats * n_cycles_per_repeat``.
        ``n_cycles_per_repeat`` -- total cycle count in one repeat.
        ``equivalent_stress_range`` -- Miner-equivalent constant-amplitude
        range at the curve's first-segment slope (MPa).
        ``max_corrected_range`` -- largest effective range after all
        corrections (MPa).
    """
    correction = mean_correction.lower().strip() if mean_correction else "none"
    if correction not in _MEAN_CORRECTIONS:
        raise ValueError(
            f"Unknown mean_correction '{mean_correction}'. "
            f"Choose from {_MEAN_CORRECTIONS}."
        )
    if correction != "none" and sigma_u is None:
        raise ValueError(
            f"sigma_u is required for '{correction}' mean-stress correction."
        )
    if strength_factor <= 0:
        raise ValueError("strength_factor must be positive.")

    n_total = 0.0
    max_range = 0.0
    effective: list[tuple[float, float]] = []
    for stress_range, mean, count in cycles_mpa:
        n_total += count
        if stress_range <= 0:
            continue
        amp = stress_range / 2.0
        mean = mean + residual_mean
        if correction == "goodman":
            amp = goodman_correction(amp, mean, sigma_u)
        elif correction == "gerber":
            amp = gerber_correction(amp, mean, sigma_u)
        eff_range = 2.0 * amp / strength_factor
        max_range = max(max_range, eff_range)
        effective.append((eff_range, count))

    # A cycle whose corrected amplitude diverges (mean at/above sigma_u)
    # implies static failure -- damage is unbounded.
    if any(math.isinf(r) for r, _ in effective):
        damage = float("inf")
    else:
        damage = miner_damage(effective, curve)

    s_eq = 0.0
    if effective and curve.segments:
        m = curve.segments[0].m
        total_count = sum(c for _, c in effective)
        weighted = sum(c * r ** m for r, c in effective)
        if total_count > 0 and weighted > 0:
            s_eq = (weighted / total_count) ** (1.0 / m)

    life_repeats = fatigue_life_from_damage(damage)
    life_cycles = life_repeats * n_total if n_total > 0 else float("inf")

    return {
        "damage": damage,
        "life_repeats": life_repeats,
        "life_cycles": life_cycles,
        "n_cycles_per_repeat": n_total,
        "equivalent_stress_range": s_eq,
        "max_corrected_range": max_range,
    }


def spectrum_life_power_law(
    life_at_reference: float,
    cycles_factor_space: Sequence[tuple[float, float, float]],
    exponent: float,
) -> float:
    """Spectrum life for a method with its own single-slope life law.

    For a life law $N = C / S^m$, the Miner damage per repeat of a
    factor-space cycle set is
    $D = \\sum_i n_i f_i^m / N_{ref}$
    where $f_i$ is the range factor relative to the reference stress and
    $N_{ref}$ the constant-amplitude life at the reference.  Returns the
    number of spectrum repeats to failure, $1/D$ -- exact for
    single-slope power laws (e.g. Dong master curve, SED).

    Parameters
    ----------
    life_at_reference : float
        Constant-amplitude life at the reference stress (cycles).
    cycles_factor_space : sequence of tuple
        ``(range_factor, mean_factor, count)`` triples in factor space.
    exponent : float
        Slope exponent *m* of the life law.

    Returns
    -------
    float
        Repeats of the cycle set to failure; infinity when the set is
        empty or the reference life is infinite.
    """
    weighted = sum(
        count * rng ** exponent
        for rng, _mean, count in cycles_factor_space
        if rng > 0
    )
    if weighted <= 0 or math.isinf(life_at_reference):
        return float("inf")
    return life_at_reference / weighted
