"""Pre-defined distribution models for weld analysis uncertainty.

Provides ready-made :class:`~feaweld.probabilistic.monte_carlo.RandomVariable`
collections for common sources of scatter in welded joint assessment.
"""

from __future__ import annotations

import math

from feaweld.probabilistic.monte_carlo import RandomVariable


# ---------------------------------------------------------------------------
# Material property scatter
# ---------------------------------------------------------------------------


def material_property_distributions(material_name: str) -> list[RandomVariable]:
    """Return standard distributions for material scatter.

    Parameters
    ----------
    material_name : str
        A material identifier (e.g. ``"S355"``).  Currently the same relative
        scatter model is used for all steels; *material_name* selects the
        nominal property values.

    Returns
    -------
    list[RandomVariable]
        Four random variables: yield strength, UTS, elastic modulus, and
        fatigue S-N curve scatter (on life).
    """

    # Nominal properties keyed by common structural steels
    _defaults: dict[str, dict[str, float]] = {
        "S235": {"yield": 235.0, "uts": 360.0, "E": 210_000.0},
        "S275": {"yield": 275.0, "uts": 410.0, "E": 210_000.0},
        "S355": {"yield": 355.0, "uts": 490.0, "E": 210_000.0},
        "S460": {"yield": 460.0, "uts": 540.0, "E": 210_000.0},
    }

    key = material_name.upper()
    props = _defaults.get(key)
    if props is None:
        # Fall back to generic medium-carbon steel
        props = {"yield": 350.0, "uts": 500.0, "E": 210_000.0}

    yield_mean = props["yield"]
    uts_mean = props["uts"]
    E_mean = props["E"]

    # Lognormal parameters: given COV and physical mean,
    # underlying normal mean mu = ln(phys_mean) - 0.5*sigma^2 and sigma = COV
    def _lognormal_params(phys_mean: float, cov: float) -> dict[str, float]:
        sigma = cov  # approximate for small COV
        mu = math.log(phys_mean) - 0.5 * sigma ** 2
        return {"mean": mu, "std": sigma}

    return [
        RandomVariable(
            name="yield_strength",
            distribution="lognormal",
            params=_lognormal_params(yield_mean, 0.07),
        ),
        RandomVariable(
            name="uts",
            distribution="lognormal",
            params=_lognormal_params(uts_mean, 0.05),
        ),
        RandomVariable(
            name="elastic_modulus",
            distribution="normal",
            params={"mean": E_mean, "std": E_mean * 0.03},
        ),
        RandomVariable(
            name="fatigue_life_scatter",
            distribution="lognormal",
            params=_lognormal_params(1.0, 0.20),  # multiplier on life
        ),
    ]


# ---------------------------------------------------------------------------
# Geometric tolerance scatter
# ---------------------------------------------------------------------------


def geometric_tolerance_distributions(
    weld_type: str = "fillet",
) -> list[RandomVariable]:
    """Return distributions for geometric variability.

    Parameters
    ----------
    weld_type : str
        ``"fillet"`` or ``"butt"``.

    Returns
    -------
    list[RandomVariable]
        Weld toe angle, weld toe radius, misalignment, and weld leg-size
        variation.
    """

    if weld_type == "fillet":
        toe_angle_mean = 45.0
        toe_angle_std = 5.0
        leg_size_nominal = 6.0  # mm default
    elif weld_type == "butt":
        toe_angle_mean = 30.0
        toe_angle_std = 4.0
        leg_size_nominal = 0.0  # not applicable; use throat instead
    else:
        toe_angle_mean = 45.0
        toe_angle_std = 5.0
        leg_size_nominal = 6.0

    variables = [
        RandomVariable(
            name="weld_toe_angle",
            distribution="normal",
            params={"mean": toe_angle_mean, "std": toe_angle_std},
        ),
        RandomVariable(
            name="weld_toe_radius",
            distribution="lognormal",
            params={
                "mean": math.log(1.0) - 0.5 * 0.5 ** 2,  # physical mean ~1.0 mm
                "std": 0.5,
            },
        ),
        RandomVariable(
            name="misalignment",
            distribution="normal",
            params={"mean": 0.0, "std": 0.5},  # mm
        ),
    ]

    if leg_size_nominal > 0:
        variables.append(
            RandomVariable(
                name="weld_leg_size",
                distribution="normal",
                params={
                    "mean": leg_size_nominal,
                    "std": leg_size_nominal * 0.10,
                },
            )
        )

    return variables


# ---------------------------------------------------------------------------
# Corrosion scatter
# ---------------------------------------------------------------------------


_CORROSION_PARAMS: dict[str, dict[str, float]] = {
    "marine": {"loc": 0.5, "scale": 0.3},          # mm pit depth
    "industrial": {"loc": 0.3, "scale": 0.2},
    "rural": {"loc": 0.1, "scale": 0.08},
    "splash_zone": {"loc": 1.0, "scale": 0.5},
}


def corrosion_distribution(environment: str = "marine") -> RandomVariable:
    """Return a Gumbel-distributed pit-depth variable.

    Parameters
    ----------
    environment : str
        ``"marine"``, ``"industrial"``, ``"rural"``, or ``"splash_zone"``.

    Returns
    -------
    RandomVariable
    """

    key = environment.lower()
    params = _CORROSION_PARAMS.get(key)
    if params is None:
        raise ValueError(
            f"Unknown environment '{environment}'. "
            f"Choose from {sorted(_CORROSION_PARAMS)}."
        )

    return RandomVariable(
        name="pit_depth",
        distribution="gumbel",
        params=params,
    )
