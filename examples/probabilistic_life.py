"""Monte Carlo simulation for fatigue life scatter in welded joints."""

import numpy as np

from feaweld.probabilistic.monte_carlo import (
    MonteCarloEngine, MonteCarloConfig, RandomVariable,
)
from feaweld.probabilistic.distributions import (
    material_property_distributions, geometric_tolerance_distributions,
)


def main():
    print("Probabilistic Fatigue Life Assessment")
    print("=" * 50)

    # Define random variables
    variables = [
        RandomVariable(
            name="stress_range",
            distribution="normal",
            params={"mean": 100.0, "std": 15.0},
        ),
        RandomVariable(
            name="yield_strength",
            distribution="lognormal",
            params={"mean": np.log(250), "std": 0.07},
        ),
        RandomVariable(
            name="weld_toe_angle",
            distribution="normal",
            params={"mean": 45.0, "std": 5.0},
        ),
    ]

    config = MonteCarloConfig(
        n_samples=5000,
        method="lhs",
        seed=42,
    )

    def fatigue_analysis(params):
        """Simple fatigue life model: N = C / (S * K_t)^m."""
        S = abs(params["stress_range"])
        theta = params["weld_toe_angle"]

        # SCF from toe angle
        K_t = 1.0 + 0.5 * (theta / 45.0)

        # FAT90: C = 90^3 * 2e6
        C = 90.0**3 * 2e6
        effective_stress = S * K_t
        if effective_stress <= 0:
            return 1e12
        return C / effective_stress**3

    print("\nRunning Monte Carlo simulation...")
    engine = MonteCarloEngine(variables, config)
    result = engine.run(fatigue_analysis)

    print(f"\nResults ({config.n_samples} samples, {config.method}):")
    print(f"  Mean life:   {result.mean:.0f} cycles")
    print(f"  Std dev:     {result.std:.0f} cycles")
    print(f"  COV:         {result.cov:.3f}")
    print(f"  5th %%ile:   {result.percentiles[5]:.0f} cycles")
    print(f"  50th %%ile:  {result.percentiles[50]:.0f} cycles")
    print(f"  95th %%ile:  {result.percentiles[95]:.0f} cycles")

    # Log-normal fit for fatigue life
    log_results = np.log10(result.results[result.results > 0])
    print(f"\n  Mean log10(N): {np.mean(log_results):.2f}")
    print(f"  Std  log10(N): {np.std(log_results):.2f}")

    # Probability of failure before target
    target_life = 1e6
    p_fail = np.mean(result.results < target_life)
    print(f"\n  P(failure before {target_life:.0e} cycles): {p_fail*100:.1f}%")

    return result


if __name__ == "__main__":
    main()
