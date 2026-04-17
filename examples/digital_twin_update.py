"""Bayesian updating of a fatigue model from synthetic sensor data.

Defines priors over two material parameters (Young's modulus and a
damage-rate coefficient), runs a simple forward model, then
sequentially updates the posterior as two batches of noisy strain
measurements arrive. Prints posterior means and 95 % credible
intervals at each step — the core loop of a digital-twin integration.
"""

import numpy as np

from feaweld.digital_twin.bayesian import BayesianUpdater, PriorSpec, ObservedData


def main():
    print("Digital Twin — Bayesian Model Update")
    print("=" * 50)

    priors = [
        PriorSpec(name="E", distribution="normal",
                  params={"mean": 200000.0, "std": 15000.0},
                  bounds=(150000.0, 250000.0)),
        PriorSpec(name="damage_rate", distribution="lognormal",
                  params={"mean": np.log(1e-5), "std": 0.4},
                  bounds=(1e-7, 1e-3)),
    ]

    # Forward model: predicts strain at 3 gauge locations under a 100 MPa load.
    sensor_positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    load_stress = 100.0  # MPa

    def forward_model(params: dict[str, float]) -> np.ndarray:
        # Simple Hookean strain with a damage-rate-driven stiffness knockdown.
        E = params["E"]
        d = params["damage_rate"]
        effective_E = E * (1.0 - 1000.0 * d)
        strain = load_stress / max(effective_E, 1.0)
        return np.full(len(sensor_positions), strain)

    # Ground truth (what we are trying to recover)
    true_params = {"E": 195000.0, "damage_rate": 2.0e-5}
    true_strain = forward_model(true_params)[0]
    print(f"  Ground truth: E = {true_params['E']:.0f} MPa, "
          f"damage_rate = {true_params['damage_rate']:.1e}")
    print(f"  Predicted strain under {load_stress} MPa: {true_strain:.3e}")

    rng = np.random.default_rng(0)

    updater = BayesianUpdater(
        priors=priors,
        forward_model=forward_model,
        n_walkers=32,
        n_steps=400,
        n_burnin=100,
    )

    for batch_idx in range(2):
        noisy = true_strain * (1.0 + 0.02 * rng.standard_normal(len(sensor_positions)))
        obs = ObservedData(
            measurement_type="strain",
            positions=sensor_positions,
            values=noisy,
            uncertainty=0.02 * true_strain,
            timestamp=float(batch_idx),
        )

        print(f"\n  Update {batch_idx + 1}: strains = {noisy.round(6)}")
        try:
            summary = updater.update(obs)
        except ImportError as exc:
            print(f"    Skipped (optional dep missing): {exc}")
            return None

        for name in summary.parameter_names:
            lo, hi = summary.ci_95[name]
            print(f"    {name:<12s}  mean = {summary.means[name]:>10.3g}  "
                  f"95% CI = [{lo:.3g}, {hi:.3g}]")

    return updater


if __name__ == "__main__":
    main()
