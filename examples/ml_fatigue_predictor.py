"""Train an ML fatigue-life predictor on synthetic IIW-class data.

Generates a synthetic training set from FAT90-class behaviour with
realistic scatter, trains a Random Forest predictor, reports the
cross-validation RMSE / R^2, then predicts fatigue life for a new
case and prints the top feature importances.
"""

import numpy as np

from feaweld.ml.features import FatigueFeatures, standard_feature_names
from feaweld.ml.models import FatiguePredictor, MLModelConfig


def main():
    print("ML Fatigue Life Predictor")
    print("=" * 50)

    rng = np.random.default_rng(42)
    n_samples = 400
    features = standard_feature_names()

    # Sample a realistic distribution for each feature.
    X = np.zeros((n_samples, len(features)))
    X[:, 0] = rng.uniform(40, 200, n_samples)        # stress_range (MPa)
    X[:, 1] = rng.uniform(-1.0, 0.8, n_samples)      # r_ratio
    X[:, 2] = rng.uniform(6, 40, n_samples)          # plate_thickness
    X[:, 3] = rng.uniform(0, 180, n_samples)         # sigma_m
    X[:, 4] = rng.uniform(0, 60, n_samples)          # sigma_b
    X[:, 5] = rng.uniform(1.0, 3.5, n_samples)       # SCF
    X[:, 6] = rng.normal(500, 40, n_samples)         # UTS
    X[:, 7] = rng.normal(350, 30, n_samples)         # yield
    X[:, 8] = rng.normal(45, 7, n_samples)           # toe angle
    X[:, 9] = rng.uniform(0.1, 2.0, n_samples)       # toe radius
    X[:, 10] = rng.uniform(0, 1.0, n_samples)        # misalignment
    X[:, 11] = rng.uniform(0, 0.8, n_samples)        # residual stress ratio
    X[:, 12] = X[:, 0] * X[:, 5] * 0.9               # hotspot ~ range * SCF

    # Synthetic target: FAT90-like S^-3 law + random scatter in log10(N).
    C = 90.0 ** 3 * 2e6
    life = C / (X[:, 12].clip(1.0)) ** 3
    log_life = np.log10(life) + rng.normal(0, 0.2, n_samples)

    training = FatigueFeatures(feature_names=features, values=X, target=log_life)
    predictor = FatiguePredictor(MLModelConfig(model_type="random_forest", n_estimators=300))

    metrics = predictor.train(training)
    print(f"  Training metrics:")
    print(f"    RMSE (log10 N): {metrics['rmse']:.3f}")
    print(f"    R^2:            {metrics['r2']:.3f}")
    print(f"    CV RMSE (5-fold mean): {np.mean(metrics['cv_scores']):.3f}")

    # Predict a new case: high SCF, high stress.
    new_case = {
        "stress_range": 120.0, "r_ratio": 0.1, "plate_thickness": 20.0,
        "structural_stress_membrane": 100.0, "structural_stress_bending": 20.0,
        "scf": 2.8, "material_uts": 500.0, "material_yield": 345.0,
        "weld_toe_angle": 45.0, "toe_radius": 0.5, "misalignment": 0.3,
        "residual_stress_ratio": 0.3, "hotspot_stress": 120.0 * 2.8,
    }
    pred = predictor.predict(new_case)

    print(f"\n  Prediction for new case (hot-spot ~ {new_case['hotspot_stress']:.0f} MPa):")
    print(f"    N = {pred.predicted_life:.2e} cycles "
          f"[CI: {pred.confidence_interval[0]:.2e}, {pred.confidence_interval[1]:.2e}]")

    top = sorted(pred.feature_importances.items(), key=lambda kv: -kv[1])[:5]
    print(f"\n  Top feature importances:")
    for name, imp in top:
        print(f"    {name:<30s} {imp:.3f}")

    return predictor


if __name__ == "__main__":
    main()
