# ML fatigue prediction

feaweld ships a machine-learning fatigue predictor that learns $\log_{10} N$ from a
table of weld features. It complements the physics-based methods: train it on your
own test or service data and it captures effects the closed-form models miss, with
a confidence interval on every prediction. Random Forest, XGBoost, and a voting
ensemble are supported, plus transfer learning to fine-tune a base model on
plant-specific data.

!!! note "Install the ML extra"
    The ML commands require scikit-learn (and XGBoost for the `xgboost` /
    `ensemble` models): `pip install -e ".[ml]"`.

## Feature engineering

The standard feature set is defined by `standard_feature_names()` in
`ml/features.py` — 13 features spanning stress, geometry, material, and residual
state:

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `stress_range` | Applied stress range (MPa) |
| 2 | `r_ratio` | Stress ratio $R = \sigma_{min}/\sigma_{max}$ |
| 3 | `plate_thickness` | Plate thickness $t$ (mm) |
| 4 | `structural_stress_membrane` | Membrane part of structural stress |
| 5 | `structural_stress_bending` | Bending part of structural stress |
| 6 | `scf` | Stress concentration factor |
| 7 | `material_uts` | Ultimate tensile strength (MPa) |
| 8 | `material_yield` | Yield strength (MPa) |
| 9 | `weld_toe_angle` | Weld toe angle (deg) |
| 10 | `toe_radius` | Weld toe radius (mm) |
| 11 | `misalignment` | Axial misalignment $e$ (mm) |
| 12 | `residual_stress_ratio` | Residual stress / yield |
| 13 | `hotspot_stress` | Hot-spot stress (MPa) |

You do not have to supply all 13 — `extract_features()` fills what it can from an
`FEAResults` + `WeldLineDefinition`, and `build_feature_matrix()` NaN-fills the
rest (the models impute missing values with the column median at train time).

## CSV format for training

`feaweld ml train` reads a headed CSV where each column is a feature and one column
is the target. The default target column is `log_life` (log₁₀ of the cycles to
failure). Any column that is not the target is treated as a feature:

```csv
stress_range,r_ratio,plate_thickness,scf,material_uts,log_life
180,0.1,20,1.8,500,5.42
140,0.1,20,1.6,500,5.98
120,0.1,25,1.5,520,6.31
```

## Commands

### Train

```bash
feaweld ml train fatigue_data.csv -m random_forest --target log_life -o fatigue_model.joblib
```

```
Training random_forest on 240 samples, 5 features

  RMSE (log10 N): 0.1832
  R^2:            0.9041
  CV RMSE:        0.2015 +/- 0.0223

Model saved: fatigue_model.joblib
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-m/--model` | `random_forest` | `random_forest`, `xgboost`, or `ensemble` |
| `--target` | `log_life` | Name of the target column (log₁₀ N) |
| `-o/--output` | `fatigue_model.joblib` | Saved model path |

### Predict

Provide feature values inline (repeat `-s`), or a CSV of rows with `-i`:

```bash
feaweld ml predict fatigue_model.joblib -s stress_range=150 -s scf=1.7 -s plate_thickness=20
```

```
  N = 6.310e+05 cycles (95% CI: 4.12e+05 .. 9.66e+05)

Top feature importances:
                stress_range: 0.512
                         scf: 0.221
             plate_thickness: 0.104
```

The 95 % confidence interval comes from the spread of the individual trees'
predictions; it widens where the training data is sparse.

| Option | Meaning |
|--------|---------|
| `-s/--set name=value` | One feature assignment; repeatable |
| `-i/--input CSV` | Predict every row of a headed CSV instead |

### Transfer learning

Fine-tune a trained base model on a smaller plant-specific dataset. The learner
trains a residual-correction model on top of the base predictor (in log-life
space):

```bash
feaweld ml transfer fatigue_model.joblib plant_data.csv --target log_life -o fatigue_model_tuned.joblib
```

```
  RMSE base:      0.4120
  RMSE corrected: 0.1875
  R^2 corrected:  0.8630

Fine-tuned model saved: fatigue_model_tuned.joblib
```

The `RMSE base` vs. `RMSE corrected` comparison quantifies how much the local data
improved the fit.

## Python API

```python
from feaweld.ml.models import FatiguePredictor, MLModelConfig
from feaweld.ml.features import standard_feature_names, build_feature_matrix

predictor = FatiguePredictor(MLModelConfig(model_type="ensemble"))
metrics = predictor.train(features)                 # {"rmse", "r2", "cv_scores"}
pred = predictor.predict({"stress_range": 150, "scf": 1.7})
print(pred.predicted_life, pred.confidence_interval)
predictor.save("model.joblib")
```

`MLModelConfig` exposes `model_type`, `n_estimators` (500), `max_depth` (8),
`cv_folds` (5), and `random_state` (42). The shipped example
`examples/ml_fatigue_predictor.py` generates synthetic training data, trains a
model, and demonstrates transfer learning end to end.

## See also

- [Probabilistic & reliability](probabilistic.md) — the closed-form response model,
  a physics-based counterpart.
- [API reference](../api/ml.md) — full `ml` package documentation.
