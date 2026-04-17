# Examples

Runnable scripts in the `examples/` directory demonstrate common feaweld workflows. Each script is self-contained and prints a summary to stdout.

Run any of them with:

```bash
python examples/<name>.py
```

## Fatigue and structural

### `fillet_t_joint.py`

Full pipeline on a fillet T-joint under axial tension — geometry, mesh, linear-elastic solve, three post-processing methods (hot-spot, Dong, Blodgett), IIW FAT90 fatigue, and HTML report. This is the recommended starting example.

### `butt_weld_fatigue.py`

Butt weld under variable-amplitude loading. Demonstrates rainflow cycle counting (ASTM E1049), Palmgren-Miner cumulative damage, and Goodman mean-stress correction.

### `pwht_comparison.py`

Post-weld heat treatment comparison study — compares as-welded vs. PWHT residual stress state and its impact on fatigue life.

### `probabilistic_life.py`

Monte Carlo fatigue assessment with Latin Hypercube Sampling. Treats weld leg size, material strength, and load as random variables and produces a life distribution with reliability-index output.

## Thermal and creep

### `thermal_goldak.py`

Transient thermal solve with a Goldak double-ellipsoid heat source traveling along a weld path. Extracts peak temperature history and cooling rate.

### `creep_norton_bailey.py`

Norton-Bailey creep relaxation during PWHT. Evolves residual stress from post-welding state through a held hold temperature and reports the relaxed stress field.

## Data-driven

### `ml_fatigue_predictor.py`

Trains an XGBoost / Random Forest fatigue-life predictor on the bundled IIW/DNV S-N dataset and predicts remaining life on a held-out case.

### `digital_twin_update.py`

Bayesian updating of a fatigue model using a synthetic sensor stream. Demonstrates MCMC prior → posterior updates on damage parameters and a live alert when the predicted life drops below threshold.

## Tips

- All examples default to `results/<example_name>/` as the output directory; delete it between runs to avoid stale reports.
- If you don't have a FEA backend installed, the solve-based examples will raise `ImportError` with instructions — install `feaweld[fenics]` or `feaweld[calculix]`.
- `probabilistic_life.py` and `ml_fatigue_predictor.py` are CPU-intensive; they run in under a minute on a laptop but scale linearly with sample count.
