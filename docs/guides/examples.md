# Examples

Runnable scripts in the `examples/` directory demonstrate common feaweld workflows. Each script is self-contained and prints a summary to stdout.

Run any of them with:

```bash
python examples/<name>.py
```

## YAML cases

The `examples/` directory also ships four declarative cases driven by the CLI
rather than Python (see `examples/README.md` for details):

- **`fillet_t_joint.yaml`** — a single fillet T-joint analysis mirroring the
  [quickstart](../quickstart.md). Run it with `feaweld run examples/fillet_t_joint.yaml`.
- **`fillet_t_joint_3d.yaml`** — the same joint extruded into a solid
  (`geometry.dimension: 3`, 40 mm weld length): coarse TET4 mesh with toe-line
  refinement, per-station hot spot along each weld-toe line, and an `R = 0.1`
  `fatigue:` block on top. Walked through in the
  [3D analysis guide](analysis_3d.md). Run it with
  `feaweld run examples/fillet_t_joint_3d.yaml` — meshing alone works on the
  core install; the solve needs a backend (FEniCS requires `element_order: 1`
  in 3D).
- **`leg_sweep_study.yaml`** — the weld-leg-size parametric study from the
  [parametric study tutorial](../tutorials/02_parametric_study.md). Run it with
  `feaweld study run examples/leg_sweep_study.yaml -j 4`.
- **`spectrum_fatigue.yaml`** — a butt weld under the variable-amplitude stress
  history in `load_history.csv`: rainflow + Miner damage on the EC3 category-90
  curve with Goodman mean-stress correction and a PWHT-relaxed residual stress.
  Walked through in the [fatigue assessment guide](fatigue_assessment.md). Run it
  with `feaweld run examples/spectrum_fatigue.yaml`, or assess the history alone
  (no FEA backend) with `feaweld fatigue --history examples/load_history.csv -c EC3_90`.

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

Samples the volumetric heat input of a traveling Goldak double-ellipsoid heat source on a regular grid and reports the peak power density and its location at a snapshot time. No transient solve — this is the source field the transient thermal solver integrates.

### `creep_norton_bailey.py`

Norton-Bailey creep relaxation during PWHT. Evolves residual stress from post-welding state through a held hold temperature and reports the relaxed stress field.

## Data-driven

### `ml_fatigue_predictor.py`

Trains a Random Forest fatigue-life predictor on synthetic FAT90-class data generated in-script with realistic scatter, reports the cross-validation RMSE / R², then predicts fatigue life for a new case and prints the top feature importances.

### `digital_twin_update.py`

Bayesian updating of a fatigue model using a synthetic sensor stream. Demonstrates MCMC prior → posterior updates on damage parameters and a live alert when the predicted life drops below threshold.

## Tips

- All examples default to `results/<example_name>/` as the output directory; delete it between runs to avoid stale reports.
- If you don't have a FEA backend installed, the solve-based examples will raise `ImportError` with instructions — install `feaweld[fenics]` or `feaweld[calculix]`.
- `probabilistic_life.py` and `ml_fatigue_predictor.py` are CPU-intensive; they run in under a minute on a laptop but scale linearly with sample count.
