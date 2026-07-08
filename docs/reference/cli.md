# CLI reference

Every feaweld command is a subcommand of `feaweld`. Run `feaweld --help` or
`feaweld <command> --help` for the built-in help. Arguments are positional;
options take `--flag value` (or the short `-x value` where shown).

| Command | Purpose | Guide |
|---------|---------|-------|
| [`run`](#run) | Full analysis from a YAML case | [YAML analysis](../tutorials/01_yaml_analysis.md) |
| [`blodgett`](#blodgett) | Weld-group hand calculations | [Custom post-processing](../tutorials/03_custom_postprocessing.md) |
| [`visualize`](#visualize) | Render an FEA result file | [Visualization](../guides/visualization.md) |
| [`goldak`](#goldak) | Render a Goldak heat source | [Solvers](../guides/solvers.md) |
| [`dashboard`](#dashboard) | Engineering dashboard PNG | [Visualization](../guides/visualization.md) |
| [`animate`](#animate) | Damage-accumulation animation | [Visualization](../guides/visualization.md) |
| [`materials`](#materials) | List bundled materials | — |
| [`study run`](#study-run) / [`study compare`](#study-compare) | Parametric studies | [Parametric study](../tutorials/02_parametric_study.md) |
| [`reliability mc`](#reliability-mc) / [`form`](#reliability-form) / [`sobol`](#reliability-sobol) | Probabilistic analysis | [Probabilistic & reliability](../guides/probabilistic.md) |
| [`ml train`](#ml-train) / [`predict`](#ml-predict) / [`transfer`](#ml-transfer) | ML fatigue prediction | [ML fatigue prediction](../guides/ml.md) |
| [`multiscale`](#multiscale) | Weld-zone property estimation | [Multiscale modeling](../guides/multiscale.md) |
| [`convergence`](#convergence) | Mesh convergence study | [Convergence & submodeling](../guides/convergence.md) |
| [`submodel`](#submodel) | Global-local submodeling | [Convergence & submodeling](../guides/convergence.md) |
| [`twin dashboard`](#twin-dashboard) / [`update`](#twin-update) | Digital twin | [Digital twin](../guides/digital_twin.md) |

---

## `run`

Run a complete analysis from a YAML case file and write an HTML report.

```bash
feaweld run case.yaml
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Path to the analysis-case YAML |
| `-o/--output` | case's `output_dir` | Output directory for results |
| `--report/--no-report` | `--report` | Generate the HTML report |
| `--interactive/--static` | `--static` | Embed interactive Plotly figures (needs `feaweld[viz]`) |

## `blodgett`

Weld-group section properties, stresses, and AISC capacities — no FEA required.

```bash
feaweld blodgett -g box --d 100 --b 50 -t 5 -P 10000
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-g/--geometry` | `line` | `line`, `parallel`, `c_shape`, `l_shape`, `box`, `circular`, `i_shape`, `t_shape`, `u_shape` |
| `--d` | required | Primary dimension d (mm) |
| `-b/--b` | `0.0` | Secondary dimension b (mm) |
| `-t/--throat` | required | Weld throat thickness (mm) |
| `-P/--axial` | `0.0` | Axial force (N) |
| `-V/--shear` | `0.0` | Shear force (N) |
| `-M/--moment` | `0.0` | Bending moment (N·mm) |
| `-T/--torsion` | `0.0` | Torsion (N·mm) |
| `--fexx` | `483.0` | Electrode strength F_EXX (MPa) |

## `visualize`

Render an FEA result file (`.vtk` / `.vtu`) through the feaweld visualization
library (semantic theme colormaps, consistent annotations). Exactly one view mode
applies, in precedence order **`--iso` → `--threshold` → `--clip` → plain
contour**; `--deformed` and `--annotate` combine with the plain contour view.

```bash
feaweld visualize results.vtu -c von_mises -o stress.png
```

| Option | Default | Meaning |
|--------|---------|---------|
| `results_file` (arg) | required | VTK/VTU result file |
| `-c/--component` | `von_mises` | `von_mises`, `tresca`, `xx`, `yy`, `zz`, `xy`, `yz`, `xz`, `principal_1`, `principal_2`, `principal_3` |
| `-o/--output` | show interactively | Output PNG (headless if set) |
| `--clip` | none | Clipping plane normal `x,y,z` |
| `--clip-origin` | mesh centre | Clipping plane origin `x,y,z` |
| `--threshold` | none | Stress threshold for filtering |
| `--below` | off | With `--threshold`, keep values below instead of above |
| `--iso` | none | Comma-separated iso-surface levels, e.g. `150,200,250` |
| `--deformed` | none | Warp by the `displacement` array with this scale factor |
| `--cmap` | `stress` | Colormap (semantic name like `stress`, or any matplotlib name) |
| `--annotate` | off | Add critical-point annotations |

## `goldak`

Render a Goldak double-ellipsoid heat source as a 3-D iso-surface.

```bash
feaweld goldak -P 4000 --speed 5 -t 2 -o goldak.png
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-P/--power` | `4000.0` | Net power (W) |
| `--a-f` | `5.0` | Front semi-axis (mm) |
| `--a-r` | `10.0` | Rear semi-axis (mm) |
| `--b` | `4.0` | Width semi-axis (mm) |
| `--c` | `3.0` | Depth semi-axis (mm) |
| `--speed` | `5.0` | Travel speed (mm/s) |
| `-t/--time` | `2.0` | Snapshot time (s) for the render |
| `--iso` | `0.1` | Iso-surface level as a fraction of peak q |
| `-o/--output` | show interactively | Save screenshot PNG |

## `dashboard`

Run a case and render the multi-panel engineering dashboard.

```bash
feaweld dashboard case.yaml -o dashboard.png
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `-o/--output` | show interactively | Save the dashboard as a PNG |

## `animate`

Animate Palmgren-Miner damage accumulation across load blocks. Uses real rainflow
blocks if the case exposes them, otherwise synthesizes a sequence.

```bash
feaweld animate case.yaml -o damage.gif --fps 12
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `-o/--output` | `damage.gif` | Output animation (`.gif` or `.mp4`) |
| `--fps` | `10` | Frames per second |
| `--blocks` | `10` | Synthetic load blocks if the case has none |

## `materials`

List the bundled materials available for `material.base_metal` / `weld_metal`.

```bash
feaweld materials
```

No options.

## `study run`

Run a parametric study from a YAML definition and produce a comparison report.

```bash
feaweld study run study.yaml -j 4
```

| Option | Default | Meaning |
|--------|---------|---------|
| `study_file` (arg) | required | Study-definition YAML |
| `-j/--max-workers` | `4` | Parallel worker count |
| `-o/--output` | `results` | Output directory for the comparison report |
| `--report/--no-report` | `--report` | Generate the HTML comparison report |

## `study compare`

Run several standalone case files and compare them in a single report.

```bash
feaweld study compare case_a.yaml case_b.yaml --baseline case_a
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_files` (args) | ≥ 2 required | Case YAML files to compare |
| `-b/--baseline` | none | Baseline case name for delta computation |
| `-o/--output` | `results` | Output directory |
| `-j/--max-workers` | `4` | Parallel workers |

## `reliability mc`

Monte Carlo analysis of the safety factor for a case.

```bash
feaweld reliability mc case.yaml -n 5000 --seed 42 --sobol
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `-n/--samples` | case value | Number of MC samples |
| `--seed` | case value | Random seed |
| `--sobol` | off | Also compute Sobol sensitivity indices |
| `--save-samples` | none | Save `samples`/`results` arrays to a `.npz` |

## `reliability form`

FORM reliability index (β) against yield for a case.

```bash
feaweld reliability form case.yaml
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |

## `reliability sobol`

Sobol sensitivity indices of the safety factor.

```bash
feaweld reliability sobol case.yaml --n-base 512 --seed 42
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `--n-base` | `256` | Base sample size (evaluations = `n_base·(2k+2)`) |
| `--seed` | none | Random seed |

## `ml train`

Train a fatigue-life predictor on a CSV of features + `log10(N)` target.

```bash
feaweld ml train data.csv -m random_forest --target log_life -o model.joblib
```

| Option | Default | Meaning |
|--------|---------|---------|
| `data_csv` (arg) | required | Headed CSV of features + target |
| `-m/--model` | `random_forest` | `random_forest`, `xgboost`, `ensemble` |
| `--target` | `log_life` | Target column name (log₁₀ N) |
| `-o/--output` | `fatigue_model.joblib` | Saved model path |

## `ml predict`

Predict fatigue life with a trained model.

```bash
feaweld ml predict model.joblib -s stress_range=150 -s scf=1.7
```

| Option | Default | Meaning |
|--------|---------|---------|
| `model_file` (arg) | required | Trained model (`.joblib`) |
| `-s/--set name=value` | — | One feature assignment; repeatable |
| `-i/--input` | none | CSV of feature rows to predict instead |

## `ml transfer`

Fine-tune a trained model on plant-specific data (transfer learning).

```bash
feaweld ml transfer model.joblib plant.csv -o model_tuned.joblib
```

| Option | Default | Meaning |
|--------|---------|---------|
| `model_file` (arg) | required | Base model (`.joblib`) |
| `data_csv` (arg) | required | Plant-specific CSV |
| `--target` | `log_life` | Target column name |
| `-o/--output` | `fatigue_model_tuned.joblib` | Saved fine-tuned model |

## `multiscale`

Weld-zone property estimation (CCT → phases → properties).

```bash
feaweld multiscale --grade A36 --cooling-rate 30
```

| Option | Default | Meaning |
|--------|---------|---------|
| `-g/--grade` | `A36` | Steel grade (CCT database key) |
| `-r/--cooling-rate` | `30.0` | Cooling rate at 700 °C (°C/s) |
| `--base-yield` | `250.0` | Base-metal yield strength (MPa) |
| `--base-uts` | `400.0` | Base-metal ultimate strength (MPa) |

## `convergence`

Mesh convergence study (Richardson extrapolation + GCI) on the peak von Mises
stress.

```bash
feaweld convergence case.yaml --levels 4 --ratio 2.0
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `--levels` | `3` | Number of refinement levels (minimum 3) |
| `--ratio` | `2.0` | Mesh-size ratio between levels |

## `submodel`

Global-local submodel analysis around a point of interest.

```bash
feaweld submodel case.yaml --center "0,20,0" --radius 5 --refine 4
```

| Option | Default | Meaning |
|--------|---------|---------|
| `case_file` (arg) | required | Analysis-case YAML |
| `--center` | required | Submodel centre `x,y,z` (mm) |
| `-r/--radius` | required | Extraction radius (mm) |
| `--refine` | `4` | Mesh refinement factor |
| `--backend` | `auto` | `auto`, `fenics`, `calculix` |

## `twin dashboard`

Serve the live digital-twin web dashboard.

```bash
feaweld twin dashboard --demo
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--host` | `localhost` | WebSocket host |
| `--port` | `8765` | WebSocket port |
| `--http-port` | `8766` | HTTP port for the web UI |
| `--demo` | off | Feed synthetic sensor data |
| `--open/--no-open` | `--open` | Open the dashboard in a browser |

## `twin update`

Bayesian update of model parameters from observed data (emcee MCMC).

```bash
feaweld twin update --priors priors.yaml --data obs.csv --walkers 16 --steps 200
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--priors` | required | YAML list of prior specs |
| `--data` | required | CSV of observations (`value`, optional `noise_std`) |
| `--walkers` | `16` | MCMC walkers |
| `--steps` | `200` | MCMC steps |
| `--burnin` | `50` | Burn-in steps discarded |
