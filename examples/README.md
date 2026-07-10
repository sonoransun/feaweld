# Examples

Runnable examples for feaweld: four declarative YAML cases driven from the CLI,
and eight self-contained Python scripts that each print a summary to stdout.

## YAML cases

Driven by the `feaweld` CLI — no Python needed.

| File | Run with |
|------|----------|
| `fillet_t_joint.yaml` | `feaweld run examples/fillet_t_joint.yaml` |
| `fillet_t_joint_3d.yaml` | `feaweld run examples/fillet_t_joint_3d.yaml` |
| `leg_sweep_study.yaml` | `feaweld study run examples/leg_sweep_study.yaml -j 4` |
| `spectrum_fatigue.yaml` | `feaweld run examples/spectrum_fatigue.yaml` |

- **`fillet_t_joint.yaml`** — a single fillet T-joint under axial tension: geometry
  → mesh → linear-elastic solve → hot-spot + Dong + Blodgett post-processing →
  IIW FAT90 fatigue → HTML report. This is the recommended starting point and
  mirrors the [quickstart](../docs/quickstart.md).
- **`fillet_t_joint_3d.yaml`** — the same joint extruded into a solid via
  `geometry.dimension: 3` (`length` = 40 mm weld length): coarse TET4 mesh with
  toe-line refinement, per-station hot spot along each weld-toe line (max over
  stations, `critical_line` reported), plus an `R = 0.1` `fatigue:` block
  showing 3D and spectrum fatigue composing. Walked through in the
  [3D analysis guide](../docs/guides/analysis_3d.md). Meshing works on the core
  install (gmsh is a core dependency); the solve needs a backend — FEniCS
  requires `element_order: 1` (TET4) in 3D.
- **`leg_sweep_study.yaml`** — a parametric study that sweeps the fillet weld leg
  size (6/8/10/12 mm) across the same case and emits an automated comparison
  report. The `base_case` is defined inline (a full `AnalysisCase`); the study
  runner does not accept a path to a separate case file.
- **`spectrum_fatigue.yaml`** — a butt weld (groove angle / root gap parameters)
  under the variable-amplitude stress history in `load_history.csv`: rainflow
  counting, Palmgren-Miner damage on the EC3 category-90 curve, Goodman
  mean-stress correction, and a PWHT-relaxed residual stress profile. Walked
  through in the
  [fatigue assessment guide](../docs/guides/fatigue_assessment.md).
- **`load_history.csv`** — the 200-point stress history (MPa) the spectrum case
  consumes. Also usable standalone, with no FEA backend:
  `feaweld fatigue --history examples/load_history.csv -c EC3_90`.

## Python scripts

Run any of them with:

```bash
python examples/<name>.py
```

### Fatigue and structural

- **`fillet_t_joint.py`** — Full pipeline on a fillet T-joint under axial tension:
  geometry, mesh, linear-elastic solve, three post-processing methods (hot-spot,
  Dong, Blodgett), IIW FAT90 fatigue, and an HTML report. The programmatic twin
  of `fillet_t_joint.yaml`.
- **`butt_weld_fatigue.py`** — Butt weld under variable-amplitude loading:
  rainflow cycle counting (ASTM E1049), Palmgren-Miner cumulative damage, and
  Goodman mean-stress correction.
- **`pwht_comparison.py`** — Post-weld heat treatment comparison: as-welded vs.
  PWHT residual stress state and its impact on fatigue life.
- **`probabilistic_life.py`** — Monte Carlo fatigue assessment with Latin
  Hypercube Sampling; treats weld leg size, material strength, and load as random
  variables and produces a life distribution with a reliability-index.

### Thermal and creep

- **`thermal_goldak.py`** — Samples the volumetric heat input of a traveling
  Goldak double-ellipsoid heat source on a regular grid and reports the peak
  power density and its location at a snapshot time (no transient solve).
- **`creep_norton_bailey.py`** — Norton-Bailey creep relaxation during PWHT:
  evolves the residual stress from the post-welding state through a held hold
  temperature and reports the relaxed stress field.

### Data-driven

- **`ml_fatigue_predictor.py`** — Trains a Random Forest fatigue-life predictor
  on synthetic FAT90-class data generated in-script with realistic scatter,
  reports the cross-validation RMSE / R², then predicts fatigue life for a new
  case and prints the top feature importances.
- **`digital_twin_update.py`** — Bayesian (MCMC) updating of a fatigue model from
  a synthetic sensor stream: prior → posterior updates on damage parameters.

## Prerequisites

Everything below runs on the core install (`pip install -e .`) unless a `pip`
extra is listed. FEA solves need a backend — install either `fenics` **or**
`calculix`.

| Item | Required extra |
|------|----------------|
| `fillet_t_joint.yaml` | `viz` + (`fenics` or `calculix`) |
| `fillet_t_joint_3d.yaml` | `viz` + (`fenics` or `calculix`); meshing alone runs on core |
| `leg_sweep_study.yaml` | `viz` + (`fenics` or `calculix`) |
| `spectrum_fatigue.yaml` | `viz` + (`fenics` or `calculix`) |
| `load_history.csv` via `feaweld fatigue` | none (core) |
| `fillet_t_joint.py` | `viz` + (`fenics` or `calculix`) |
| `butt_weld_fatigue.py` | none (core) |
| `pwht_comparison.py` | none (core) |
| `probabilistic_life.py` | none (core) |
| `thermal_goldak.py` | none (core) |
| `creep_norton_bailey.py` | none (core) |
| `ml_fatigue_predictor.py` | `ml` |
| `digital_twin_update.py` | `digital-twin` |

Install extras with, e.g., `pip install -e ".[viz,calculix]"` or `pip install -e ".[ml]"`.

## Expected outputs

The report-producing runs write a self-contained HTML report (figures embedded
as base64) to `results/<name>/report.html` — e.g. `fillet_t_joint.yaml` writes
`results/fillet_t_joint/report.html`, and the study writes a comparison report
under `results/`. The remaining scripts print their summaries to stdout. Delete
`results/<name>/` between runs to avoid stale reports.
