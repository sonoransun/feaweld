# Examples

Runnable examples for feaweld: two declarative YAML cases driven from the CLI,
and eight self-contained Python scripts that each print a summary to stdout.

## YAML cases

Driven by the `feaweld` CLI — no Python needed.

| File | Run with |
|------|----------|
| `fillet_t_joint.yaml` | `feaweld run examples/fillet_t_joint.yaml` |
| `leg_sweep_study.yaml` | `feaweld study run examples/leg_sweep_study.yaml -j 4` |

- **`fillet_t_joint.yaml`** — a single fillet T-joint under axial tension: geometry
  → mesh → linear-elastic solve → hot-spot + Dong + Blodgett post-processing →
  IIW FAT90 fatigue → HTML report. This is the recommended starting point and
  mirrors the [quickstart](../docs/quickstart.md).
- **`leg_sweep_study.yaml`** — a parametric study that sweeps the fillet weld leg
  size (6/8/10/12 mm) across the same case and emits an automated comparison
  report. The `base_case` is defined inline (a full `AnalysisCase`); the study
  runner does not accept a path to a separate case file.

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

- **`thermal_goldak.py`** — Evaluates a Goldak double-ellipsoid heat source over a
  grid and reports the peak power density and its location.
- **`creep_norton_bailey.py`** — Norton-Bailey creep relaxation during PWHT:
  evolves the residual stress from the post-welding state through a held hold
  temperature and reports the relaxed stress field.

### Data-driven

- **`ml_fatigue_predictor.py`** — Trains a Random Forest / XGBoost fatigue-life
  predictor on physics-informed features and predicts life on a held-out case.
- **`digital_twin_update.py`** — Bayesian (MCMC) updating of a fatigue model from
  a synthetic sensor stream: prior → posterior updates on damage parameters.

## Prerequisites

Everything below runs on the core install (`pip install -e .`) unless a `pip`
extra is listed. FEA solves need a backend — install either `fenics` **or**
`calculix`.

| Item | Required extra |
|------|----------------|
| `fillet_t_joint.yaml` | `viz` + (`fenics` or `calculix`) |
| `leg_sweep_study.yaml` | `viz` + (`fenics` or `calculix`) |
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
