# Installation

feaweld requires Python 3.10 or newer. The core package is lightweight; heavy optional dependencies (FEA solvers, visualization, ML, digital-twin stack) are pulled in via extras so you install only what you need.

## Basic install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The core install gives you: geometry builders, mesh generation (via Gmsh), post-processing, fatigue assessment (S-N curves, rainflow, Miner), parametric studies, and HTML report generation.

## Optional extras

Choose the extras that match your workflow.

### Visualization — `[viz]`

```bash
pip install -e ".[viz]"
```

Adds matplotlib (2D engineering plots, dashboards, animations), PyVista/VTK (3D stress fields, clipping, iso-surfaces), Plotly (interactive HTML report figures), and `imageio-ffmpeg` (MP4/GIF animation export with a bundled `ffmpeg` binary).

### FEA solvers

Pick one or both backends. `get_backend("auto")` prefers FEniCSx and falls back to CalculiX.

```bash
pip install -e ".[fenics]"     # FEniCSx — nonlinear, thermomechanical, coupled
pip install -e ".[calculix]"   # CalculiX — standard linear/thermal via pygccx
```

### Machine learning — `[ml]`

```bash
pip install -e ".[ml]"
```

Adds scikit-learn, XGBoost, and joblib for the Random Forest / XGBoost fatigue predictors with transfer learning.

### Digital twin — `[digital-twin]`

```bash
pip install -e ".[digital-twin]"
```

Adds `emcee` (MCMC for Bayesian updating), `paho-mqtt` (MQTT ingest), `asyncua` (OPC-UA ingest), and `websockets` (real-time dashboard).

### Documentation toolchain — `[docs]`

```bash
pip install -e ".[docs]"
```

Adds MkDocs + Material theme + `mkdocstrings` so you can build this site locally:

```bash
mkdocs serve                 # live-reload preview at http://127.0.0.1:8000
mkdocs build --strict        # production build, fail on warnings
```

### Development — `[dev]`

```bash
pip install -e ".[dev]"
```

Adds `pytest` and `pytest-cov`.

### Everything — `[all]`

```bash
pip install -e ".[all]"
```

Aggregates `fenics`, `calculix`, `viz`, `ml`, and `digital-twin`. Large install; use only if you need the full feature set.

## System dependencies

- **Gmsh** — required by the mesh generator. On macOS/Linux `pip install gmsh` pulls prebuilt wheels. Headless machines need a system `libGL` or the `-nopopup` option.
- **FEniCSx** — requires system MPI and PETSc. On macOS, prefer `conda install -c conda-forge fenics-dolfinx` then `pip install -e .` into the same env.
- **CalculiX** — `pygccx` wraps the `ccx` binary; install `ccx` via Homebrew (`brew install calculix-ccx`) or from http://www.calculix.de.
- **ffmpeg** — bundled by `imageio-ffmpeg`; no action needed. If you prefer the system binary, install via Homebrew (`brew install ffmpeg`).

## Verify the install

```bash
feaweld --version
feaweld materials              # prints the bundled material database
pytest tests/ -x               # 332+ tests; some require optional extras
```

Tests requiring optional backends are marked:

```bash
pytest tests/ -m "not requires_fenics and not requires_calculix"
```
