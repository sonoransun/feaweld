# feaweld

Finite element analysis toolkit for weld joint stress, fatigue life, and structural integrity assessment.

<p align="center">
  <img src="docs/images/pipeline_overview.svg" alt="feaweld analysis pipeline" width="100%">
</p>

## Overview

feaweld is a Python package for engineers who need to evaluate welded connections in metal structures. It covers the full analysis workflow from parametric joint geometry and mesh generation through FEA solving, post-processing, fatigue assessment, and visualization — producing HTML reports with embedded engineering figures.

The package implements methods from major welding and pressure vessel codes (ASME VIII, IIW, DNV-RP-C203, AWS D1.1, BS 7910, API 579) and ships with a reference database of 49 materials, 80 IIW weld detail categories, S-N curves for three standards, CCT diagrams for 20 steel grades, and parametric SCF data for 10 weld geometries. Analysis cases are defined in YAML and can be run individually or as concurrent parametric studies with automated comparison reporting.

Beyond conventional deterministic methods, feaweld includes probabilistic fatigue assessment (Monte Carlo with Latin Hypercube Sampling), machine-learning fatigue predictors (Random Forest / XGBoost with transfer learning), multi-scale material modeling (Hall-Petch, dislocation density, phase transformation), and a digital twin framework for real-time sensor integration and Bayesian model updating.

## Visual Overview

<table>
<tr>
<td width="50%"><img src="docs/images/example_sn_curve.svg" alt="S-N Curve"></td>
<td width="50%"><img src="docs/images/example_through_thickness.svg" alt="Through-Thickness Linearization"></td>
</tr>
<tr>
<td><em>S-N fatigue curve with operating point, regime bands, and CAFL</em></td>
<td><em>Through-thickness stress linearization per ASME VIII</em></td>
</tr>
<tr>
<td width="50%"><img src="docs/images/example_hotspot.svg" alt="Hot-Spot Extrapolation"></td>
<td width="50%"><img src="docs/images/example_dong.svg" alt="Dong Decomposition"></td>
</tr>
<tr>
<td><em>IIW hot-spot stress extrapolation with weld toe schematic</em></td>
<td><em>Dong structural stress decomposition (membrane + bending)</em></td>
</tr>
</table>

## How It Works

### Joint Types

feaweld supports five parametric weld joint geometries, each defined by plate thickness, weld leg size, and connection dimensions:

<p align="center">
  <img src="docs/images/joint_types.svg" alt="Joint types" width="90%">
</p>

### Hot-Spot Stress Method

The IIW hot-spot method extracts structural stress at the weld toe by extrapolating from reference points away from the stress concentration zone:

<p align="center">
  <img src="docs/images/hotspot_concept.svg" alt="Hot-spot stress concept" width="80%">
</p>

### Through-Thickness Linearization

ASME VIII Division 2 decomposes the actual stress distribution into membrane, bending, and peak components for comparison against code allowables:

<p align="center">
  <img src="docs/images/linearization_concept.svg" alt="Linearization concept" width="85%">
</p>

### S-N Fatigue Assessment

Fatigue life is predicted using S-N curves from IIW, DNV, or ASME standards with proper handling of the knee point (CAFL) and variable-amplitude loading via Miner's rule:

<p align="center">
  <img src="docs/images/sn_concept.svg" alt="S-N curve fundamentals" width="75%">
</p>

### Dong Mesh-Insensitive Structural Stress

The Battelle/Dong method uses nodal force equilibrium at the weld toe to compute structural stress, eliminating mesh sensitivity:

<p align="center">
  <img src="docs/images/dong_concept.svg" alt="Dong method concept" width="80%">
</p>

### Goldak Heat Source

Welding simulation uses the Goldak double-ellipsoid heat source model for accurate thermal cycle prediction:

<p align="center">
  <img src="docs/images/goldak_concept.svg" alt="Goldak heat source" width="70%">
</p>

### Strain Energy Density (SED)

The Lazzarin SED method averages strain energy density over a control volume at the notch tip, providing a local damage parameter:

<p align="center">
  <img src="docs/images/sed_concept.svg" alt="SED control volume" width="60%">
</p>

### Pipeline Architecture

The analysis pipeline is modeled as a DAG. Independent stages execute concurrently, and each batch boundary is a checkpoint save point for crash recovery.

```mermaid
flowchart LR
    YAML["YAML Case"] --> MAT["Materials"] & DEF["Defects"]
    MAT --> GEO["Geometry"]
    GEO --> MESH["Mesh"]
    MESH --> SOLVE["Solver"]
    SOLVE --> HS["Hotspot"] & DG["Dong"] & LN["Linearization"] & SD["SED"]
    HS & DG & LN & SD --> FAT["Fatigue"]
    FAT --> PROB["Probabilistic"]
    PROB --> RPT["Report"]
```

## Key Capabilities

**Structural Analysis**
- Four FEA solver backends: FEniCSx (nonlinear thermomechanical), CalculiX (standard linear/thermal), JAX (differentiable), and Neural (surrogate)
- Five parametric joint types: fillet T-joint, butt weld, lap joint, corner joint, cruciform
- Goldak double-ellipsoid heat source for welding simulation with element birth-death
- J2 elastoplastic constitutive model with radial return mapping
- Norton-Bailey creep for post-weld heat treatment (PWHT) stress relaxation

**Fatigue Assessment**
- Eight post-processing methods: nominal (ASME VIII), hot-spot (IIW Type A/B), Battelle/Dong mesh-insensitive structural stress, effective notch stress (FAT225), strain energy density (Lazzarin), through-thickness linearization, Blodgett hand calculations
- Six multi-axial fatigue criteria: Findley, Dang Van, Sines, Crossland, Fatemi-Socie, McDiarmid
- S-N curves: 14 IIW FAT classes, 17 DNV-RP-C203 categories, ASME VIII ferritic/austenitic
- Rainflow cycle counting (ASTM E1049), Palmgren-Miner cumulative damage, Goodman/Gerber mean stress correction
- Fatigue knockdown factors for surface finish, size, environment

**Visualization**
- 3D stress contours, deformed shapes, clipping planes, threshold filtering, iso-surfaces, weld region highlighting (PyVista)
- 2D engineering plots: through-thickness linearization, hot-spot extrapolation, S-N curves, Dong decomposition, ASME stress check bars, weld group geometry (Matplotlib)
- Engineering dashboards combining multiple panels with critical point annotations and safety factor overlay
- HTML reports with embedded base64 figures

**Parametric Studies**
- Concurrent multi-case execution with spawn-safe `ProcessPoolExecutor`
- Parameter sweeps (grid or one-at-a-time) over loads, materials, mesh refinement
- Distributed scaling to Dask or Ray clusters for multi-node studies
- Automated comparison reports with metric tables, delta computation, sensitivity plots

**Pipeline & Orchestration**
- DAG-based pipeline execution with concurrent post-processing stage batches
- Pipeline hooks for pre/post stage callbacks, timing, and memory profiling
- Checkpoint/restart for crash recovery of long-running analyses
- SQLite-backed job queue with priority scheduling and worker loop
- Shared-memory IPC for zero-copy numpy array transfer between study workers
- Graceful SIGTERM/SIGINT shutdown with partial result serialization

**Deployment & Operations**
- Docker multi-stage builds and docker-compose stack (feaweld + MQTT broker + optional Grafana/Prometheus)
- systemd service unit for digital twin daemon with watchdog, cgroups limits, and security hardening
- Structured logging: console text, JSON lines (containers), systemd journal
- Resource monitoring via /proc integration and RLIMIT subprocess enforcement
- Memory-mapped arrays for models that exceed available RAM
- File watching (inotify) for hot-reload of case YAML files and material databases

**Developer Tooling**
- Pre-commit hooks (ruff linting + formatting, mypy type checking)
- GitHub Actions CI with test matrix (Python 3.10-3.12, optional dep profiles)
- Property-based tests (Hypothesis), benchmark suite (pytest-benchmark)
- Makefile for common tasks: `make test`, `make lint`, `make typecheck`, `make docs`
- mkdocs-based API documentation with mkdocstrings

**Reference Data**
- 49 materials with temperature-dependent properties (carbon steel, stainless, high-strength, pipeline, aluminum, filler metals)
- Lazy-loading data cache with LRU eviction for on-demand access
- SCF parametric coefficients for 10 weld geometries
- 80 IIW weld detail-to-FAT class mappings
- CCT diagrams for 20 steel grades
- Residual stress profiles from BS 7910, API 579, R6, FITNET, DNV
- 82 AWS A5 filler metal classifications with base metal matching
- 25 weld joint efficiency factors (ASME, AWS, EN)

## CLI Overview

```mermaid
mindmap
  root((feaweld))
    Analysis
      run
      validate
      profile
    Stress Methods
      blodgett
      j-integral
    Visualization
      visualize
      dashboard
    Mesh
      mesh generate
      mesh inspect
    Studies
      study run
      study compare
      convergence
      sensitivity
    Data
      materials
      groove-types
      defects list
    Export
      export
    Digital Twin
      twin start
      twin status
    Job Queue
      queue submit
      queue status
```

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[viz]"    # core + matplotlib + pyvista

# Validate a case file
feaweld validate examples/fillet_t_joint.yaml

# Run an analysis from YAML
feaweld run examples/fillet_t_joint.yaml

# Blodgett hand calculation
feaweld blodgett -g box --d 100 --b 50 -t 5 -P 50000

# Generate and inspect mesh without solving
feaweld mesh generate examples/fillet_t_joint.yaml -o mesh.vtk
feaweld mesh inspect mesh.vtk

# Run a parametric study
feaweld study run study.yaml -j 4

# Mesh convergence study
feaweld convergence examples/fillet_t_joint.yaml -n 4

# Single-parameter sensitivity sweep
feaweld sensitivity examples/fillet_t_joint.yaml --param load.axial_force --range 10000:50000:5

# Export results to CSV
feaweld export results/stress.vtk --format csv

# Profile per-stage timing
feaweld profile examples/fillet_t_joint.yaml

# Submit to job queue
feaweld queue submit examples/fillet_t_joint.yaml -p 1
feaweld queue status

# Start digital twin daemon
feaweld twin start --host mqtt.local
```

**Programmatic usage:**

```python
from feaweld.pipeline.workflow import AnalysisCase, run_analysis
from feaweld.pipeline.report import generate_report

case = AnalysisCase(name="my_joint")
result = run_analysis(case)
generate_report(result, "output/")
```

## Weld Group Shapes (Blodgett)

All nine standard weld group shapes for hand calculations per the Blodgett method:

<p align="center">
  <img src="docs/images/weld_groups_gallery.svg" alt="Weld group shapes" width="90%">
</p>

## ASME Stress Check

ASME VIII Division 2 stress categorization with gradient utilization display and limit equations:

<p align="center">
  <img src="docs/images/example_asme_check.svg" alt="ASME stress check" width="70%">
</p>

<!-- ADVANCED_CONCEPTS_START -->
## Advanced Concepts

feaweld extends well beyond the classical IIW / ASME / DNV workflow. The full
catalog of advanced concepts — differentiable FEA, phase-field fracture,
Bayesian surrogates, active learning, multi-pass welding, spline paths,
volumetric joints, defect populations, multi-axial fatigue, fracture mechanics
— lives in [docs/CONCEPTS.md](docs/CONCEPTS.md).

### Phase-field fracture (crack propagation)

<p align="center">
  <img src="docs/animations/phase_field_crack_propagation.gif" alt="Phase field crack propagation" width="80%">
</p>

### Multi-pass welding thermal cycle

<p align="center">
  <img src="docs/animations/multipass_thermal_cycle.gif" alt="Multipass thermal cycle" width="80%">
</p>

### Active learning over parametric studies

<p align="center">
  <img src="docs/animations/active_learning_convergence.gif" alt="Active learning convergence" width="80%">
</p>

### EnKF crack-length assimilation

<p align="center">
  <img src="docs/animations/enkf_crack_tracking.gif" alt="EnKF crack-length tracking" width="80%">
</p>

### Solver backend hierarchy

```mermaid
classDiagram
    class SolverBackend {
        <<abstract>>
        +solve_static(mesh, material, load_case, temperature)
        +solve_thermal_steady(mesh, material, load_case)
        +solve_thermal_transient(mesh, material, load_case, time_steps)
        +solve_coupled(mesh, material, mech_lc, thermal_lc, time_steps)
    }
    class FEniCSBackend
    class CalculiXBackend
    class JAXBackend
    class NeuralBackend
    SolverBackend <|-- FEniCSBackend
    SolverBackend <|-- CalculiXBackend
    SolverBackend <|-- JAXBackend
    SolverBackend <|-- NeuralBackend
    class JAXConstitutiveModel {
        <<protocol>>
        +stress(strain)
        +tangent(strain)
    }
    JAXBackend o-- JAXConstitutiveModel : uses
```

See [docs/CONCEPTS.md](docs/CONCEPTS.md) for the full index of 8 mermaid
architecture diagrams, 23 high-resolution concept images, and 9 animations
covering every advanced feature.
<!-- ADVANCED_CONCEPTS_END -->

## Standards Coverage

| Standard | Implementation |
|----------|---------------|
| ASME VIII Division 2 | Stress categorization, allowable checks, design fatigue curves |
| IIW-2006-09 / IIW-2008 | 14 FAT classes, 80 weld detail categories, hot-spot stress, effective notch stress |
| DNV-RP-C203 | 17 S-N curve categories (in-air and seawater) |
| ASME 2007 Annex 5-C | Battelle/Dong mesh-insensitive structural stress, master S-N curve |
| ASTM E1049 | Rainflow cycle counting |
| BS 7910 / API 579 | Residual stress through-thickness profiles (Level 1 and 2) |
| AWS D1.1 | Weld joint efficiency factors, filler metal matching |
| Lazzarin (2001) | Strain energy density method with control volume |

## Deployment

feaweld supports containerised deployment with Docker and production monitoring via systemd. See the [deployment guide](docs/deployment.md) for full details.

```mermaid
flowchart TB
    subgraph compose["docker-compose"]
        FW["feaweld"] -->|MQTT| MQ["Mosquitto :1883"]
        FW --> VOL[("results/")]
    end
    subgraph systemd
        SVC["feaweld-twin.service"] --> TWIN["daemon"]
        SVC --> JRNL["journald"]
    end
    MQ --> TWIN
    SENSOR["Sensors"] --> MQ
    TWIN --> WS["WebSocket Dashboard"]

    subgraph monitoring["optional"]
        PROM["Prometheus"] --> GRAF["Grafana :3000"]
    end
    TWIN --> PROM
```

## Project Metrics

- 110+ source modules, ~30,000 lines of code
- 506 passing tests across 60 test modules
- 49 material databases (7 categories) with temperature-dependent properties
- 6 JSON reference datasets (SCF, CCT, S-N details, residual stress, filler metals, weld efficiency)
- 5 joint geometry types, 4 solver backends, 8+ post-processing methods, 6 multi-axial criteria
