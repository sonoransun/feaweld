# feaweld

Finite element analysis toolkit for weld joint stress, fatigue life, and structural integrity assessment.

## Overview

feaweld is a Python package for engineers who need to evaluate welded connections in metal structures. It covers the full analysis workflow from parametric joint geometry and mesh generation through FEA solving, post-processing, fatigue assessment, and visualization — producing HTML reports with embedded engineering figures.

The package implements methods from major welding and pressure vessel codes (ASME VIII, IIW, DNV-RP-C203, AWS D1.1, BS 7910, API 579) and ships with a reference database of 49 materials, 80 IIW weld detail categories, S-N curves for three standards, CCT diagrams for 20 steel grades, and parametric SCF data for 10 weld geometries. Analysis cases are defined in YAML and can be run individually or as concurrent parametric studies with automated comparison reporting.

Beyond conventional deterministic methods, feaweld includes probabilistic fatigue assessment (Monte Carlo with Latin Hypercube Sampling), machine-learning fatigue predictors (Random Forest / XGBoost with transfer learning), multi-scale material modeling (Hall-Petch, dislocation density, phase transformation), and a digital twin framework for real-time sensor integration and Bayesian model updating.

## Key Capabilities

**Structural Analysis**
- Dual FEA solver backend: FEniCSx (nonlinear thermomechanical) and CalculiX (standard linear/thermal)
- Five parametric joint types: fillet T-joint, butt weld, lap joint, corner joint, cruciform
- Goldak double-ellipsoid heat source for welding simulation with element birth-death
- J2 elastoplastic constitutive model with radial return mapping
- Norton-Bailey creep for post-weld heat treatment (PWHT) stress relaxation

**Fatigue Assessment**
- Eight post-processing methods: nominal (ASME VIII), hot-spot (IIW Type A/B), Battelle/Dong mesh-insensitive structural stress, effective notch stress (FAT225), strain energy density (Lazzarin), through-thickness linearization, Blodgett hand calculations
- S-N curves: 14 IIW FAT classes, 17 DNV-RP-C203 categories, ASME VIII ferritic/austenitic
- Rainflow cycle counting (ASTM E1049), Palmgren-Miner cumulative damage, Goodman/Gerber mean stress correction
- Fatigue knockdown factors for surface finish, size, environment

**Visualization**
- 3D stress contours, deformed shapes, clipping planes, threshold filtering, iso-surfaces, weld region highlighting (PyVista)
- 2D engineering plots: through-thickness linearization, hot-spot extrapolation, S-N curves, Dong decomposition, ASME stress check bars, weld group geometry (Matplotlib)
- Engineering dashboards combining multiple panels with critical point annotations and safety factor overlay
- HTML reports with embedded base64 figures

**Parametric Studies**
- Concurrent multi-case execution via ProcessPoolExecutor
- Parameter sweeps (grid or one-at-a-time) over loads, materials, mesh refinement
- Automated comparison reports with metric tables, delta computation, sensitivity plots

**Reference Data**
- 49 materials with temperature-dependent properties (carbon steel, stainless, high-strength, pipeline, aluminum, filler metals)
- Lazy-loading data cache with LRU eviction for on-demand access
- SCF parametric coefficients for 10 weld geometries
- 80 IIW weld detail-to-FAT class mappings
- CCT diagrams for 20 steel grades
- Residual stress profiles from BS 7910, API 579, R6, FITNET, DNV
- 82 AWS A5 filler metal classifications with base metal matching
- 25 weld joint efficiency factors (ASME, AWS, EN)

## Real-World Applications

### Pressure Vessel Weld Qualification (ASME VIII)

Evaluate longitudinal and circumferential seam welds in vessels per ASME Section VIII Division 2. feaweld categorizes stress into membrane, bending, and peak components against code allowables (Pm, Pm+Pb, PL+Pb+Q), applies joint efficiency factors based on NDE method, and assesses fatigue life using ASME design curves. PWHT simulation predicts residual stress relaxation for code-compliant heat treatment schedules.

### Offshore Structural Fatigue (DNV-RP-C203)

Assess tubular joints, stiffener connections, and bracket details on jacket structures and floating platforms using DNV S-N curves with seawater environment corrections. The hot-spot stress method extracts structural stress at weld toes for fatigue life prediction under variable-amplitude wave loading (rainflow counting + Miner's rule). Parametric studies sweep plate thickness and weld detail category to identify design-governing connections.

### Pipeline Girth Weld Integrity (API 5L)

Analyze circumferential girth welds in API 5L line pipe (X42 through X80 grades) for installation and operational loading. The material database includes pipeline steels with temperature-dependent properties. CCT diagrams predict HAZ microstructure from welding thermal cycles, and residual stress profiles from API 579 inform fitness-for-service assessments at girth weld repair locations.

### Structural Steel Fabrication (AWS D1.1 / Blodgett)

Design welded connections in building and bridge structures using Blodgett's weld-as-a-line method for preliminary sizing (9 standard weld group shapes with LRFD/ASD capacity checks per AISC). Verify designs with FEA using IIW FAT class fatigue curves mapped from 80 weld detail categories. The filler metal database recommends matching AWS A5 consumables for each base metal grade.

### Post-Weld Heat Treatment Optimization

Compare as-welded vs. PWHT residual stress states and their impact on fatigue life. The coupled thermomechanical solver simulates the full welding thermal cycle (Goldak heat source), then the creep solver applies Norton-Bailey stress relaxation through a defined PWHT schedule (heating rate, holding temperature/time, cooling rate). Parametric studies vary PWHT parameters to find the minimum treatment that meets fatigue life targets.

### Manufacturing Quality Control and Digital Twin

Ingest real-time sensor data from welding cells (thermocouples, arc monitors, strain gauges) via MQTT or OPC-UA. The Bayesian updater refines material property estimates from observed temperature fields using MCMC sampling, and propagates the posterior through the fatigue model for remaining-life prediction with confidence bounds. A WebSocket dashboard provides real-time alerts for out-of-specification conditions.

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[viz]"    # core + matplotlib + pyvista

# Run an analysis from YAML
feaweld run examples/fillet_t_joint.yaml

# Blodgett hand calculation
feaweld blodgett -g box --d 100 --b 50 -t 5 -P 50000

# List available materials
feaweld materials

# Run a parametric study
feaweld study run study.yaml -j 4
```

**Programmatic usage:**

```python
from feaweld.pipeline.workflow import AnalysisCase, run_analysis
from feaweld.pipeline.report import generate_report

case = AnalysisCase(name="my_joint")
result = run_analysis(case)
generate_report(result, "output/")
```

## Architecture

The package follows a linear pipeline with modular, independently testable stages:

```
YAML case definition
  -> Geometry (Gmsh parametric joints)
    -> Mesh (refinement at weld toes)
      -> Solve (FEniCSx or CalculiX)
        -> Post-process (8 stress methods)
          -> Fatigue (S-N curves + Miner's rule)
            -> Visualize (2D/3D) + Report (HTML)
```

Parametric studies run multiple pipelines concurrently and produce comparison reports with metric tables and sensitivity plots.

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

## Project Metrics

- 64 source modules, ~17,700 lines of code
- 332 passing tests across 18 test modules
- 49 material databases (7 categories) with temperature-dependent properties
- 6 JSON reference datasets (SCF, CCT, S-N details, residual stress, filler metals, weld efficiency)
- 5 joint geometry types, 2 solver backends, 8 post-processing methods
