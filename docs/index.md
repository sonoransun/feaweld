# feaweld

Finite element analysis toolkit for weld joint stress, fatigue life, and structural integrity assessment.

![pipeline overview](images/pipeline_overview.svg)

## Overview

feaweld is a Python package for engineers who need to evaluate welded connections in metal structures. It covers the full analysis workflow from parametric joint geometry and mesh generation through FEA solving, post-processing, fatigue assessment, and visualization — producing HTML reports with embedded engineering figures.

The package implements methods from major welding and pressure vessel codes (ASME VIII, IIW, DNV-RP-C203, AWS D1.1, BS 7910, API 579) and ships with a reference database of 49 materials, 100 IIW weld detail categories, S-N curves for six standards (IIW, DNV, ASME, Eurocode 3, BS 7608, AWS D1.1), CCT diagrams for 20 steel grades, and parametric SCF data for 10 weld geometries.

Analysis cases are defined in YAML and can be run individually or as concurrent parametric studies with automated comparison reporting.

Beyond a single static check, feaweld includes:

- **Spectrum fatigue** — rainflow counting, Palmgren-Miner damage, mean-stress correction, knockdowns, and residual-stress integration from a `fatigue:` YAML block.
- **3D analysis** — extruded solid joints (`geometry.dimension: 3`) with hot-spot stations along each weld toe line.
- **Probabilistic fatigue** — Monte Carlo with Latin Hypercube Sampling, Sobol sensitivity, FORM reliability.
- **Machine learning** — Random Forest / XGBoost fatigue predictors with transfer learning.
- **Multi-scale modeling** — Hall-Petch, dislocation density, CCT interpolation.
- **Digital twin** — MQTT/OPC-UA sensor ingestion with Bayesian model updating.

## Where to start

<div class="grid cards" markdown>

-   **[Installation](installation.md)**

    Set up a virtual environment and install the right optional extras.

-   **[Quickstart](quickstart.md)**

    Run your first analysis from the CLI in under a minute.

-   **[Architecture](architecture.md)**

    The pipeline, design patterns, and module map — with diagrams.

-   **[Tutorials](tutorials/01_yaml_analysis.md)**

    Step-by-step walkthroughs of YAML cases, parametric studies, and custom post-processors.

-   **[CLI reference](reference/cli.md)**

    Every command and option, with an example each.

-   **[API reference](api/core.md)**

    Auto-generated reference for all 14 sub-packages.

</div>

## Guides

Task-oriented guides for each capability:

- **[Solvers](guides/solvers.md)** — the six solver types, backend auto-detection, and the elastoplastic caveat.
- **[Loads & boundary conditions](guides/loads_and_bcs.md)** — how each load field becomes a nodal BC.
- **[PWHT](guides/pwht.md)** — post-weld heat treatment stress relaxation.
- **[Fatigue assessment](guides/fatigue_assessment.md)** — spectrum loading, rainflow + Miner, mean stress, residual stress, and the six S-N standards.
- **[3D analysis](guides/analysis_3d.md)** — extruded solid models, 3D meshing, and hot-spot stations along the weld toe.
- **[Probabilistic & reliability](guides/probabilistic.md)** — Monte Carlo, Sobol, and FORM.
- **[ML fatigue prediction](guides/ml.md)** — Random Forest / XGBoost predictors and transfer learning.
- **[Multiscale modeling](guides/multiscale.md)** — CCT phases, zone properties, and Hall-Petch.
- **[Convergence & submodeling](guides/convergence.md)** — GCI, the singularity check, and submodels.
- **[Digital twin](guides/digital_twin.md)** — live dashboard and Bayesian model updating.
- **[Visualization](guides/visualization.md)** — 2-D / 3-D plots and automatic report figures.

## Standards coverage

| Standard | Implementation |
|----------|----------------|
| ASME VIII Division 2 | Stress categorization, allowable checks, design fatigue curves |
| IIW-2006-09 / IIW-2008 | 14 FAT classes, 100 weld detail categories, hot-spot stress, effective notch stress |
| DNV-RP-C203 | 14 S-N curve categories (in-air; seawater via environment knockdown) |
| Eurocode 3 (EN 1993-1-9) | S-N detail categories (14 categories, two-slope) |
| BS 7608 | S-N classes B–W1, mean and design curves |
| ASME 2007 Annex 5-C | Battelle/Dong mesh-insensitive structural stress, master S-N curve |
| ASTM E1049 | Rainflow cycle counting, spectrum fatigue via Palmgren-Miner |
| BS 7910 / API 579 | Residual stress through-thickness profiles (Level 1 and 2) |
| AWS D1.1 | S-N categories A–E', weld joint efficiency factors, filler metal matching |
| Lazzarin (2001) | Strain energy density method with control volume |

## Project metrics

- 87 source modules, ~26,000 lines of code
- 1,089 tests across 38 test modules
- 49 material databases with temperature-dependent properties
- 6 JSON reference datasets
- 5 joint geometry types, 2 solver backends, 8 post-processing methods
