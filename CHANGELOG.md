# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Structured logging with systemd journal support
- DAG-based pipeline execution with concurrent post-processing
- Pipeline hooks for pre/post stage callbacks
- Checkpoint/restart for crash recovery
- Distributed execution support (Dask/Ray)
- SQLite-backed job queue
- Resource monitoring (memory pressure, /proc integration)
- Memory-mapped arrays for large result fields
- File watching for hot-reload of case files
- systemd service unit for digital twin daemon
- New CLI commands: validate, mesh generate/inspect, convergence, sensitivity, export, profile
- Digital twin daemon with MQTT and WebSocket support
- GitHub Actions CI with test matrix
- Docker and docker-compose configurations
- Pre-commit hooks (ruff, mypy)
- Makefile for common development tasks
- mkdocs API documentation
- Property-based tests (Hypothesis)
- Benchmark suite (pytest-benchmark)

### Changed
- ProcessPoolExecutor uses spawn context for MPI safety
- CalculiX backend respects FEAWELD_TMPDIR environment variable
- Pipeline logging at each stage for observability

## [0.1.0] - Initial Release

### Added
- Core analysis pipeline (geometry, mesh, solve, postprocess, fatigue)
- FEniCSx and CalculiX solver backends
- JAX differentiable solver backend
- Neural surrogate solver backend
- 8 post-processing methods (hotspot, Dong, nominal, Blodgett, SED, linearization, notch stress, multiaxial)
- S-N curve fatigue assessment (IIW, DNV, ASME)
- Rainflow cycle counting and Miner cumulative damage
- Monte Carlo probabilistic analysis with LHS
- 49 bundled material databases
- 80+ IIW weld detail classifications
- Parametric study with concurrent execution
- PyVista 3D and Matplotlib 2D visualization
- HTML report generation
- CLI with run, blodgett, visualize, dashboard commands
