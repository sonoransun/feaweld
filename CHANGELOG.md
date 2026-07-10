# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-09

### Added

**Analysis capabilities**
- All 8 `StressMethod` values are now reachable from YAML cases: `hotspot_quadratic`,
  `notch_stress` (effective notch, FAT225), `strain_energy_density` (Lazzarin SED),
  and `linearization` (ASME through-thickness) join the previously wired methods.
  New `postprocess` config fields: `notch_radius`, `notch_nominal_stress`,
  `sed_control_radius`, `sed_w_ref`, `linearization_points`.
- `solver.solver_type` is now dispatched: `elastoplastic` (J2 radial-return stress
  correction), `thermal_steady`, `thermal_transient`, `thermomechanical`
  (sequential coupling with a Goldak heat source), and `creep` (Norton-Bailey
  isothermal relaxation via new `solver.creep_temperature` / `creep_time_hours`).
  `solver.nonlinear`, `max_iterations`, and `tolerance` are honored.
- Full load application: `bending_moment` (self-equilibrating nodal couple via new
  `core.loads.moment_to_nodal_forces`), `shear_force`, `pressure`, and
  `temperature_delta` (thermoelastic load) are applied by both solver backends.
- PWHT wiring: `thermal.pwht_enabled` now runs Norton-Bailey stress relaxation on
  the solved field (new `pwht_heating_rate` / `pwht_cooling_rate` config; the
  as-welded peak stress is preserved in result metadata).
- `postprocess.singularity_check` is now wired: a coarse-mesh re-solve flags
  non-converging stress peaks and attaches a summary + human-readable warning.
- Probabilistic upgrades: `include_material_scatter` and
  `include_geometric_tolerance` are honored (misalignment magnification and
  parametric toe SCF enter the response model); results now include raw
  `samples`/`results` arrays, `percentiles`, convergence info, and optional
  Sobol indices (`probabilistic.sobol`, `sobol_n_base`, `seed`).
- Real submodel FE solve: `singularity.submodeling.SubmodelSolver` re-solves the
  refined local region through a solver backend with cut-boundary displacement
  BCs (new `solve_submodel()` convenience; graceful interpolation fallback with
  a `RuntimeWarning` when no backend/material is available).
- `WorkflowResult.warnings` — non-fatal analysis warnings, also shown in reports.
- New CLI commands: `feaweld reliability mc|form|sobol`, `feaweld ml
  train|predict|transfer`, `feaweld multiscale`, `feaweld convergence`,
  `feaweld submodel`, and `feaweld twin dashboard|update`.
- `u_shape` added to `feaweld blodgett` geometry choices.
- `fatigue.sn_curves.parse_sn_spec()` for `"IIW_FAT90"`-style curve specs.
- YAML enum values are now case-insensitive and accept member names
  (`FILLET_T`, `fillet_t`, and `SED` all validate); serialization still
  emits the canonical lowercase values.

**Fatigue**
- A top-level `fatigue:` block turns the one-shot check into a spectrum
  assessment: cyclic loading as an R-ratio, multi-block spectra, an inline
  history, or a `history_file` CSV (factor or absolute stress), rainflow-counted
  (ASTM E1049) and Palmgren-Miner-summed in `run_analysis`.
- New `fatigue/assessment.py` spectrum engine (`build_cycle_set`,
  `scale_cycles`, `assess_spectrum`, `spectrum_life_power_law`); the governing
  cycles are stashed to `postprocess_results["rainflow"]`, feeding report
  figures and `feaweld animate`.
- Mean-stress corrections (Goodman/Gerber, optionally including a residual
  mean) plus IIW thickness `f(t)`, surface-roughness, and environment
  knockdowns are applied to the spectrum ranges.
- Methods that mandate their own S-N curve are assessed on it: effective notch
  on FAT225 (with corrections), Dong and SED power-law-scaled.
- New `feaweld fatigue` CLI: standalone spectrum assessment from a CSV history
  or a constant-amplitude cycle, with corrections, a residual profile, and
  `--list-curves`.

**S-N standards**
- Eurocode 3 EN 1993-1-9: 14 detail categories, two-slope (m = 3/5), CAFL at
  5e6, cutoff at 1e8 (`ec3_curve`).
- BS 7608: 8 classes with mean and design (mean − 2SD) curves and a Haibach
  knee at 1e7 (`bs7608_curve`).
- AWS D1.1: 7 categories A–E′ (including primed aliases) with a CAFL threshold
  (`aws_curve`).
- `SNStandard` gains `EC3`/`BS7608`/`AWS` members, `get_sn_curve` dispatches to
  them, and the spec grammar accepts `EC3_90` / `BS7608_F2` / `AWS_C`; each
  ships a bundled S-N data file.

**Residual stress**
- Residual stress enters fatigue as a mean-stress shift from a bundled profile,
  a fixed value, or the as-welded yield; an optional `superimpose` adds the
  through-thickness field (new `residual_stress_field` /
  `surface_residual_stress`, plate-surface anchored) to the saved stress output.
- PWHT now relaxes the residual field via a Norton-Bailey probe (new
  `solver.creep.pwht_relaxation_factor`) when a residual profile is configured.

**3D analysis**
- `geometry.dimension: 3` extrudes all five joint types along `length`
  (template-method refactor with volume/face/toe-edge physical groups and a
  `get_weld_toe_lines` / `get_weld_toe_normals` API).
- Tetrahedral 3D meshing with toe-line Distance refinement (`element_type_3d`,
  `refinement_distance`; hex guarded); hot-spot stress is evaluated per station
  along each weld toe line with `critical_line` / `n_stations` reporting and
  unresolved-station exclusion.
- Linearization, nominal, notch, SED, and Blodgett run in 3D; `structural_dong`
  and the singularity check are 2D-only (informative guard / auto-skip).
- FEniCSx 3D linear-tet solves (TET10 guarded) and CalculiX C3D4/C3D10 with the
  Gmsh → ccx TET10 mid-edge node permutation.

**Configuration**
- Butt-weld `groove_angle`, `root_gap`, and `penetration` (validated
  `full`/`partial`) are exposed in `GeometryConfig`.
- `PostProcessConfig.weld_efficiency` (a lookup name or a direct value) scales
  all four ASME allowable checks.

**Visualization & reporting**
- HTML reports now embed 3D PyVista renderings (stress contour, deformed shape,
  mesh preview, temperature field) plus spatial 2D contours, through-thickness
  linearization, ASME check bars, rainflow histograms, fatigue-life and Miner
  damage maps, Monte Carlo distributions, and Sobol index charts — each included
  automatically when its data is present (declarative `FigureSpec` registry).
- Report generation migrated to real Jinja2 templates
  (`feaweld/pipeline/templates/`); reports remain single self-contained files.
- New plot families: `visualization.probabilistic_plots` (Sobol indices, MC
  histogram + CDF, FORM design point) and `visualization.material_plots`
  (CCT diagrams, residual-stress profiles, Norton-Bailey creep curves,
  Hall-Petch curves); `plot_stress_contour_2d` and `plot_temperature_history`.
- 3D plot functions accept ready PyVista grids as well as `FEMesh`/`StressField`
  (new `stress_plots.resolve_grid` / `resolve_component`).
- Study comparison reports gain per-parameter sensitivity figures (auto-detected
  swept parameters via new `detect_swept_parameters`) and a stress-difference
  contour against the baseline.
- `feaweld visualize` rewritten over the visualization library: semantic
  colormaps (turbo, not jet), `--iso`, `--deformed`, `--clip-origin`, `--below`,
  `--cmap`, and principal-stress components.
- `feaweld.visualization` exposes a lazy public API (PEP 562) — plot functions
  are importable from the package root without importing matplotlib/pyvista.
- Digital-twin web dashboard: a self-contained HTML/JS client (live sensor strip
  charts, weld state machine, alert feed, predictions) served by
  `feaweld twin dashboard`, with an offline demo mode (`--demo`).

**Documentation**
- Mermaid diagram support in MkDocs; new architecture page with pipeline,
  module-map, dispatch, and class diagrams; new guides for solvers, loads,
  PWHT, probabilistic/reliability, ML, multiscale, convergence/submodeling,
  and the digital twin; a full CLI reference.
- Shipped runnable example YAMLs (`examples/fillet_t_joint.yaml`,
  `examples/leg_sweep_study.yaml`) and `examples/README.md`.
- CONTRIBUTING guide and this changelog.
- Two new guides (fatigue assessment, 3D analysis) and an `api/data.md` page;
  six new concept figures and three new report figures (rainflow matrix, damage
  per block, Haigh diagram); three new examples (`spectrum_fatigue.yaml`,
  `load_history.csv`, `fillet_t_joint_3d.yaml`); Python 3.14 classifier.

### Changed
- Docstrings standardized to numpy style across the codebase (matching the
  mkdocstrings configuration); inline Sphinx roles replaced with mkdocstrings
  cross-references and MathJax math.
- Report engine replaced (`str.replace` string template → Jinja2).
- `generate_report_figures()` now returns
  `{key: {"b64", "caption", "layout"}}` instead of `{key: str}`.
- Probabilistic results for existing cases now include geometric tolerance
  scatter by default (the `include_geometric_tolerance` flag was previously
  parsed but ignored), so mean/cov values shift accordingly.
- `pyproject.toml` metadata: authors, keywords, classifiers, project URLs.
- Residual stress `superimpose` adds the through-thickness field to the saved
  stress output only, after post-processing — never into the fatigue ranges.
- PWHT relaxes the residual-stress field rather than the load field when a
  residual profile is configured (the legacy load-field path is preserved with
  a warning otherwise).
- Study parameter sweeps now walk nested attribute paths of any depth and raise
  on an invalid path (deeper-than-two-level paths were previously dropped
  silently); `load_study` sets the study `_base_dir`.

### Fixed
- Meshes now carry a `weld_toe` node set (nearest nodes to the joint's
  weld-toe coordinates). Previously no such set was ever created, so every
  YAML-driven stress method silently evaluated the clamped corner at node 0
  instead of the weld toe; a loud warning is emitted if the set is missing.
- J2 radial-return plasticity subtracted the plastic corrector twice,
  returning stresses far inside the yield surface; the return-mapped stress
  now lands exactly on the hardened yield surface, and the yield surface
  hardens across load increments (increment-count-independent results).
- FEniCSx backend contract violations: Dirichlet boundary conditions were
  applied to the wrong nodes after DOLFINx vertex reordering (now matched by
  coordinates); results were returned in DOLFINx internal order (now input
  node order, with DG0 cell stress averaged to nodes so both backends return
  nodal fields); directional force loads were smeared as uniform traction
  over the whole exterior boundary (now applied as per-node forces matching
  CalculiX totals); stress recovery omitted the thermal eigenstrain for
  temperature loads.
- Effective notch stress was mesh-dependent (raw singular corner peak over an
  inconsistent nominal section); it is now a parametric-SCF estimate applied
  to the FE-linearized structural stress at the toe, and the FAT225 curve is
  a proper two-slope IIW curve with a continuous knee at 131.6 MPa.
- Probabilistic yield/UTS/modulus distributions are re-centered on the
  actual case material (the previous generic fallback reported safety factors
  ~40% unconservative for the default A36); axial misalignment now magnifies
  only the membrane stress component.
- Singularity check no longer compares a linear coarse solve against
  plasticity-capped or PWHT-relaxed fields (false negatives); it runs only
  for plain linear-elastic solves.
- Digital-twin dashboard broadcasts from MQTT network threads were silently
  dropped (loop-safe scheduling via `run_coroutine_threadsafe`); alert and
  sensor history buffers are now bounded.
- `ml predict` rejected neither unknown feature names (silently zero-filled)
  nor could it load models saved by `ml transfer`; transfer learning now
  aligns feature columns to the base model's order.
- `study compare` ran a phantom duplicate baseline case; `study run` ignored
  the study file's `max_workers`.
- Fatigue-life maps flattened when any node's life was infinite (field now
  clipped at both ends); 2D stress contours painted only half of each quad
  element; comparison-report columns vanished if the first case failed;
  reports are written as UTF-8 explicitly.
- Axial force loads are now actually applied: the previous
  `values=[0, F, 0]` / no-direction boundary condition was silently skipped by
  the FEniCSx backend and applied as zero by the CalculiX backend.
- Welding thermal runs (`thermal.enabled`) now build a proper thermal load case
  (ambient sink + convection) and a Goldak heat source instead of passing the
  mechanical load case to both fields of the coupled solve.
- `feaweld animate` called `get_sn_curve()` with the wrong arity.
- Submodel boundary interpolation no longer fails on planar (quasi-2D) meshes
  (degenerate RBF axes are dropped, with nearest-neighbour fallback).
- `docs/api/probabilistic.md` and `docs/api/ml.md` described modules that do
  not exist (`reliability`, `training`); README and quickstart referenced
  example YAML files that were not shipped; the quickstart study snippet used
  an invalid `base_case: <path>` form.
- `examples/butt_weld_fatigue.py` imported a nonexistent
  `fatigue_life_from_damage`; the helper now exists in `fatigue/miner.py`.
- Orphan duplicate nodes left by Distance-field construction (2D free points and
  3D toe lines) are filtered out in `extract_mesh_from_gmsh`.
- The O(n_group × n_elem) physical-group-to-element mapping is vectorized.
- Hot-spot extrapolation no longer assumes sorted weld-tangent node ids and
  signs the surface direction away from the weld, with snapping diagnostics.
- Documentation corrections: IIW detail count 80 → 100, the DNV category count,
  `--clip` usage, image paths, the nonexistent `A387_Gr91` material, the notch
  nominal-fallback description, stale index metrics, and a missing `api/data.md`
  page.

### Removed
- `docs/VISUALIZATION_GUIDE.md` (byte-identical orphan duplicate of
  `docs/guides/visualization.md`).
- Unused `mkdocs-gen-files` from the `docs` extra.

### Notes
- `postprocess.singularity_check` defaults to `true` and now performs one
  additional coarse-mesh solve per static analysis. Set it to `false` to skip.

## [0.1.0] - 2026-07-07

### Added
- Initial release: YAML-driven FEA pipeline for weld joints (geometry → Gmsh
  mesh → FEniCSx/CalculiX solve → post-processing → fatigue → HTML report),
  7 post-processing method modules, IIW/DNV/ASME S-N curves, rainflow counting
  and Palmgren-Miner damage, Blodgett weld-group hand calculations, parametric
  studies, probabilistic analysis, ML fatigue prediction, digital-twin
  ingestion/updating, multiscale property estimation, singularity detection,
  and a bundled reference-data library (49 materials, 100 IIW weld details,
  CCT/SCF/residual-stress/filler-metal/weld-efficiency datasets).
