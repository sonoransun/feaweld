# Tutorial: Parametric study

This tutorial walks through sweeping a parameter (weld leg size) across an analysis case, running every variant in parallel, and producing an automated comparison report.

Prerequisites: complete the [YAML analysis tutorial](01_yaml_analysis.md) first.

## Two ways to define a sweep

### 1. YAML study file (declarative)

Save this as `leg_sweep.yaml`:

```yaml
name: weld_leg_sweep
description: Effect of fillet weld leg size on fatigue life
mode: grid             # "grid" = Cartesian product of sweeps; "one_at_a_time" = vary one field at a time
max_workers: 4

base_case:
  name: base
  material: {base_metal: A36, weld_metal: E70XX}
  geometry:
    joint_type: FILLET_T
    base_width: 200.0
    base_thickness: 20.0
    web_height: 100.0
    web_thickness: 10.0
    weld_leg_size: 8.0   # placeholder, overridden by sweep
  mesh: {global_size: 2.0, weld_toe_size: 0.2, element_order: 2}
  solver: {solver_type: LINEAR_ELASTIC, backend: auto}
  load: {axial_force: 50000.0}
  postprocess:
    stress_methods: [HOTSPOT_LINEAR, STRUCTURAL_DONG]
    sn_curve: IIW_FAT90

parameters:
  - name: geometry.weld_leg_size
    values: [6.0, 8.0, 10.0, 12.0]
```

Parameter names are **dot-paths** into `AnalysisCase`, so any nested field works:

```yaml
parameters:
  - name: load.axial_force
    values: [20000, 40000, 60000, 80000]
  - name: mesh.global_size
    values: [1.0, 2.0, 4.0]
```

With `mode: grid` that produces 4 × 3 = 12 cases.

Run it:

```bash
feaweld study run leg_sweep.yaml -j 4
```

Expected:

```
Loading study: leg_sweep.yaml
Study: weld_leg_sweep (grid mode)
Generated 4 cases, running with 4 workers...

Completed in 28.3s
  Succeeded: 4/4

Comparison report: results/comparison_report.html
```

### 2. Python fluent API (programmatic)

```python
from feaweld.pipeline.workflow import AnalysisCase, GeometryConfig, LoadConfig
from feaweld.pipeline.study import Study
from feaweld.pipeline.comparison import generate_comparison_report
from feaweld.core.types import JointType

base = AnalysisCase(
    name="base",
    geometry=GeometryConfig(joint_type=JointType.FILLET_T, weld_leg_size=8.0),
    load=LoadConfig(axial_force=50000.0),
)

study = (
    Study("leg_sweep", base)
    .vary("geometry.weld_leg_size", [6.0, 8.0, 10.0, 12.0])
)
results = study.run(max_workers=4, mode="grid")

report_path = generate_comparison_report(results, "results/")
print(f"Report: {report_path}")
```

The `Study` API chains fluently: call `.vary()` multiple times to build up the sweep, `.add_case()` to add a one-off named case, then `.run()` to execute.

## What's in the comparison report

The comparison report contains:

- **Metric table** — per-case summary of max von Mises stress, hot-spot stress, predicted life, and any user-defined metric.
- **Delta vs. baseline** — pick any case as the baseline (`--baseline <case_name>`); the table then shows Δ % relative to it.
- **Sensitivity plots** — for every swept parameter, a line plot of the response (e.g. fatigue life) against the parameter.
- **Stress envelopes** — min / max stress contours across all cases overlaid.

## Accessing results programmatically

```python
results = study.run(max_workers=4, mode="grid")

# Iterate all successful results
for case_name, workflow_result in results.successful_results.items():
    life = workflow_result.fatigue_results["hotspot_linear"]["life"]
    print(f"{case_name}: N = {life:.0f} cycles")

# Access a specific case
r = results["leg_sweep_weld_leg_size_12.0"]
print(r.fea_results.stress.von_mises.max())
```

`StudyResults` supports dict-style access (`results[case_name]`), iteration (`for case, r in results`), and `.successful_results` / `.errors` dicts.

## Comparing cases from different YAML files

Sometimes you want to compare cases built separately (e.g. different joint types, not a regular sweep):

```bash
feaweld study compare case_a.yaml case_b.yaml case_c.yaml --baseline case_a
```

Each YAML is loaded as its own `AnalysisCase`, all are run, and a single comparison report is produced with `case_a` as the baseline.

## Tips

- **Use `mode: one_at_a_time` for sensitivity analyses.** Grid mode explodes combinatorially; one-at-a-time varies each parameter independently around the base case (`n` parameters × `k` values = `n·k` cases, not `k^n`).
- **Start coarse.** Run a 4-value sweep first; refine to 8 or 16 values only after you've seen the trend.
- **Concurrency = cores, not more.** `-j 4` is sensible on a 4- or 8-core machine; more workers than physical cores wastes memory and is typically slower for FEA workloads.
- **Pickle compatibility.** `Study.run()` uses `ProcessPoolExecutor`, so every case must be picklable. This is automatic for Pydantic models but breaks if you attach a lambda or live Gmsh handle to the case.

## Next

- [Custom post-processing tutorial](03_custom_postprocessing.md) — extend the pipeline with your own stress method.
