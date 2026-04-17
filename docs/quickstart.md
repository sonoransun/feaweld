# Quickstart

Run your first fatigue analysis in under a minute.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[viz]"
```

See [Installation](installation.md) for optional extras.

## 2. Run the bundled example

```bash
python examples/fillet_t_joint.py
```

This runs a full pipeline on a fillet T-joint: geometry → mesh → linear-elastic solve → hot-spot + Dong + Blodgett post-processing → IIW FAT90 fatigue → HTML report.

Expected output:

```
Running analysis: fillet_t_joint_example
Analysis completed successfully!
Report: results/fillet_t_joint/report.html
```

Open the report in a browser — you'll see embedded figures for through-thickness linearization, hot-spot extrapolation, the S-N curve with operating point, and the Dong decomposition, along with the fatigue life prediction.

## 3. Run from the CLI with a YAML case

The same analysis can be driven from a YAML file:

```yaml
# case.yaml
name: fillet_t_joint_example
description: Fillet welded T-joint under axial tension
material:
  base_metal: A36
  weld_metal: E70XX
  haz: A36
  temperature: 20.0
geometry:
  joint_type: FILLET_T
  base_width: 200.0
  base_thickness: 20.0
  web_height: 100.0
  web_thickness: 10.0
  weld_leg_size: 8.0
  length: 1.0
mesh:
  global_size: 2.0
  weld_toe_size: 0.2
  element_order: 2
solver:
  solver_type: LINEAR_ELASTIC
  backend: auto
load:
  axial_force: 50000.0
postprocess:
  stress_methods: [HOTSPOT_LINEAR, STRUCTURAL_DONG, BLODGETT]
  sn_curve: IIW_FAT90
  fatigue_assessment: true
output_dir: results/fillet_t_joint
```

```bash
feaweld run case.yaml
```

## 4. Hand calculation (no FEA)

For weld-group sizing per Blodgett:

```bash
feaweld blodgett -g box --d 100 --b 50 -t 5 -P 50000
```

Outputs weld-group section properties, component stresses, and LRFD/ASD capacities.

## 5. Parametric study

Define sweeps in YAML and run them in parallel:

```yaml
# study.yaml
name: weld_leg_sweep
mode: grid
base_case: case.yaml
parameters:
  - name: geometry.weld_leg_size
    values: [6.0, 8.0, 10.0, 12.0]
```

```bash
feaweld study run study.yaml -j 4
```

A comparison report is emitted with delta tables and sensitivity plots.

## Next steps

- [YAML analysis tutorial](tutorials/01_yaml_analysis.md) — deeper dive into case options.
- [Parametric study tutorial](tutorials/02_parametric_study.md) — grid vs. one-at-a-time, comparison reports.
- [Custom post-processing tutorial](tutorials/03_custom_postprocessing.md) — adding your own stress-extraction method.
- [Visualization guide](guides/visualization.md) — every plot the package produces, with signatures.
