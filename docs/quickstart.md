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
  joint_type: fillet_t
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
  solver_type: linear_elastic
  backend: auto
load:
  axial_force: 50000.0
postprocess:
  stress_methods: [hotspot_linear, structural_dong, blodgett]
  sn_curve: IIW_FAT90
  fatigue_assessment: true
output_dir: results/fillet_t_joint
```

```bash
feaweld run case.yaml
```

A ready-made copy of this case ships with the package as `examples/fillet_t_joint.yaml`, so you can run it directly without writing `case.yaml` first:

```bash
feaweld run examples/fillet_t_joint.yaml
```

## 4. Hand calculation (no FEA)

For weld-group sizing per Blodgett:

```bash
feaweld blodgett -g box --d 100 --b 50 -t 5 -P 50000
```

Outputs weld-group section properties, component stresses, and LRFD/ASD capacities.

## 5. Parametric study

Define sweeps in YAML and run them in parallel. The `base_case` is a full inline
analysis case (a path to a separate case file is **not** accepted), and each
`parameters` entry is a dot-path into that case:

```yaml
# study.yaml
name: weld_leg_sweep
mode: grid
base_case:
  geometry: {joint_type: fillet_t, weld_leg_size: 8.0}   # weld_leg_size overridden by the sweep
  load: {axial_force: 50000.0}
  postprocess: {stress_methods: [hotspot_linear, structural_dong], sn_curve: IIW_FAT90}
parameters:
  - name: geometry.weld_leg_size
    values: [6.0, 8.0, 10.0, 12.0]
```

```bash
feaweld study run study.yaml -j 4
```

A ready-made copy ships as `examples/leg_sweep_study.yaml` — run it directly with
`feaweld study run examples/leg_sweep_study.yaml -j 4`. A comparison report is
emitted with delta tables and sensitivity plots.

!!! tip "Spectrum fatigue"
    A top-level `fatigue:` block upgrades the one-shot S-N check to a full
    spectrum assessment — rainflow counting, Miner damage, mean-stress
    correction, and residual stress. Try the shipped variable-amplitude case:

    ```bash
    feaweld run examples/spectrum_fatigue.yaml
    ```

    or assess a stress history with no FEA at all:

    ```bash
    feaweld fatigue --history examples/load_history.csv -c EC3_90
    ```

    See the [Fatigue assessment guide](guides/fatigue_assessment.md).

!!! tip "Go 3D"
    Setting `geometry.dimension: 3` extrudes any joint into a solid — `length`
    becomes the real weld length, and the hot-spot method runs per station
    along each weld-toe line. Try the shipped extruded T-joint (note its
    deliberately coarser mesh sizes):

    ```bash
    feaweld run examples/fillet_t_joint_3d.yaml
    ```

    See the [3D analysis guide](guides/analysis_3d.md).

## Next steps

- [YAML analysis tutorial](tutorials/01_yaml_analysis.md) — deeper dive into case options.
- [Parametric study tutorial](tutorials/02_parametric_study.md) — grid vs. one-at-a-time, comparison reports.
- [Custom post-processing tutorial](tutorials/03_custom_postprocessing.md) — adding your own stress-extraction method.
- [Visualization guide](guides/visualization.md) — every plot the package produces, with signatures.
