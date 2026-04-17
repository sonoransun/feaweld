# Tutorial: YAML analysis case

This tutorial walks through defining a complete fatigue-assessment analysis case in YAML, running it from the CLI, and inspecting the HTML report that comes out.

## Prerequisites

- feaweld installed with visualization extras: `pip install -e ".[viz]"`
- At least one FEA backend: `pip install -e ".[fenics]"` **or** `pip install -e ".[calculix]"`

## The case file

Analysis cases are Pydantic models (see `feaweld.pipeline.workflow.AnalysisCase`) that serialize cleanly to YAML. Every field has a sensible default, so you only specify what differs from the defaults.

Save the following as `my_joint.yaml`:

```yaml
name: fillet_t_50kn
description: Fillet T-joint under 50 kN axial load, IIW hot-spot + Dong

material:
  base_metal: A36          # any key from `feaweld materials`
  weld_metal: E70XX
  haz: A36
  temperature: 20.0        # °C

geometry:
  joint_type: FILLET_T     # FILLET_T, BUTT, LAP, CORNER, CRUCIFORM
  base_width: 200.0        # mm
  base_thickness: 20.0
  web_height: 100.0
  web_thickness: 10.0
  weld_leg_size: 8.0
  length: 1.0              # extrusion depth (1.0 = quasi-2D)

mesh:
  global_size: 2.0         # mm — background element size
  weld_toe_size: 0.2       # mm — refinement near the toe
  element_order: 2         # 1 = linear, 2 = quadratic
  element_type: tri        # tri | quad | tet | hex

solver:
  solver_type: LINEAR_ELASTIC  # LINEAR_ELASTIC, ELASTOPLASTIC, THERMAL_*, THERMOMECHANICAL, CREEP
  backend: auto                # auto | fenics | calculix

load:
  axial_force: 50000.0     # N
  bending_moment: 0.0      # N·mm
  shear_force: 0.0
  pressure: 0.0

postprocess:
  stress_methods:
    - HOTSPOT_LINEAR
    - STRUCTURAL_DONG
    - BLODGETT
  sn_curve: IIW_FAT90
  fatigue_assessment: true

output_dir: results/fillet_t_50kn
```

### Field reference

| Section | Purpose | Where to look |
|---------|---------|---------------|
| `material` | Base / weld / HAZ metal and service temperature | `feaweld.core.materials` |
| `geometry` | Joint type + parametric dimensions | `feaweld.geometry.joints` |
| `mesh` | Gmsh sizing + element order | `feaweld.mesh.generator` |
| `solver` | Physics type + backend selection | `feaweld.solver.backend` |
| `load` | Mechanical + pressure + thermal delta | `feaweld.core.loads` |
| `postprocess` | Which stress methods to run + S-N curve | `feaweld.postprocess.*` |

## Run it

```bash
feaweld run my_joint.yaml
```

Expected output:

```
Loading case: my_joint.yaml
Running analysis: fillet_t_50kn
  Joint: fillet_t
  Material: A36
  Solver: linear_elastic

Analysis completed successfully.

Max von Mises stress: 287.43 MPa

Fatigue results:
  hotspot_linear: N = 482133 cycles
  structural_dong: N = 515408 cycles

Report: results/fillet_t_50kn/report.html
```

## What's in the report

Open `results/fillet_t_50kn/report.html` in a browser. Each section is driven by one post-processing method:

- **Through-thickness linearization** — membrane / bending / peak decomposition per ASME VIII Div 2 (only if `LINEARIZATION` is in `stress_methods`).
- **Hot-spot extrapolation** — 0.4 t / 1.0 t reference stresses and σ<sub>hs</sub> (IIW).
- **Dong decomposition** — stacked membrane + bending + bending-ratio overlay.
- **S-N curve** — operating point plotted against the selected curve.
- **Blodgett summary** — weld-group geometry and LRFD / ASD capacities.

Figures are embedded as base64 PNGs — the report is a single self-contained HTML file you can email or archive.

## Generate interactive figures instead

Pass `--interactive` to replace the static PNGs with Plotly figures (hover to inspect exact stress values, toggle series, zoom in):

```bash
feaweld run my_joint.yaml --interactive
```

See [Visualization guide](../guides/visualization.md) for the full list of available plots.

## Programmatic equivalent

The same case, built from Python:

```python
from feaweld.pipeline.workflow import (
    AnalysisCase, MaterialConfig, GeometryConfig, MeshConfig,
    SolverConfig, LoadConfig, PostProcessConfig, run_analysis,
)
from feaweld.pipeline.report import generate_report
from feaweld.core.types import JointType, SolverType, StressMethod

case = AnalysisCase(
    name="fillet_t_50kn",
    material=MaterialConfig(base_metal="A36", weld_metal="E70XX"),
    geometry=GeometryConfig(joint_type=JointType.FILLET_T, weld_leg_size=8.0),
    solver=SolverConfig(solver_type=SolverType.LINEAR_ELASTIC),
    load=LoadConfig(axial_force=50000.0),
    postprocess=PostProcessConfig(
        stress_methods=[StressMethod.HOTSPOT_LINEAR, StressMethod.STRUCTURAL_DONG],
        sn_curve="IIW_FAT90",
    ),
)

result = run_analysis(case)
report_path = generate_report(result)
```

## Common extensions

- **Change the S-N curve** — set `postprocess.sn_curve` to any `IIW_FAT*`, `DNV_*`, or `ASME_*` name. Run `python -c "from feaweld.fatigue.sn_curves import list_curves; print(list_curves())"` for the full list.
- **Use a different joint** — swap `joint_type` and the matching `geometry` fields (e.g. `BUTT` uses `base_width` + `base_thickness` only; `LAP` adds `web_height` as the overlap).
- **Add thermal welding simulation** — set `solver.solver_type: THERMOMECHANICAL` and supply a `thermal:` section (see `ThermalConfig` in `workflow.py`).

## Next

- [Parametric study tutorial](02_parametric_study.md) — sweep weld-leg size or load and produce a comparison report.
- [Custom post-processing tutorial](03_custom_postprocessing.md) — add your own stress-extraction method.
