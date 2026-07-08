# Convergence & submodeling

Stress at a weld toe is mesh-sensitive, and at a sharp re-entrant corner it is
genuinely singular — the peak stress keeps rising as you refine, never converging.
feaweld gives you three tools to handle this: an automatic **singularity check**
after every linear/elastoplastic solve, a **mesh convergence study** with Richardson
extrapolation and the Grid Convergence Index, and **submodeling** to re-solve a
refined local region driven by the global solution.

## Richardson extrapolation and GCI

Run the same case at successively coarser meshes and you can extrapolate to the
zero-mesh-size limit. With three grids of sizes $h_1 < h_2 < h_3$ and results
$f_1, f_2, f_3$, the observed order of convergence is

$$p = \frac{\ln\!\big((f_3 - f_2)/(f_2 - f_1)\big)}{\ln r}, \qquad r = h_2/h_1$$

and the Richardson-extrapolated value is

$$f_{h\to 0} = f_1 + \frac{f_1 - f_2}{r^{p} - 1}.$$

The **Grid Convergence Index** turns the extrapolation error into a reported
uncertainty band:

$$\text{GCI} = F_s\,\frac{|\varepsilon|}{r^{p} - 1}, \qquad
\varepsilon = \frac{f_{coarse} - f_{fine}}{f_{fine}}$$

with a safety factor $F_s = 1.25$ for three-or-more grids. feaweld considers the
quantity converged when GCI < 5 %. A very low observed order or a non-converging
GCI is the signature of a stress singularity, not of a good enough mesh.

## The `feaweld convergence` command

```bash
feaweld convergence case.yaml --levels 4 --ratio 2.0
```

This runs the case at `--levels` meshes, each `--ratio`× coarser than the last
(the singularity check is disabled during the study to avoid redundant coarse
solves), and reports the GCI of the peak von Mises stress:

```
Level 0: global=2 mm, toe=0.2 mm ...
  max von Mises = 291.44 MPa
Level 1: global=4 mm, toe=0.4 mm ...
  max von Mises = 268.10 MPa
Level 2: global=8 mm, toe=0.8 mm ...
  max von Mises = 231.77 MPa

Convergence study (peak von Mises):
  extrapolated value = 305.12 MPa
  observed order     = 1.31
  GCI                = 4.2%
  status             = converged
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--levels` | `3` | Number of refinement levels (minimum 3) |
| `--ratio` | `2.0` | Mesh-size ratio between successive levels |

The underlying `convergence_study(values, sizes)` returns a `ConvergenceResult`
with `extrapolated_value`, `convergence_order`, `gci`, and `is_converged`.

## The automatic singularity check

For `linear_elastic` and `elastoplastic` solves without a thermal pass,
`run_analysis()` runs a lightweight singularity check by default. It re-solves the
case on one coarser mesh and compares the two stress fields node-by-node
(`detect_singularities`): a node whose stress keeps climbing with refinement and
whose convergence rate is below 0.5 is flagged as likely singular.

Configure it in the `postprocess:` block:

```yaml
postprocess:
  singularity_check: true        # default
  singularity_threshold: 0.20    # flag nodes whose stress rises > 20% on refinement
  singularity_coarsening: 2.0    # coarse-mesh size factor for the check solve
```

The result is attached to `WorkflowResult.postprocess_results["singularity_check"]`
(with `n_flagged`, `n_singular`, `singular_node_ids`, `min_convergence_rate`), and
if any node is flagged a warning is added to `WorkflowResult.warnings`:

!!! warning "What a flag means"
    A flagged node usually indicates a mesh-driven peak at a re-entrant corner, not
    a real stress. Prefer a mesh-insensitive assessment — hot-spot extrapolation,
    the Dong structural stress, effective notch stress, or SED — over the raw peak
    von Mises at that node. Notch rounding (the fictitious 1 mm radius) is the other
    standard remedy.

## Submodeling

When you *do* need an accurate local stress, submodeling re-solves a small region
at high refinement, imposing cut-boundary displacements interpolated from the
global solution:

```bash
feaweld submodel case.yaml --center "0,20,0" --radius 5 --refine 4 --backend auto
```

```
Running global analysis: fillet_t_joint_example
Solving submodel (centre=[0.0, 20.0, 0.0], radius=5.0, refine=4)

  solve mode:     CalculiXBackend
  submodel nodes: 4821
  max von Mises:  global 231.77 MPa -> submodel 288.13 MPa
  boundary BC residual: 3.114e-04 mm
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--center` | required | Submodel centre `"x,y,z"` (mm) |
| `-r/--radius` | required | Extraction radius (mm) |
| `--refine` | `4` | Mesh refinement factor for the local region |
| `--backend` | `auto` | FEA backend for the local re-solve |

### Re-solve vs. interpolation fallback

`solve_submodel()` has two modes, reported in `metadata["submodel_solve"]`:

- **Backend re-solve** — when a backend and material are available, the refined
  region is genuinely re-solved with `solve_static`, and
  `metadata["submodel_solve"]` is the backend class name (e.g. `CalculiXBackend`).
  A `boundary_bc_residual` is reported: the maximum mismatch between the imposed and
  solved cut-boundary displacements, a sanity check on the boundary interpolation.
- **Interpolation fallback** — if no backend/material is available or the re-solve
  fails, the parent displacement and stress are interpolated onto the refined mesh
  and `metadata["submodel_solve"]` is the literal `"interpolated"` (no
  `boundary_bc_residual`). This gives a smoother visualization but not an
  independent solve.

```python
from feaweld.singularity.submodeling import solve_submodel
from feaweld.core.materials import load_material

sub = solve_submodel(
    global_results, center=[0, 20, 0], radius=5.0,
    material=load_material("A36"), refinement_factor=4, backend="auto",
)
print(sub.metadata["submodel_solve"])
```

## See also

- [Solvers](solvers.md) — which solve types the singularity check applies to.
- [Custom post-processing](../tutorials/03_custom_postprocessing.md) — the
  mesh-insensitive methods to prefer near a singularity.
- [API reference](../api/singularity.md) — full `singularity` package docs.
