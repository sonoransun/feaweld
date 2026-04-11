# Visualization Guide

This guide covers all visualization capabilities in feaweld, with examples and usage patterns.

## Installation

All visualization functions require optional dependencies:

```bash
pip install feaweld[viz]   # matplotlib + pyvista + vtk
```

Functions are safe to import even without these packages — they raise a helpful error only when called.

## 2D Plots (Matplotlib)

All 2D functions live in `feaweld.visualization.plots_2d` and follow a consistent signature:

```python
def plot_*(result, title="...", show=True, ax=None) -> Figure
```

- `show=False` for non-interactive/headless use
- Pass `ax` to embed in an existing subplot (e.g., dashboards)

### Through-Thickness Linearization

Decomposes stress into membrane, bending, and peak per ASME VIII Div 2.

```python
from feaweld.visualization.plots_2d import plot_through_thickness
fig = plot_through_thickness(linearization_result, show=False)
```

<img src="images/example_through_thickness.svg" alt="Through-thickness" width="80%">

Features: Reference lines at membrane/MB scalar values, decomposition equation box, inner/outer surface labels.

### Hot-Spot Stress Extrapolation

Extrapolates structural stress to the weld toe from reference points per IIW.

```python
from feaweld.visualization.plots_2d import plot_hotspot_extrapolation
fig = plot_hotspot_extrapolation(hotspot_result, show=False)
```

<img src="images/example_hotspot.svg" alt="Hot-spot extrapolation" width="80%">

Features: Schematic weld toe profile, horizontal hot-spot reference line, IIW type annotation (Type A/B).

### Dong Structural Stress Decomposition

Stacked bars for membrane/bending with bending ratio overlay.

```python
from feaweld.visualization.plots_2d import plot_dong_decomposition
fig = plot_dong_decomposition(dong_result, show=False)
```

<img src="images/example_dong.svg" alt="Dong decomposition" width="80%">

Features: Dual-axis (stress + bending ratio), formula box with structural stress equation.

### S-N Curve

Log-log fatigue curve with operating point and regime bands.

```python
from feaweld.visualization.plots_2d import plot_sn_curve
fig = plot_sn_curve(curve, stress_range=120.0, show=False)
```

<img src="images/example_sn_curve.svg" alt="S-N curve" width="80%">

Features: LCF/HCF/endurance regime bands, CAFL vertical line, knee point markers, standard name badge.

### Stress Along Path

Line plot of stress vs. distance with reference lines.

```python
from feaweld.visualization.plots_2d import plot_stress_along_path
fig = plot_stress_along_path(distances, stresses, labels={"Toe": 4.0}, show=False)
```

Features: Min/mean/max horizontal reference lines with value labels, optional labeled points.

### Weld Group Geometry

Draws weld group outlines for the Blodgett method with dimensions and centroid.

```python
from feaweld.visualization.plots_2d import plot_weld_group_geometry
fig = plot_weld_group_geometry(WeldGroupShape.BOX, d=100, b=60, props=props, show=False)
```

<img src="images/weld_groups_gallery.svg" alt="Weld group shapes" width="90%">

Supports 9 shapes: LINE, PARALLEL, C_SHAPE, L_SHAPE, BOX, CIRCULAR, I_SHAPE, T_SHAPE, U_SHAPE. Features: Dimension arrows, centroid with coordinates, section properties box.

### ASME VIII Stress Check

Horizontal bars comparing Pm, Pm+Pb, PL+Pb+Q against allowables.

```python
from feaweld.visualization.plots_2d import plot_asme_check
fig = plot_asme_check(categorization, S_m=160, S_y=275, show=False)
```

<img src="images/example_asme_check.svg" alt="ASME check" width="70%">

Features: Gradient utilization coloring (green-yellow-red), PASS/FAIL badges, limit equations next to each bar.

### Cross-Section Stress Contour

2D contour at a y-level slice through the mesh.

```python
from feaweld.visualization.plots_2d import plot_cross_section_stress
fig = plot_cross_section_stress(mesh, stress, y_level=5.0, show=False)
```

Features: Filled contour with labeled contour line overlay, uses perceptually uniform `turbo` colormap.

## 3D Plots (PyVista)

All 3D functions return a `pyvista.Plotter` and share:

```python
def plot_*(mesh, stress, component="von_mises", show=True, **kwargs) -> Plotter
```

### Stress Field

```python
from feaweld.visualization.stress_plots import plot_stress_field
plotter = plot_stress_field(mesh, stress, component="von_mises", show=False)
```

13 selectable components: `von_mises`, `tresca`, `xx`, `yy`, `zz`, `xy`, `yz`, `xz`, `principal_1`, `principal_2`, `principal_3`. Mesh edges shown automatically for meshes under 50k elements.

### Deformed Shape

```python
from feaweld.visualization.stress_plots import plot_deformed
plotter = plot_deformed(mesh, displacement, scale=10.0, stress=stress, show=False)
```

### Temperature Field

```python
from feaweld.visualization.stress_plots import plot_temperature_field
plotter = plot_temperature_field(mesh, temperature, show=False)
```

Uses `inferno` colormap (dark-to-bright, heat-intuitive).

### Enhanced 3D (feaweld.visualization.enhanced_3d)

| Function | Purpose |
|----------|---------|
| `plot_stress_with_clipping()` | Stress on a clipped cross-section |
| `plot_stress_threshold()` | Highlight regions above/below a threshold |
| `plot_iso_surface()` | Iso-surfaces of constant stress |
| `plot_force_vectors()` | Arrow glyphs for force/displacement vectors |
| `plot_weld_region_highlight()` | Highlight weld element group |
| `plot_sed_control_volume()` | SED averaging sphere visualization |
| `plot_mesh_preview()` | Mesh-only view with element/node set highlighting |
| `plot_annotated_stress()` | Stress contour with auto-detected critical points |

### Fatigue Maps (feaweld.visualization.fatigue_maps)

| Function | Purpose | Colormap |
|----------|---------|----------|
| `plot_fatigue_life()` | Fatigue life contour (log10 or linear) | `RdYlBu` (red=short, blue=long) |
| `plot_damage()` | Miner's cumulative damage | `YlOrRd` (yellow=safe, red=critical) |

Both include text annotations: life plot shows color interpretation guide; damage plot shows failure warning when D >= 1.0.

## Dashboards

Multi-panel composite views in `feaweld.visualization.dashboard`:

```python
from feaweld.visualization.dashboard import engineering_dashboard, fatigue_dashboard
fig = engineering_dashboard(workflow_result, show=False)   # 2x3 grid
fig = fatigue_dashboard(workflow_result, show=False)        # 2x2 grid
```

### Engineering Dashboard (2x3)

| Panel | Content |
|-------|---------|
| [1] Stress distribution | von Mises histogram with max/mean lines |
| [2] Linearization | Through-thickness profile or stress scatter |
| [3] S-N curve | With operating point from fatigue results |
| [4] Post-process method | Dong decomposition, hot-spot, or ASME check |
| [5] Weld geometry | Blodgett weld group or joint dimensions |
| [6] Summary text | Status, stresses, displacement, safety factor |

### Comparison Dashboard

```python
from feaweld.visualization.comparison import comparison_dashboard
fig = comparison_dashboard(study_results, show=False)
```

Auto-detects swept parameters and shows metric bars, sensitivity plot, stress envelope overlay, and summary.

## Annotations

The `feaweld.visualization.annotations` module provides critical point detection and severity-coded markers:

```python
from feaweld.visualization.annotations import find_critical_points, annotate_2d, annotate_3d

points = find_critical_points(mesh, stress, n_max=5, weld_line=weld_line)
annotate_2d(ax, points)       # for matplotlib
annotate_3d(plotter, points)  # for pyvista
```

Severity colors: green (#27ae60) = info, orange (#f39c12) = warning, red (#e74c3c) = critical.

## Theme Customization

The centralized theme module (`feaweld.visualization.theme`) provides:

```python
from feaweld.visualization.theme import get_cmap, apply_feaweld_style, configure_plotter

# Override a colormap for a specific plot
plotter.add_mesh(grid, cmap="coolwarm")  # pass cmap= to override default

# Available semantic colormaps
get_cmap("stress")        # "turbo"
get_cmap("temperature")   # "inferno"
get_cmap("fatigue_life")  # "RdYlBu"
get_cmap("damage")        # "YlOrRd"
get_cmap("diverging")     # "RdBu_r"
get_cmap("safety_factor") # "RdYlGn"
get_cmap("displacement")  # "viridis"
```

All 2D and 3D functions accept a `cmap` keyword to override the theme default.

## Report Integration

Figures are embedded as base64 PNG in HTML reports:

```python
from feaweld.visualization.report_figures import generate_report_figures, figure_to_base64

figures = generate_report_figures(workflow_result)  # dict of name -> base64 string
```

The report template automatically arranges individual plots in a 2-column grid with the engineering dashboard at full width.

## Export

```python
from feaweld.visualization.export import export_vtk, export_png, export_gltf

export_vtk(fea_results, "output.vtu")     # ParaView-compatible
export_png(plotter, "screenshot.png")      # raster screenshot
export_gltf(plotter, "model.gltf")        # web-ready 3D
```

## Regenerating Documentation Images

All diagrams and example outputs in this guide are generated programmatically.
The legacy round-0 conceptual SVGs (joint types, Blodgett shapes, S-N concept,
linearization, Dong, hotspot, SED, Goldak, pipeline overview) are produced by:

```bash
python scripts/generate_docs_images.py
```

The 23 advanced-concept images for the JAX backend, phase-field fracture,
multi-axial fatigue, J-integral, defects, spline paths, groove profiles, and
every other post-round-1/round-2 feature are produced by a separate script:

```bash
python scripts/generate_docs_concept_images.py
# or only a subset:
python scripts/generate_docs_concept_images.py --only jax_backend_flow bayesian_ensemble_uq
```

Both write SVG + PNG@300dpi pairs under `docs/images/`. Every figure calls
into live feaweld code, so the images stay consistent with the current API.

## Mermaid architecture diagrams

Eight mermaid diagrams describing the run_analysis pipeline, solver / constitutive
hierarchies, neural-operator training pipeline, active-learning and digital-twin
state machines, defect insertion workflow, and multi-pass welding sequence live
under `docs/diagrams/`. They render natively in GitHub Markdown and can also be
rendered offline with the `mmdc` CLI.

```bash
python scripts/generate_docs_mermaid.py
```

The `solver_backend_hierarchy.mmd` diagram is generated via live class
introspection (`SolverBackend.__subclasses__()`), so new backends appear
automatically on regeneration.

## Animations

Nine animations (GIF + MP4 pairs) illustrate time-evolving phenomena — phase-
field crack propagation, multi-pass thermal cycling, Goldak heat source sweep,
active-learning convergence, Monte Carlo convergence, EnKF crack tracking,
Bayesian posterior update, rainflow cycle counting, cyclic stress field.

```bash
python scripts/generate_docs_animations.py
# or only a subset:
python scripts/generate_docs_animations.py --only phase_field_crack_propagation
```

GIFs use palette quantization via pillow to stay under ~5 MB each. MP4s use
H.264 via `imageio-ffmpeg`'s bundled ffmpeg binary (no system ffmpeg required).
Every animation uses deterministic seeds; re-running the generator reproduces
the asset byte-for-byte.

## Master catalog

All 40 advanced-concept assets (8 mermaid + 23 images + 9 animations) are
indexed with descriptions and source references in
[`docs/CONCEPTS.md`](CONCEPTS.md). To rebuild everything in one command:

```bash
python scripts/build_docs_assets.py
```

This runs the legacy script, the mermaid generator, the concept-images generator,
and the animation generator in sequence and prints elapsed time per group.
