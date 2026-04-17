# visualization

2D / 3D / interactive plotting and report figure generation.

- `theme` — semantic colormap registry (`stress`, `temperature`, `fatigue_life`, `damage`) and matplotlib/PyVista styling helpers.
- `plots_2d` — engineering 2D plots: through-thickness linearization, hot-spot extrapolation, S-N curve, Dong decomposition, convergence plots, weld-group geometry, ASME check, stress along path.
- `stress_plots`, `enhanced_3d` — 3D PyVista stress / temperature fields with clipping and iso-surfaces.
- `fatigue_plots` — rainflow histogram, S-N + damage stacked bars, damage-evolution animation.
- `thermal_plots` — Goldak heat-source iso-surface rendering and path animation.
- `fatigue_maps` — nodal fatigue-life and Miner-damage contour maps.
- `comparison` — multi-case comparison charts, sensitivity tornados, stress envelopes.
- `annotations` — critical-point detection and severity coding.
- `dashboard` — composite multi-panel engineering summaries.
- `report_figures`, `plotly_figures` — figure-to-base64 helpers and Plotly interactive mirrors for HTML reports.

::: feaweld.visualization
