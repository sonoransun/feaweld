# pipeline

Workflow orchestration, parametric studies, and HTML report generation.

- `workflow` — `AnalysisCase` Pydantic model and the `run_analysis()` orchestrator.
- `study` — `Study` class, grid and one-at-a-time sweeps, `ProcessPoolExecutor` parallelism.
- `comparison` — multi-case delta tables and sensitivity plots.
- `report` — Jinja2 HTML report with embedded (static or Plotly-interactive) figures.

::: feaweld.pipeline
