# core

Shared data types, material properties, and load definitions.

- `types` — frozen dataclasses and Pydantic models that every sub-package consumes (`FEMesh`, `StressField`, `FEAResults`, enums like `JointType`, `SolverType`, `StressMethod`).
- `materials` — temperature-dependent material database with `scipy.interpolate` lazy caching.
- `loads` — mechanical, thermal, and weld heat-input load builders.

::: feaweld.core
