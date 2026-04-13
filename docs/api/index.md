# API Reference

This section documents the public Python API of feaweld, organized by module.

## Modules

| Module | Description |
|--------|-------------|
| [Core Types](core-types.md) | Shared dataclasses and enums (`FEMesh`, `FEAResults`, `StressField`, `LoadCase`, `JointType`, `StressMethod`, `SolverType`, etc.) used across the entire pipeline |
| [Materials](materials.md) | `Material` and `MaterialSet` with temperature-dependent properties, bundled material database access |
| [Pipeline](pipeline.md) | `AnalysisCase` pydantic configuration, `run_analysis()` orchestrator, `WorkflowResult` container |
| [Solver Backends](solver.md) | `SolverBackend` ABC and `get_backend()` factory for FEniCSx, CalculiX, JAX, and neural backends |
| [Post-Processing](postprocess.md) | All 8 stress extraction methods: hot-spot, Dong, nominal, linearization, notch stress, SED, Blodgett, and multi-axial criteria |
| [Fatigue](fatigue.md) | S-N curve evaluation, rainflow cycle counting, Palmgren-Miner cumulative damage, FAT classification lookup |

## Importing

All public symbols are accessible from their module paths:

```python
from feaweld.core.types import FEMesh, FEAResults, JointType, StressMethod
from feaweld.core.materials import Material, load_material
from feaweld.pipeline.workflow import AnalysisCase, run_analysis, load_case
from feaweld.solver.backend import SolverBackend, get_backend
from feaweld.fatigue.sn_curves import evaluate_sn_curve
```

Heavy optional modules should be imported inside functions to keep `import feaweld` fast and to allow the optional extras mechanism to work:

```python
# Do this inside a function, not at module top level
from feaweld.visualization.enhanced_3d import plot_stress_3d
from feaweld.digital_twin.daemon import TwinDaemon
```

## Conventions

- **Units are SI-mm**: lengths in mm, forces in N, stresses in MPa, moments in N-mm, temperatures in degrees C.
- **Pydantic for config, dataclasses for runtime**: YAML-loadable config objects are `pydantic.BaseModel`; in-memory result containers are `@dataclass`.
- **Errors are collected, not raised**: `run_analysis` catches per-stage exceptions and appends them to `WorkflowResult.errors` so partial results still come back.
