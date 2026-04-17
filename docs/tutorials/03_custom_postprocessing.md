# Tutorial: Custom post-processing method

feaweld's post-processing is intentionally late-bound — each method is a standalone module function dispatched from `_run_postprocess()` in `pipeline/workflow.py`. Adding a new method is two small edits.

This tutorial adds a **maximum principal stress** extractor (σ<sub>1,max</sub>) that reports the peak principal stress and its location — useful as a simple sanity check alongside IIW hot-spot methods.

## Where the existing methods live

Every built-in method sits under `src/feaweld/postprocess/` and consumes `FEAResults` (solver-agnostic). The exemplar pattern is `postprocess/hotspot.py`:

```python
@dataclass
class HotSpotResult:
    hot_spot_stress: float
    reference_stresses: list[float]
    reference_distances: list[float]
    extrapolation_type: HotSpotType
    weld_toe_location: NDArray[np.float64]

def hotspot_stress_linear(results: FEAResults, weld_line: WeldLineDefinition,
                          hot_spot_type: HotSpotType = HotSpotType.TYPE_A
                          ) -> list[HotSpotResult]:
    ...
```

Three things make a method pluggable:

1. A **frozen result dataclass** with the fields downstream code needs.
2. A **function** that takes `FEAResults` (and method-specific config) and returns the result.
3. An **enum entry** in `StressMethod` (in `core/types.py`) plus a **dispatch entry** in `pipeline/workflow.py`.

## Step 1 — Write the result dataclass and function

Create `src/feaweld/postprocess/max_principal.py`:

```python
"""Maximum principal stress extractor.

Reports the peak principal stress (σ_1) across the mesh and its
spatial location. A fast sanity-check complementing the more
sophisticated IIW / Dong structural-stress methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults


@dataclass
class MaxPrincipalResult:
    """Output of ``max_principal_stress``.

    Attributes
    ----------
    sigma_1_max : float
        Peak tensile principal stress in MPa.
    node_id : int
        Mesh node index where the peak occurs.
    location : NDArray[np.float64]
        ``(x, y, z)`` coordinates of the peak node (mm).
    """
    sigma_1_max: float
    node_id: int
    location: NDArray[np.float64]


def max_principal_stress(results: FEAResults) -> MaxPrincipalResult:
    """Extract the peak tensile principal stress from an FEA result.

    Parameters
    ----------
    results
        Solved FEA results with a populated ``stress`` field.

    Returns
    -------
    MaxPrincipalResult
        Peak value, node index, and coordinates.

    Raises
    ------
    ValueError
        If ``results.stress`` is ``None``.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    sigma_1 = results.stress.principal_1
    idx = int(np.argmax(sigma_1))
    return MaxPrincipalResult(
        sigma_1_max=float(sigma_1[idx]),
        node_id=idx,
        location=results.mesh.nodes[idx],
    )
```

## Step 2 — Register the enum value

In `src/feaweld/core/types.py` find the `StressMethod` enum and add:

```python
class StressMethod(str, Enum):
    NOMINAL = "nominal"
    HOTSPOT_LINEAR = "hotspot_linear"
    HOTSPOT_QUADRATIC = "hotspot_quadratic"
    STRUCTURAL_DONG = "structural_dong"
    NOTCH_STRESS = "notch_stress"
    SED = "strain_energy_density"
    LINEARIZATION = "linearization"
    BLODGETT = "blodgett"
    MAX_PRINCIPAL = "max_principal"   # <-- new
```

## Step 3 — Wire it into the dispatcher

In `src/feaweld/pipeline/workflow.py` find `_run_postprocess()` and add a branch:

```python
elif method == StressMethod.MAX_PRINCIPAL:
    from feaweld.postprocess.max_principal import max_principal_stress
    return max_principal_stress(fea_results)
```

Note the **lazy import** — this matches the project convention (see `feaweld.pipeline.workflow` for the established pattern). It keeps the dispatcher cheap to import and avoids pulling heavy deps until a method is actually requested.

## Step 4 — Use it

Add `MAX_PRINCIPAL` to your `PostProcessConfig.stress_methods`:

```yaml
postprocess:
  stress_methods:
    - HOTSPOT_LINEAR
    - MAX_PRINCIPAL
  sn_curve: IIW_FAT90
```

Or programmatically:

```python
from feaweld.core.types import StressMethod

case.postprocess.stress_methods = [
    StressMethod.HOTSPOT_LINEAR,
    StressMethod.MAX_PRINCIPAL,
]
```

After `run_analysis(case)` you'll find the result under:

```python
result.postprocess_results["max_principal"]
# -> MaxPrincipalResult(sigma_1_max=312.4, node_id=4217, location=array([...]))
```

## Step 5 — Add a test

Shared fixtures live in `tests/conftest.py` (`simple_plate_mesh`, `uniform_stress_results`, `gradient_stress_results`). Create `tests/test_max_principal.py`:

```python
from feaweld.postprocess.max_principal import max_principal_stress


def test_max_principal_uniform(uniform_stress_results):
    result = max_principal_stress(uniform_stress_results)
    # Uniform 100 MPa σ_yy → σ_1 should be 100 everywhere.
    assert 99.0 < result.sigma_1_max < 101.0


def test_max_principal_gradient(gradient_stress_results):
    result = max_principal_stress(gradient_stress_results)
    # Peak should be at one extreme of the gradient.
    assert result.sigma_1_max > 0
    assert result.node_id >= 0
```

Run:

```bash
pytest tests/test_max_principal.py -v
```

## Optional — plot it

To make the new method appear in HTML reports, add a small plotter in `src/feaweld/visualization/plots_2d.py` (or `report_figures.py` if you want base64-embedded output) that consumes `MaxPrincipalResult` and follows the existing `plot_hotspot_extrapolation` signature pattern. Then extend `pipeline/report.py` to call it when `max_principal` is present in `postprocess_results`.

## Why it's this easy

The key design choice is `FEAResults` being solver-agnostic and frozen — every post-processor consumes the same input regardless of whether FEniCSx or CalculiX produced it, and no post-processor can mutate state that another relies on. This means:

- **Methods fail independently** — if `max_principal_stress` raises, hot-spot still runs.
- **No inheritance.** A method is a function; it cannot override behavior in a way that surprises callers.
- **Adding a method never modifies the orchestrator core** — only the `StressMethod` enum and one dispatch branch.

Apply the same pattern when integrating domain-specific stress extractors (e.g. a code-compliance check for AWS D1.8, a proprietary in-house method, or an ML-based surrogate).
