"""Parametric study management and concurrent multi-pipeline execution.

Provides infrastructure for running multiple analysis cases concurrently
(varying materials, loads, mesh refinements, etc.) and collecting results
for comparison.
"""

from __future__ import annotations

import itertools
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, Field

from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult, run_analysis


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class ParameterSweep(BaseModel):
    """A single parameter to vary across cases."""
    name: str          # dot-path, e.g. "load.axial_force"
    values: list[Any]  # values to sweep


class StudyConfig(BaseModel):
    """YAML-loadable parametric study definition."""
    name: str = "study"
    description: str = ""
    base_case: AnalysisCase = Field(default_factory=AnalysisCase)
    parameters: list[ParameterSweep] = Field(default_factory=list)
    mode: Literal["grid", "one_at_a_time"] = "grid"
    max_workers: int = 4


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class StudyResults:
    """Container for all results from a parametric study."""
    study_name: str
    cases: dict[str, AnalysisCase] = field(default_factory=dict)
    results: dict[str, WorkflowResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def successful_results(self) -> dict[str, WorkflowResult]:
        return {k: v for k, v in self.results.items() if v.success}

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_succeeded(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return len(self.errors)

    @property
    def case_names(self) -> list[str]:
        return list(self.cases.keys())

    def __len__(self) -> int:
        return self.n_cases

    def __getitem__(self, key: str) -> WorkflowResult:
        return self.results[key]

    def __iter__(self):
        return iter(self.results.items())

    def __contains__(self, key: str) -> bool:
        return key in self.results


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_single_case(case: AnalysisCase) -> WorkflowResult:
    """Execute a single analysis case. Top-level function for pickling."""
    return run_analysis(case)


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

class Study:
    """Parametric study builder with fluent API and concurrent execution.

    Example::

        study = (
            Study("load_sweep", base_case)
            .vary("load.axial_force", [10000, 50000, 100000])
            .vary("mesh.global_size", [1.0, 2.0])
        )
        results = study.run(max_workers=4, mode="grid")
    """

    def __init__(self, name: str, base_case: AnalysisCase) -> None:
        self._name = name
        self._base_case = base_case
        self._sweeps: list[ParameterSweep] = []
        self._extra_cases: dict[str, AnalysisCase] = {}

    def vary(self, param_path: str, values: list[Any]) -> Study:
        """Add a parameter sweep. Returns self for chaining."""
        self._sweeps.append(ParameterSweep(name=param_path, values=values))
        return self

    def add_case(self, name: str, case: AnalysisCase) -> Study:
        """Add a specific named case (not from parameter sweep)."""
        self._extra_cases[name] = case
        return self

    def _generate_cases(self, mode: str = "grid") -> dict[str, AnalysisCase]:
        """Generate all analysis cases based on sweeps and mode."""
        if mode == "grid":
            cases = self._grid_cases()
        else:
            cases = self._oat_cases()

        # Add extra named cases
        cases.update(self._extra_cases)
        return cases

    def _grid_cases(self) -> dict[str, AnalysisCase]:
        """Factorial combination of all parameter sweeps."""
        if not self._sweeps:
            return {"baseline": self._base_case.model_copy(deep=True)}

        param_names = [s.name for s in self._sweeps]
        param_values = [s.values for s in self._sweeps]

        cases: dict[str, AnalysisCase] = {}
        for combo in itertools.product(*param_values):
            # Build descriptive name from short param names
            name_parts = []
            case = self._base_case.model_copy(deep=True)
            for param_path, value in zip(param_names, combo):
                short_name = param_path.rsplit(".", 1)[-1]
                name_parts.append(f"{short_name}={value}")
                case = _set_nested_attr(case, param_path, value)

            case_name = "_".join(name_parts)
            case.name = case_name
            cases[case_name] = case

        return cases

    def _oat_cases(self) -> dict[str, AnalysisCase]:
        """One-at-a-time: baseline plus one variant per parameter value."""
        cases: dict[str, AnalysisCase] = {}

        # Baseline
        baseline = self._base_case.model_copy(deep=True)
        baseline.name = "baseline"
        cases["baseline"] = baseline

        for sweep in self._sweeps:
            short_name = sweep.name.rsplit(".", 1)[-1]
            base_value = _get_nested_attr(self._base_case, sweep.name)

            for value in sweep.values:
                if value == base_value:
                    continue  # skip baseline duplicate
                case_name = f"{short_name}={value}"
                case = _set_nested_attr(
                    self._base_case.model_copy(deep=True), sweep.name, value
                )
                case.name = case_name
                cases[case_name] = case

        return cases

    def run(
        self,
        max_workers: int = 4,
        mode: str = "grid",
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> StudyResults:
        """Execute all cases concurrently.

        Args:
            max_workers: Number of parallel processes.
            mode: "grid" for factorial, "one_at_a_time" for OAT.
            progress_callback: Called with (case_name, completed, total).

        Returns:
            StudyResults containing all results and errors.
        """
        case_dict = self._generate_cases(mode)
        callback = progress_callback or _default_progress

        results: dict[str, WorkflowResult] = {}
        errors: dict[str, str] = {}
        start = time.time()
        total = len(case_dict)

        if total == 0:
            return StudyResults(study_name=self._name, elapsed_seconds=0.0)

        # Use max_workers=1 to run sequentially (useful for debugging)
        if max_workers <= 1:
            completed = 0
            for name, case in case_dict.items():
                completed += 1
                try:
                    results[name] = run_analysis(case)
                except Exception as e:
                    errors[name] = str(e)
                callback(name, completed, total)
        else:
            # Use 'spawn' context to avoid fork-safety issues with MPI
            # (FEniCSx uses MPI.COMM_WORLD internally).
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                future_to_name = {
                    executor.submit(_run_single_case, case): name
                    for name, case in case_dict.items()
                }
                completed = 0
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    completed += 1
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        errors[name] = str(e)
                    callback(name, completed, total)

        return StudyResults(
            study_name=self._name,
            cases=case_dict,
            results=results,
            errors=errors,
            elapsed_seconds=time.time() - start,
        )


# ---------------------------------------------------------------------------
# Nested attribute helpers
# ---------------------------------------------------------------------------

def _set_nested_attr(case: AnalysisCase, path: str, value: Any) -> AnalysisCase:
    """Set a deeply nested attribute on an AnalysisCase via dot-path.

    Example: _set_nested_attr(case, "load.axial_force", 50000)
    """
    parts = path.split(".")

    if len(parts) == 1:
        return case.model_copy(update={parts[0]: value})

    # Navigate: top-level field -> sub-model field
    top_key = parts[0]
    sub_path = ".".join(parts[1:])
    sub_model = getattr(case, top_key)

    if len(parts) == 2:
        updated_sub = sub_model.model_copy(update={parts[1]: value})
    else:
        # Deeper nesting (unlikely but supported)
        updated_sub = sub_model.model_copy(update={parts[1]: value})

    return case.model_copy(update={top_key: updated_sub})


def _get_nested_attr(case: AnalysisCase, path: str) -> Any:
    """Read a deeply nested attribute from an AnalysisCase via dot-path."""
    obj = case
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def _default_progress(case_name: str, completed: int, total: int) -> None:
    """Default progress callback: prints to stdout."""
    pct = 100 * completed // total
    print(f"  [{completed}/{total}] ({pct}%) Completed: {case_name}")


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_study(path: str | Path) -> StudyConfig:
    """Load a parametric study definition from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return StudyConfig(**data)


def save_study(config: StudyConfig, path: str | Path) -> None:
    """Save a parametric study definition to YAML."""
    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f, default_flow_style=False, sort_keys=False,
        )
