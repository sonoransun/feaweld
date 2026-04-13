"""Analysis workflow orchestrator.

Binds all feaweld modules into end-to-end analysis cases defined via
YAML configuration or programmatic API. Manages the full pipeline:
geometry → mesh → solve → postprocess → visualize → report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
import numpy as np
from numpy.typing import NDArray

from pydantic import BaseModel, Field

from feaweld.core.logging import get_logger
from feaweld.core.types import (
    JointType, SolverType, StressMethod, FEAResults, FEMesh, ElementType,
)

logger = get_logger(__name__)


class MaterialConfig(BaseModel):
    """Material specification in analysis case."""
    base_metal: str = "A36"
    weld_metal: str = "E70XX"
    haz: str = "A36"  # often same as base with modified properties
    temperature: float = 20.0  # ambient temperature (C)
    rve_path: str | None = None  # optional voxel RVE for FFT homogenization (.npz)


class GeometryConfig(BaseModel):
    """Joint geometry specification."""
    joint_type: JointType = JointType.FILLET_T
    base_width: float = 200.0
    base_thickness: float = 20.0
    web_height: float = 100.0
    web_thickness: float = 10.0
    weld_leg_size: float = 8.0
    length: float = 1.0  # extrusion depth (mm), 1.0 for quasi-2D


class MeshConfig(BaseModel):
    """Mesh generation configuration."""
    global_size: float = 2.0
    weld_toe_size: float = 0.2
    element_order: int = 2
    element_type: str = "tri"


class SolverConfig(BaseModel):
    """Solver configuration."""
    solver_type: SolverType = SolverType.LINEAR_ELASTIC
    backend: str = "auto"  # "auto", "fenics", "calculix"
    nonlinear: bool = False
    max_iterations: int = 50
    tolerance: float = 1e-8


class LoadConfig(BaseModel):
    """Loading specification."""
    axial_force: float = 0.0       # N
    bending_moment: float = 0.0    # N·mm
    shear_force: float = 0.0       # N
    pressure: float = 0.0          # MPa
    temperature_delta: float = 0.0 # C (for thermal stress)


class PostProcessConfig(BaseModel):
    """Post-processing configuration."""
    stress_methods: list[StressMethod] = Field(
        default_factory=lambda: [StressMethod.HOTSPOT_LINEAR]
    )
    sn_curve: str = "IIW_FAT90"
    fatigue_assessment: bool = True
    singularity_check: bool = True
    multiaxial_criterion: Literal[
        "none", "findley", "dang_van", "sines", "crossland",
        "fatemi_socie", "mcdiarmid",
    ] = "none"
    compute_k_factors: bool = False


class ThermalConfig(BaseModel):
    """Welding thermal simulation configuration."""
    enabled: bool = False
    voltage: float = 25.0
    current: float = 250.0
    travel_speed: float = 5.0
    efficiency: float = 0.8
    pwht_enabled: bool = False
    pwht_temperature: float = 620.0
    pwht_time_hours: float = 2.0


class ProbabilisticConfig(BaseModel):
    """Probabilistic analysis configuration."""
    enabled: bool = False
    n_samples: int = 1000
    method: str = "lhs"
    include_material_scatter: bool = True
    include_geometric_tolerance: bool = True


class DefectConfig(BaseModel):
    """Weld-defect population configuration.

    Controls ISO 5817 stochastic defect sampling or explicit defect
    definitions fed into the optional Track E/H defect pipeline. The
    defect hook runs post-geometry and populates
    ``WorkflowResult.extensions["defects"]``.
    """
    enabled: bool = False
    standard: str = "ISO 5817"
    quality_level: str = "B"
    weld_length: float = 100.0       # mm
    weld_width: float = 10.0         # mm
    population_seed: int = 0
    # If non-empty, use these explicit defect dicts instead of sampling.
    explicit_defects: list[dict] = Field(default_factory=list)


class MultiPassConfig(BaseModel):
    """Multi-pass welding sequence configuration.

    When enabled, the workflow loads a named
    :class:`feaweld.core.types.WeldSequence` JSON from the bundled
    ``multipass_sequences/`` data directory and records summary metadata
    on ``WorkflowResult.extensions["multipass"]``.
    """
    enabled: bool = False
    sequence_name: str | None = None  # e.g. "v_groove_3pass"
    preheat_temp: float = 20.0
    interpass_temp_max: float = 250.0


class WeldPathConfig(BaseModel):
    """Weld-path trajectory configuration (straight vs. spline)."""
    mode: Literal["straight", "spline"] = "straight"
    control_points: list[tuple[float, float, float]] = Field(default_factory=list)
    spline_mode: Literal["linear", "bspline", "catmull_rom"] = "bspline"
    spline_degree: int = 3


class AnalysisCase(BaseModel):
    """Complete analysis case definition."""
    name: str = "default"
    description: str = ""
    material: MaterialConfig = Field(default_factory=MaterialConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)
    defects: DefectConfig = Field(default_factory=DefectConfig)
    multipass: MultiPassConfig = Field(default_factory=MultiPassConfig)
    weld_path: WeldPathConfig = Field(default_factory=WeldPathConfig)
    output_dir: str = "results"


@dataclass
class WorkflowResult:
    """Complete results from an analysis workflow run."""
    case: AnalysisCase
    mesh: FEMesh | None = None
    fea_results: FEAResults | None = None
    postprocess_results: dict[str, Any] = field(default_factory=dict)
    fatigue_results: dict[str, Any] = field(default_factory=dict)
    probabilistic_results: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
    errors: list[str] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def load_case(path: str | Path) -> AnalysisCase:
    """Load analysis case from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return AnalysisCase(**data)


def save_case(case: AnalysisCase, path: str | Path) -> None:
    """Save analysis case to YAML file."""
    with open(path, "w") as f:
        yaml.dump(
            case.model_dump(mode="json"),
            f, default_flow_style=False, sort_keys=False,
        )


def run_analysis(case: AnalysisCase) -> WorkflowResult:
    """Execute a complete analysis workflow.

    Steps:
    1. Load materials
    2. Build geometry
    3. Generate mesh
    4. Run solver (with optional thermal)
    5. Post-process (stress methods, fatigue)
    6. Optional: probabilistic analysis
    7. Generate report

    Args:
        case: AnalysisCase configuration

    Returns:
        WorkflowResult with all results.
    """
    logger.info("Starting analysis: %s", case.name)
    result = WorkflowResult(case=case)

    # Step 0: Defect population (Track E/H) — runs outside the main
    # pipeline try block so it still populates extensions even if the
    # geometry/mesh/solver stages fail (e.g. missing Gmsh in sandbox).
    if case.defects.enabled:
        try:
            logger.info("Stage: defect population")
            _apply_defect_population(result, case)
        except Exception as e:  # pragma: no cover - best-effort hook
            result.errors.append(f"Defect population: {e}")

    try:
        # Step 1: Materials
        logger.info("Stage: materials")
        from feaweld.core.materials import load_material, MaterialSet
        base = load_material(case.material.base_metal)
        weld = load_material(case.material.weld_metal)
        haz = load_material(case.material.haz)
        mat_set = MaterialSet(base_metal=base, weld_metal=weld, haz=haz)
        logger.debug("Materials loaded: base=%s weld=%s haz=%s",
                      case.material.base_metal, case.material.weld_metal, case.material.haz)

        # Step 2: Geometry
        logger.info("Stage: geometry (%s)", case.geometry.joint_type.value)
        joint = _build_geometry(case.geometry)

        # Step 3: Mesh
        logger.info("Stage: mesh generation (global=%.2f, toe=%.2f)",
                      case.mesh.global_size, case.mesh.weld_toe_size)
        from feaweld.mesh.generator import generate_mesh, WeldMeshConfig
        mesh_config = WeldMeshConfig(
            global_size=case.mesh.global_size,
            weld_toe_size=case.mesh.weld_toe_size,
            element_order=case.mesh.element_order,
            element_type_2d=case.mesh.element_type,
        )
        mesh = generate_mesh(joint, mesh_config)
        result.mesh = mesh
        logger.debug("Mesh: %d nodes, %d elements", mesh.n_nodes, mesh.n_elements)

        # Step 4: Solver
        logger.info("Stage: solver (backend=%s, thermal=%s)",
                      case.solver.backend, case.thermal.enabled)
        from feaweld.solver.backend import get_backend
        from feaweld.core.types import LoadCase, BoundaryCondition, LoadType
        backend = get_backend(case.solver.backend)

        # Build load case
        load_case_obj = _build_load_case(case.load, mesh)

        if case.thermal.enabled:
            from feaweld.core.loads import WeldingHeatInput
            heat = WeldingHeatInput(
                voltage=case.thermal.voltage,
                current=case.thermal.current,
                travel_speed=case.thermal.travel_speed,
                efficiency=case.thermal.efficiency,
            )
            # Coupled solve
            time_steps = np.linspace(0, 100, 50)
            fea_results = backend.solve_coupled(
                mesh, base, load_case_obj, load_case_obj, time_steps
            )
        else:
            fea_results = backend.solve_static(
                mesh, base, load_case_obj,
                temperature=case.material.temperature,
            )

        result.fea_results = fea_results
        logger.info("Solver complete")

        # Step 4b: Multi-pass sequence metadata (Track G/H).
        if case.multipass.enabled:
            try:
                logger.info("Stage: multi-pass metadata")
                _apply_multipass_metadata(result, case)
            except Exception as e:  # pragma: no cover - best-effort hook
                result.errors.append(f"Multi-pass metadata: {e}")

        # Step 5: Post-processing
        if fea_results.stress is not None:
            for method in case.postprocess.stress_methods:
                try:
                    logger.info("Stage: post-processing (%s)", method.value)
                    pp_result = _run_postprocess(method, fea_results, mesh, case)
                    result.postprocess_results[method.value] = pp_result
                except Exception as e:
                    result.errors.append(f"Post-processing {method}: {e}")

        # Step 5b: Multi-axial fatigue criterion (Track F/H).
        if case.postprocess.multiaxial_criterion != "none":
            try:
                logger.info("Stage: multi-axial fatigue (%s)",
                             case.postprocess.multiaxial_criterion)
                _apply_multiaxial_criterion(result, case)
            except Exception as e:
                result.errors.append(f"Multi-axial fatigue: {e}")

        # Step 6: Fatigue
        if case.postprocess.fatigue_assessment and result.postprocess_results:
            try:
                logger.info("Stage: fatigue assessment")
                fatigue = _run_fatigue_assessment(result.postprocess_results, case)
                result.fatigue_results = fatigue
            except Exception as e:
                result.errors.append(f"Fatigue assessment: {e}")

            # UQ propagation: lognormal scatter around deterministic life.
            try:
                _apply_fatigue_uq(result, case)
            except Exception as e:
                result.errors.append(f"Fatigue UQ: {e}")

        # Step 6b: Optional J-integral / K-factor evaluation (Track F/H).
        if case.postprocess.compute_k_factors:
            try:
                logger.info("Stage: J-integral / K-factor")
                _apply_j_integral(result, case)
            except Exception as e:
                result.errors.append(f"J-integral: {e}")

        # Step 7: Probabilistic
        if case.probabilistic.enabled:
            try:
                logger.info("Stage: probabilistic (n=%d, method=%s)",
                             case.probabilistic.n_samples, case.probabilistic.method)
                prob = _run_probabilistic(case, mat_set)
                result.probabilistic_results = prob
            except Exception as e:
                result.errors.append(f"Probabilistic: {e}")

    except Exception as e:
        result.errors.append(f"Workflow error: {e}")

    if result.errors:
        logger.warning("Analysis %s completed with %d error(s)", case.name, len(result.errors))
    else:
        logger.info("Analysis %s completed successfully", case.name)

    return result


def _build_geometry(config: GeometryConfig):
    """Build joint geometry from config."""
    from feaweld.geometry.joints import (
        FilletTJoint, ButtWeld, LapJoint, CornerJoint, CruciformJoint,
    )

    builders = {
        JointType.FILLET_T: lambda: FilletTJoint(
            base_width=config.base_width,
            base_thickness=config.base_thickness,
            web_height=config.web_height,
            web_thickness=config.web_thickness,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
        ),
        JointType.BUTT: lambda: ButtWeld(
            plate_width=config.base_width,
            plate_thickness=config.base_thickness,
            length=config.length,
        ),
        JointType.LAP: lambda: LapJoint(
            plate_thickness=config.base_thickness,
            overlap_length=config.web_height,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
        ),
        JointType.CORNER: lambda: CornerJoint(
            plate_thickness_h=config.base_thickness,
            plate_thickness_v=config.web_thickness,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
        ),
        JointType.CRUCIFORM: lambda: CruciformJoint(
            plate_thickness=config.base_thickness,
            web_thickness=config.web_thickness,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
        ),
    }

    return builders[config.joint_type]()


def _build_load_case(config: LoadConfig, mesh: FEMesh):
    """Build LoadCase from config."""
    from feaweld.core.types import LoadCase, BoundaryCondition, LoadType

    loads = []
    constraints = []

    # Fixed bottom
    if "bottom" in mesh.node_sets:
        constraints.append(BoundaryCondition(
            node_set="bottom",
            bc_type=LoadType.DISPLACEMENT,
            values=np.array([0.0, 0.0, 0.0]),
        ))

    # Applied loads on top
    if config.axial_force != 0 and "top" in mesh.node_sets:
        loads.append(BoundaryCondition(
            node_set="top",
            bc_type=LoadType.FORCE,
            values=np.array([0.0, config.axial_force, 0.0]),
        ))

    return LoadCase(name="static", loads=loads, constraints=constraints)


def _run_postprocess(method, fea_results, mesh, case):
    """Run a single post-processing method."""
    from feaweld.core.types import WeldLineDefinition

    # Create a default weld line definition
    weld_line = WeldLineDefinition(
        name="weld_toe",
        node_ids=np.array(list(mesh.node_sets.get("weld_toe", [0]))),
        plate_thickness=case.geometry.base_thickness,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    if method.value == "hotspot_linear":
        from feaweld.postprocess.hotspot import hotspot_stress_linear, HotSpotType
        results = hotspot_stress_linear(fea_results, weld_line, HotSpotType.TYPE_A)
        return {"results": results, "max_stress": max(r.hot_spot_stress for r in results) if results else 0.0}

    elif method.value == "structural_dong":
        from feaweld.postprocess.dong import dong_structural_stress, dong_fatigue_life
        dong_result = dong_structural_stress(fea_results, weld_line)
        dong_result = dong_fatigue_life(dong_result, case.geometry.base_thickness)
        return {"dong_result": dong_result}

    elif method.value == "nominal":
        from feaweld.postprocess.nominal import categorize_stress_section
        return {"method": "nominal"}

    elif method.value == "blodgett":
        from feaweld.postprocess.blodgett import weld_group_properties, weld_stress
        from feaweld.core.types import WeldGroupShape
        props = weld_group_properties(WeldGroupShape.LINE, case.geometry.base_width)
        stress = weld_stress(props, case.geometry.weld_leg_size / np.sqrt(2),
                           P=case.load.axial_force, M=case.load.bending_moment)
        return {"properties": props, "stress": stress}

    return {"method": method.value}


def _run_fatigue_assessment(postprocess_results, case):
    """Run fatigue assessment on post-processing results."""
    from feaweld.fatigue.sn_curves import get_sn_curve

    # Parse SN curve spec
    sn_spec = case.postprocess.sn_curve
    if "_" in sn_spec:
        parts = sn_spec.split("_", 1)
        curve = get_sn_curve(parts[0].lower(), parts[1])
    else:
        curve = get_sn_curve("iiw", sn_spec)

    fatigue = {"sn_curve": sn_spec}

    for method, pp_result in postprocess_results.items():
        if "max_stress" in pp_result:
            stress = pp_result["max_stress"]
            life = curve.life(stress)
            fatigue[method] = {"stress_range": stress, "life": life}

    return fatigue


def _apply_fatigue_uq(
    result: WorkflowResult,
    case: AnalysisCase,
    scatter_std_log10_N: float = 0.2,
    n_mc: int = 2000,
    seed: int = 0,
) -> None:
    """Augment fatigue_results with lognormal UQ bands around each life."""
    from feaweld.fatigue.sn_curves import get_sn_curve, life_with_scatter_stress

    if not result.fatigue_results:
        return

    sn_spec = case.postprocess.sn_curve
    if "_" in sn_spec:
        parts = sn_spec.split("_", 1)
        curve = get_sn_curve(parts[0].lower(), parts[1])
    else:
        curve = get_sn_curve("iiw", sn_spec)

    for method, entry in result.fatigue_results.items():
        if not isinstance(entry, dict):
            continue
        if "stress_range" not in entry or "life" not in entry:
            continue
        stress_mean = float(entry["stress_range"])
        if stress_mean <= 0:
            continue
        stress_std = 0.05 * stress_mean
        band = life_with_scatter_stress(
            curve,
            stress_mean=stress_mean,
            stress_std=stress_std,
            scatter_std_log10_N=scatter_std_log10_N,
            n_samples=n_mc,
            seed=seed,
        )
        entry["mean"] = band["mean"]
        entry["std"] = band["std"]
        entry["p05"] = band["p05"]
        entry["p95"] = band["p95"]

    result.extensions["uq"] = {
        "scatter_std_log10_N": scatter_std_log10_N,
        "n_mc": n_mc,
    }


def _apply_defect_population(
    result: WorkflowResult, case: AnalysisCase
) -> None:
    """Sample (or pass through) a defect population and record it.

    MVP: no Gmsh-level insertion — we merely capture the list on
    ``result.extensions["defects"]`` so downstream consumers (reports,
    fatigue knockdown helpers, CLI) can inspect it.
    """
    defects: list = []
    if case.defects.explicit_defects:
        # Explicit dict form: we don't reconstruct concrete classes for the
        # MVP — just carry the dicts through so the population count is
        # non-zero and reports can render them verbatim.
        defects = list(case.defects.explicit_defects)
        result.extensions["defects"] = {
            "population": defects,
            "count": len(defects),
            "source": "explicit",
        }
        return

    from feaweld.defects.population import sample_iso5817_population

    level = case.defects.quality_level
    sampled = sample_iso5817_population(
        level=level,
        weld_length=case.defects.weld_length,
        weld_width=case.defects.weld_width,
        plate_thickness=case.geometry.base_thickness,
        seed=case.defects.population_seed,
    )
    result.extensions["defects"] = {
        "population": [d.description() for d in sampled],
        "count": len(sampled),
        "source": "sampled",
        "standard": case.defects.standard,
        "quality_level": level,
    }


def _apply_multipass_metadata(
    result: WorkflowResult, case: AnalysisCase
) -> None:
    """Load a named multipass sequence JSON and stash summary metadata."""
    sequence_name = case.multipass.sequence_name
    if not sequence_name:
        return

    from feaweld.data.registry import DataRegistry
    from feaweld.core.types import WeldPass, WeldSequence
    import json

    reg = DataRegistry()
    key = f"multipass_sequences/{sequence_name}"
    try:
        path = reg.get_dataset_path(key)
    except KeyError:
        # Graceful pass-through when the sequence isn't bundled.
        result.extensions["multipass"] = {
            "sequence_name": sequence_name,
            "n_passes": 0,
            "total_duration": 0.0,
            "status": "sequence_not_found",
        }
        return

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    passes_data = data.get("passes", [])
    passes = [WeldPass(**p) for p in passes_data]
    seq = WeldSequence(
        passes=passes,
        preheat_temp=case.multipass.preheat_temp,
        interpass_temp_max=case.multipass.interpass_temp_max,
    )
    result.extensions["multipass"] = {
        "sequence_name": sequence_name,
        "n_passes": len(seq.passes),
        "total_duration": seq.total_duration(),
    }


def _apply_multiaxial_criterion(
    result: WorkflowResult, case: AnalysisCase
) -> None:
    """Run a single multi-axial fatigue criterion over the FEA stress field."""
    fea = result.fea_results
    if fea is None or fea.stress is None:
        return

    from feaweld.postprocess import multiaxial as mx

    criterion_map = {
        "findley": mx.findley_criterion,
        "dang_van": mx.dang_van_criterion,
        "sines": mx.sines_criterion,
        "crossland": mx.crossland_criterion,
        "fatemi_socie": mx.fatemi_socie_criterion,
        "mcdiarmid": mx.mcdiarmid_criterion,
    }
    name = case.postprocess.multiaxial_criterion
    fn = criterion_map.get(name)
    if fn is None:
        return

    # Treat current stress field as a single-timestep history.  Aggregate by
    # picking the node with the largest damage-parameter contribution — for
    # MVP we iterate over nodes and keep the worst.
    values = np.asarray(fea.stress.values, dtype=float)  # (n_nodes, 6)
    best_damage = -np.inf
    best_normal = np.zeros(3)
    for i in range(values.shape[0]):
        history = values[i].reshape(1, 6)
        try:
            res = fn(history)
        except Exception:
            continue
        if res.damage_parameter > best_damage:
            best_damage = float(res.damage_parameter)
            best_normal = np.asarray(res.critical_plane_normal, dtype=float)
    if not np.isfinite(best_damage):
        return
    result.extensions["multiaxial_fatigue"] = {
        "criterion": name,
        "damage_parameter": best_damage,
        "critical_plane_normal": best_normal.tolist(),
    }


def _apply_j_integral(
    result: WorkflowResult, case: AnalysisCase
) -> None:
    """Run a 2D J-integral at the von-Mises peak as a crack-tip proxy."""
    fea = result.fea_results
    if fea is None or fea.stress is None:
        return
    mesh = fea.mesh
    if mesh is None or mesh.element_type != ElementType.TRI3:
        return

    from feaweld.fracture.j_integral import j_integral_2d

    vm = fea.stress.von_mises
    tip_idx = int(np.argmax(vm))
    tip = np.asarray(mesh.nodes[tip_idx, :2], dtype=float)

    j_result = j_integral_2d(
        fea_results=fea,
        crack_tip=tip,
        q_function_radius=2.0,
    )
    result.extensions["j_integral"] = {
        "J": float(j_result.J_value),
        "K_I": float(j_result.K_I),
    }


def _run_probabilistic(case, mat_set):
    """Run probabilistic analysis."""
    from feaweld.probabilistic.monte_carlo import MonteCarloEngine, MonteCarloConfig, RandomVariable
    from feaweld.probabilistic.distributions import material_property_distributions

    variables = material_property_distributions(case.material.base_metal)
    config = MonteCarloConfig(
        n_samples=case.probabilistic.n_samples,
        method=case.probabilistic.method,
    )
    engine = MonteCarloEngine(variables, config)

    def analysis_func(params):
        # Simplified deterministic analysis with varied parameters
        sigma_y = params.get("yield_strength", 250.0)
        stress = case.load.axial_force / (case.geometry.base_width * case.geometry.base_thickness)
        return sigma_y / max(abs(stress), 1e-12)

    result = engine.run(analysis_func)
    return {"mean": result.mean, "std": result.std, "cov": result.cov}
