"""Analysis workflow orchestrator.

Binds all feaweld modules into end-to-end analysis cases defined via
YAML configuration or programmatic API. Manages the full pipeline:
geometry → mesh → solve → postprocess → visualize → report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import numpy as np
from numpy.typing import NDArray

from pydantic import BaseModel, Field

from feaweld.core.types import (
    JointType, SolverType, StressMethod, FEAResults, FEMesh,
)


class MaterialConfig(BaseModel):
    """Material specification in analysis case."""
    base_metal: str = "A36"
    weld_metal: str = "E70XX"
    haz: str = "A36"  # often same as base with modified properties
    temperature: float = 20.0  # ambient temperature (C)


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
    result = WorkflowResult(case=case)

    try:
        # Step 1: Materials
        from feaweld.core.materials import load_material, MaterialSet
        base = load_material(case.material.base_metal)
        weld = load_material(case.material.weld_metal)
        haz = load_material(case.material.haz)
        mat_set = MaterialSet(base_metal=base, weld_metal=weld, haz=haz)

        # Step 2: Geometry
        joint = _build_geometry(case.geometry)

        # Step 3: Mesh
        from feaweld.mesh.generator import generate_mesh, WeldMeshConfig
        mesh_config = WeldMeshConfig(
            global_size=case.mesh.global_size,
            weld_toe_size=case.mesh.weld_toe_size,
            element_order=case.mesh.element_order,
            element_type_2d=case.mesh.element_type,
        )
        mesh = generate_mesh(joint, mesh_config)
        result.mesh = mesh

        # Step 4: Solver
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

        # Step 5: Post-processing
        if fea_results.stress is not None:
            for method in case.postprocess.stress_methods:
                try:
                    pp_result = _run_postprocess(method, fea_results, mesh, case)
                    result.postprocess_results[method.value] = pp_result
                except Exception as e:
                    result.errors.append(f"Post-processing {method}: {e}")

        # Step 6: Fatigue
        if case.postprocess.fatigue_assessment and result.postprocess_results:
            try:
                fatigue = _run_fatigue_assessment(result.postprocess_results, case)
                result.fatigue_results = fatigue
            except Exception as e:
                result.errors.append(f"Fatigue assessment: {e}")

        # Step 7: Probabilistic
        if case.probabilistic.enabled:
            try:
                prob = _run_probabilistic(case, mat_set)
                result.probabilistic_results = prob
            except Exception as e:
                result.errors.append(f"Probabilistic: {e}")

    except Exception as e:
        result.errors.append(f"Workflow error: {e}")

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
