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
    time_end: float = 100.0        # s, transient/coupled analyses
    n_time_steps: int = 50
    creep_temperature: float = 550.0  # C, hold temperature for SolverType.CREEP
    creep_time_hours: float = 10.0


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
    singularity_threshold: float = 0.20   # fractional stress increase flag level
    singularity_coarsening: float = 2.0   # coarse-mesh size factor for the check solve
    notch_radius: float = 1.0             # mm, IIW fictitious radius (steel, t >= 5 mm)
    notch_nominal_stress: float | None = None  # MPa; None -> computed from load/geometry
    sed_control_radius: float = 0.28      # mm, R0 for structural steel
    sed_w_ref: float | None = None        # MJ/m^3 at N_ref; None -> skip SED life estimate
    linearization_points: int = 20


class ThermalConfig(BaseModel):
    """Welding thermal simulation configuration."""
    enabled: bool = False
    voltage: float = 25.0
    current: float = 250.0
    travel_speed: float = 5.0
    efficiency: float = 0.8
    ambient_temperature: float = 20.0
    film_coefficient: float = 15.0        # W/(m^2 K), convection to ambient
    pwht_enabled: bool = False
    pwht_temperature: float = 620.0
    pwht_time_hours: float = 2.0
    pwht_heating_rate: float = 55.0       # C/hour
    pwht_cooling_rate: float = 55.0       # C/hour


class ProbabilisticConfig(BaseModel):
    """Probabilistic analysis configuration."""
    enabled: bool = False
    n_samples: int = 1000
    method: str = "lhs"
    include_material_scatter: bool = True
    include_geometric_tolerance: bool = True
    seed: int | None = None
    sobol: bool = False
    sobol_n_base: int = 256


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
    warnings: list[str] = field(default_factory=list)

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

    Parameters
    ----------
    case : AnalysisCase
        AnalysisCase configuration

    Returns
    -------
    WorkflowResult
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
        backend = get_backend(case.solver.backend)

        # Build load case
        load_case_obj = _build_load_case(case.load, mesh)

        fea_results = _run_solver(case, backend, mesh, base, load_case_obj, result)

        # Step 4b: PWHT stress relaxation on the solved field
        if case.thermal.pwht_enabled and fea_results.stress is not None:
            from feaweld.core.loads import PWHTSchedule
            from feaweld.solver.creep import simulate_pwht
            if not case.thermal.enabled:
                result.warnings.append(
                    "PWHT relaxation applied to a load-stress field, not a "
                    "residual-stress field (thermal.enabled is false)."
                )
            schedule = PWHTSchedule(
                heating_rate=case.thermal.pwht_heating_rate,
                holding_temperature=case.thermal.pwht_temperature,
                holding_time=case.thermal.pwht_time_hours,
                cooling_rate=case.thermal.pwht_cooling_rate,
            )
            as_welded_vm_max = float(np.max(fea_results.stress.von_mises))
            fea_results = simulate_pwht(fea_results, base, schedule)
            fea_results.metadata["as_welded_max_von_mises"] = as_welded_vm_max

        result.fea_results = fea_results

        # Step 5: Post-processing
        if fea_results.stress is not None:
            if "weld_toe" not in mesh.node_sets:
                # _run_postprocess still falls back to node 0 for API
                # compatibility, but that node is the clamped corner, not the
                # weld toe — surface it so a real run is never silently wrong.
                result.warnings.append(
                    "no weld_toe node set on the mesh; stress methods "
                    "evaluated at node 0"
                )
            for method in case.postprocess.stress_methods:
                try:
                    pp_result = _run_postprocess(method, fea_results, mesh, case, base)
                    result.postprocess_results[method.value] = pp_result
                except Exception as e:
                    result.errors.append(f"Post-processing {method}: {e}")

        # Step 5b: Mesh-sensitivity / singularity check (best-effort).
        # The check compares the fine solve against a coarse *linear elastic*
        # re-solve, so it is only valid when the effective fine solve was also
        # plain linear elastic.  Plasticity capping (ELASTOPLASTIC / nonlinear)
        # or PWHT relaxation would make the coarse baseline incomparable and
        # produce false negatives, so guard those out.
        if (
            case.postprocess.singularity_check
            and fea_results.stress is not None
            and case.solver.solver_type == SolverType.LINEAR_ELASTIC
            and not case.solver.nonlinear
            and not case.thermal.enabled
            and not case.thermal.pwht_enabled
        ):
            try:
                check = _run_singularity_check(case, backend, joint, fea_results, base)
                fea_results.metadata["singularity_check"] = check
                result.postprocess_results["singularity_check"] = check
                if check["n_singular"] > 0:
                    result.warnings.append(
                        f"Singularity check flagged {check['n_singular']} node(s) "
                        "with non-converging stress (likely mesh-driven peaks at "
                        "re-entrant corners); consider notch rounding or SED/hot-spot "
                        "methods instead of raw peak stress."
                    )
            except Exception as e:
                result.warnings.append(f"Singularity check skipped: {e}")

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
    """Build a LoadCase from the YAML load block.

    Loads act on the "top" node set with reactions at the fixed "bottom"
    set. Force-type loads are emitted per node so backends apply exact
    totals: ``values=[magnitude_per_node]`` with a unit ``direction``.
    """
    from feaweld.core.types import LoadCase, BoundaryCondition, LoadType
    from feaweld.core.loads import moment_to_nodal_forces

    loads = []
    constraints = []

    # Fixed bottom
    if "bottom" in mesh.node_sets:
        constraints.append(BoundaryCondition(
            node_set="bottom",
            bc_type=LoadType.DISPLACEMENT,
            values=np.array([0.0, 0.0, 0.0]),
        ))

    has_top = "top" in mesh.node_sets
    n_top = len(mesh.node_sets["top"]) if has_top else 0

    if config.axial_force != 0 and n_top:
        loads.append(BoundaryCondition(
            node_set="top",
            bc_type=LoadType.FORCE,
            values=np.array([config.axial_force / n_top]),
            direction=np.array([0.0, 1.0, 0.0]),
        ))

    if config.shear_force != 0 and n_top:
        loads.append(BoundaryCondition(
            node_set="top",
            bc_type=LoadType.FORCE,
            values=np.array([config.shear_force / n_top]),
            direction=np.array([1.0, 0.0, 0.0]),
        ))

    if config.bending_moment != 0 and n_top:
        loads.append(moment_to_nodal_forces(mesh, "top", config.bending_moment))

    if config.pressure != 0:
        loads.append(BoundaryCondition(
            node_set="top" if has_top else "all",
            bc_type=LoadType.PRESSURE,
            values=np.array([config.pressure]),
        ))

    if config.temperature_delta != 0:
        # Stored as absolute temperature; backends use a 20 C stress-free
        # reference, so thermal strain is alpha * temperature_delta.
        loads.append(BoundaryCondition(
            node_set="all",
            bc_type=LoadType.TEMPERATURE,
            values=np.array([20.0 + config.temperature_delta]),
        ))

    return LoadCase(name="static", loads=loads, constraints=constraints)


def _build_thermal_load_case(case: AnalysisCase, mesh: FEMesh):
    """Thermal BCs for welding simulations: ambient sink plus convection."""
    from feaweld.core.types import LoadCase, BoundaryCondition, LoadType

    constraints = []
    if "bottom" in mesh.node_sets:
        constraints.append(BoundaryCondition(
            node_set="bottom",
            bc_type=LoadType.TEMPERATURE,
            values=np.array([case.thermal.ambient_temperature]),
        ))

    loads = [BoundaryCondition(
        node_set="all",
        bc_type=LoadType.CONVECTION,
        values=np.array([
            case.thermal.film_coefficient,
            case.thermal.ambient_temperature,
        ]),
    )]

    return LoadCase(name="thermal", loads=loads, constraints=constraints)


def _build_heat_source(case: AnalysisCase, mesh: FEMesh):
    """Goldak double-ellipsoid heat source from the welding parameters."""
    from feaweld.core.loads import WeldingHeatInput
    from feaweld.solver.thermal import GoldakHeatSource

    heat = WeldingHeatInput(
        voltage=case.thermal.voltage,
        current=case.thermal.current,
        travel_speed=case.thermal.travel_speed,
        efficiency=case.thermal.efficiency,
    )

    toe_ids = mesh.node_sets.get("weld_toe")
    if toe_ids is not None and len(toe_ids):
        start = mesh.nodes[toe_ids].mean(axis=0)
    else:
        start = mesh.nodes.mean(axis=0)
    start = np.asarray(start, dtype=np.float64)
    if start.shape[0] == 2:
        start = np.append(start, 0.0)

    return GoldakHeatSource(
        power=heat.power,
        a_f=heat.a_f, a_r=heat.a_r, b=heat.b, c=heat.c,
        f_f=heat.f_f, f_r=heat.f_r,
        travel_speed=heat.travel_speed,
        start_position=start,
        direction=np.array([1.0, 0.0, 0.0]),
    )


def _run_solver(case, backend, mesh, material, load_case_obj, result):
    """Dispatch the solve on ``SolverType`` (honouring thermal coupling)."""
    solver_cfg = case.solver
    solver_type = solver_cfg.solver_type
    time_steps = np.linspace(0.0, solver_cfg.time_end, solver_cfg.n_time_steps)

    # Back-compat: a welding thermal pass with the default solver type is a
    # sequentially-coupled thermomechanical analysis.
    if case.thermal.enabled and solver_type == SolverType.LINEAR_ELASTIC:
        solver_type = SolverType.THERMOMECHANICAL
    elif solver_type == SolverType.LINEAR_ELASTIC and solver_cfg.nonlinear:
        solver_type = SolverType.ELASTOPLASTIC

    if solver_type == SolverType.LINEAR_ELASTIC:
        return backend.solve_static(
            mesh, material, load_case_obj,
            temperature=case.material.temperature,
        )

    if solver_type == SolverType.ELASTOPLASTIC:
        from feaweld.solver.mechanical import solve_elastoplastic
        return solve_elastoplastic(
            backend, mesh, material, load_case_obj,
            temperature=case.material.temperature,
            max_iterations=solver_cfg.max_iterations,
            tolerance=solver_cfg.tolerance,
        )

    if solver_type == SolverType.THERMAL_STEADY:
        return backend.solve_thermal_steady(
            mesh, material, _build_thermal_load_case(case, mesh),
        )

    if solver_type == SolverType.THERMAL_TRANSIENT:
        heat_source = _build_heat_source(case, mesh) if case.thermal.enabled else None
        return backend.solve_thermal_transient(
            mesh, material, _build_thermal_load_case(case, mesh),
            time_steps, heat_source=heat_source,
        )

    if solver_type == SolverType.THERMOMECHANICAL:
        from feaweld.solver.thermomechanical import sequential_coupled_solve
        heat_source = _build_heat_source(case, mesh) if case.thermal.enabled else None
        return sequential_coupled_solve(
            backend=backend,
            mesh=mesh,
            material=material,
            thermal_lc=_build_thermal_load_case(case, mesh),
            mechanical_lc=load_case_obj,
            time_steps=time_steps,
            heat_source=heat_source,
        )

    if solver_type == SolverType.CREEP:
        from feaweld.solver.creep import simulate_creep_relaxation
        static = backend.solve_static(
            mesh, material, load_case_obj,
            temperature=case.material.temperature,
        )
        if material.creep_A <= 0.0:
            result.warnings.append(
                f"Material '{material.name}' has no creep parameters "
                "(creep_A == 0); creep relaxation is a no-op."
            )
        return simulate_creep_relaxation(
            static, material,
            temperature=solver_cfg.creep_temperature,
            duration_hours=solver_cfg.creep_time_hours,
        )

    raise ValueError(f"Unhandled solver type: {solver_type}")


def _run_singularity_check(case, backend, joint, fine_results, material):
    """Re-solve on a coarser mesh and flag non-converging stress peaks."""
    from feaweld.mesh.generator import generate_mesh, WeldMeshConfig
    from feaweld.singularity.detection import detect_singularities

    k = case.postprocess.singularity_coarsening
    coarse_cfg = WeldMeshConfig(
        global_size=case.mesh.global_size * k,
        weld_toe_size=case.mesh.weld_toe_size * k,
        element_order=case.mesh.element_order,
        element_type_2d=case.mesh.element_type,
    )
    coarse_mesh = generate_mesh(joint, coarse_cfg)
    coarse_results = backend.solve_static(
        coarse_mesh, material,
        _build_load_case(case.load, coarse_mesh),
        temperature=case.material.temperature,
    )
    detections = detect_singularities(
        coarse_results, fine_results,
        threshold=case.postprocess.singularity_threshold,
    )
    singular = [d for d in detections if d.is_singular]
    return {
        "n_flagged": len(detections),
        "n_singular": len(singular),
        "singular_node_ids": [int(d.node_id) for d in singular[:50]],
        "min_convergence_rate": (
            float(min(d.convergence_rate for d in detections)) if detections else None
        ),
        "threshold": case.postprocess.singularity_threshold,
    }


def _run_postprocess(method, fea_results, mesh, case, material):
    """Run a single post-processing method (one branch per StressMethod)."""
    from feaweld.core.types import WeldLineDefinition

    # Create a default weld line definition
    weld_line = WeldLineDefinition(
        name="weld_toe",
        node_ids=np.array(list(mesh.node_sets.get("weld_toe", [0]))),
        plate_thickness=case.geometry.base_thickness,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    # Critical weld-line node: highest von Mises along the toe
    vm = fea_results.stress.von_mises
    toe_ids = weld_line.node_ids[weld_line.node_ids < vm.shape[0]]
    if len(toe_ids):
        toe_node = int(toe_ids[np.argmax(vm[toe_ids])])
    else:
        toe_ids = np.array([0])
        toe_node = 0

    if method.value == "hotspot_linear":
        from feaweld.postprocess.hotspot import hotspot_stress_linear, HotSpotType
        results = hotspot_stress_linear(fea_results, weld_line, HotSpotType.TYPE_A)
        return {"results": results, "max_stress": max(r.hot_spot_stress for r in results) if results else 0.0}

    elif method.value == "hotspot_quadratic":
        from feaweld.postprocess.hotspot import hotspot_stress_quadratic
        results = hotspot_stress_quadratic(fea_results, weld_line)
        return {"results": results, "max_stress": max(r.hot_spot_stress for r in results) if results else 0.0}

    elif method.value == "structural_dong":
        from feaweld.postprocess.dong import dong_structural_stress, dong_fatigue_life
        dong_result = dong_structural_stress(fea_results, weld_line)
        dong_result = dong_fatigue_life(dong_result, case.geometry.base_thickness)
        out = {
            "dong_result": dong_result,
            "max_stress": float(np.max(dong_result.structural_stress)),
        }
        if dong_result.fatigue_life is not None and len(dong_result.fatigue_life):
            out["fatigue_life"] = float(np.min(dong_result.fatigue_life))
            out["sn_curve_used"] = "ASME master S-N curve (Dong)"
        return out

    elif method.value == "notch_stress":
        # Parametric effective-notch estimate per IIW.  A *true* effective-notch
        # analysis needs a fictitiously-rounded FE model
        # (feaweld.geometry.notch.create_notched_model); the raw singular corner
        # peak of an un-rounded mesh is mesh-size dependent and not comparable
        # to FAT225.  Instead we combine the mesh-insensitive structural stress
        # at the toe (through-thickness linearization) with a parametric
        # weld-toe SCF K_t, then read life off FAT225.
        from feaweld.postprocess.linearization import linearize_at_weld_toe
        from feaweld.postprocess.notch_stress import (
            FAT225_CURVE, NotchStressResult, notch_stress_scf_parametric,
        )
        t = case.geometry.base_thickness
        lin = linearize_at_weld_toe(
            fea_results, toe_node,
            plate_thickness=t,
            surface_normal=weld_line.normal_direction,
            n_points=case.postprocess.linearization_points,
        )
        structural_stress = lin.membrane_plus_bending_scalar

        is_butt = case.geometry.joint_type == JointType.BUTT
        K_t = notch_stress_scf_parametric(
            toe_radius=case.postprocess.notch_radius,
            toe_angle=30.0 if is_butt else 45.0,
            plate_thickness=t,
            weld_toe_type="butt" if is_butt else "fillet",
        )

        nominal = case.postprocess.notch_nominal_stress
        reference_stress = (
            nominal if (nominal is not None and nominal > 0) else structural_stress
        )
        notch_range = K_t * reference_stress
        notch = NotchStressResult(
            max_notch_stress=notch_range,
            notch_stress_range=notch_range,
            stress_concentration_factor=K_t,
            fatigue_life=FAT225_CURVE.life(notch_range),
            notch_stress_field=None,
            fictitious_radius=case.postprocess.notch_radius,
        )
        return {
            "notch_result": notch,
            "max_stress": notch.notch_stress_range,
            "scf": notch.stress_concentration_factor,
            "fatigue_life": notch.fatigue_life,
            "sn_curve_used": "IIW FAT225 (effective notch)",
        }

    elif method.value == "strain_energy_density":
        from feaweld.postprocess.sed import averaged_sed, sed_fatigue_life
        T = case.material.temperature
        sed_result = averaged_sed(
            fea_results,
            center_point=mesh.nodes[toe_node],
            control_radius=case.postprocess.sed_control_radius,
            elastic_modulus=material.E(T),
            poisson_ratio=material.nu(T),
        )
        out = {
            "sed_result": sed_result,
            "averaged_sed": sed_result.averaged_sed,
            "control_radius": sed_result.control_radius,
        }
        if case.postprocess.sed_w_ref:
            sed_result = sed_fatigue_life(sed_result, W_ref=case.postprocess.sed_w_ref)
            out["sed_result"] = sed_result
            out["fatigue_life"] = sed_result.fatigue_life
            out["sn_curve_used"] = "SED power law (Lazzarin)"
        return out

    elif method.value == "linearization":
        from feaweld.postprocess.linearization import linearize_at_weld_toe
        lin = linearize_at_weld_toe(
            fea_results, toe_node,
            plate_thickness=case.geometry.base_thickness,
            surface_normal=weld_line.normal_direction,
            n_points=case.postprocess.linearization_points,
        )
        return {
            "linearization": lin,
            "membrane": lin.membrane_scalar,
            "bending": lin.bending_scalar,
            "peak": lin.peak_scalar,
            "max_stress": lin.membrane_plus_bending_scalar,
        }

    elif method.value == "nominal":
        from feaweld.postprocess.nominal import (
            asme_allowable_check, categorize_stress_section,
            extract_stress_along_path,
        )
        from scipy.spatial import cKDTree
        t = case.geometry.base_thickness
        inner_target = mesh.nodes[toe_node] - t * weld_line.normal_direction
        _, start_node = cKDTree(mesh.nodes).query(inner_target)
        distances, sigmas = extract_stress_along_path(
            fea_results, int(start_node), toe_node,
            n_points=case.postprocess.linearization_points,
        )
        cat = categorize_stress_section(sigmas, thickness=t, z_coords=distances)
        T = case.material.temperature
        S_y = material.sigma_y(T)
        try:
            S_m = min(material.sigma_u(T) / 2.4, S_y / 1.5)
        except (KeyError, ValueError):
            S_m = S_y / 1.5
        checks = asme_allowable_check(cat, S_m=S_m, S_y=S_y)
        return {
            "categorization": cat,
            "membrane": cat.membrane,
            "bending": cat.bending,
            "asme_checks": checks,
            "max_stress": cat.primary_plus_bending,
        }

    elif method.value == "blodgett":
        from feaweld.postprocess.blodgett import weld_group_properties, weld_stress
        from feaweld.core.types import WeldGroupShape
        props = weld_group_properties(WeldGroupShape.LINE, case.geometry.base_width)
        stress = weld_stress(props, case.geometry.weld_leg_size / np.sqrt(2),
                           P=case.load.axial_force, M=case.load.bending_moment)
        return {"properties": props, "stress": stress}

    # Defensive fallthrough: every StressMethod should be dispatched above.
    return {
        "method": method.value,
        "warning": f"No dispatch entry for stress method '{method.value}'",
    }


def _run_fatigue_assessment(postprocess_results, case):
    """Run fatigue assessment on post-processing results."""
    from feaweld.fatigue.sn_curves import parse_sn_spec

    sn_spec = case.postprocess.sn_curve
    curve = parse_sn_spec(sn_spec)

    fatigue = {"sn_curve": sn_spec}

    for method, pp_result in postprocess_results.items():
        if not isinstance(pp_result, dict):
            continue
        if pp_result.get("fatigue_life") is not None:
            # Methods with a mandated curve (notch/FAT225, Dong master curve,
            # SED power law) carry their own life estimate.
            fatigue[method] = {
                "stress_range": pp_result.get("max_stress"),
                "life": float(pp_result["fatigue_life"]),
                "sn_curve": pp_result.get("sn_curve_used", sn_spec),
            }
        elif "max_stress" in pp_result:
            stress = pp_result["max_stress"]
            fatigue[method] = {"stress_range": stress, "life": curve.life(stress)}

    return fatigue


def build_probabilistic_model(case: AnalysisCase):
    """Build the random variables and response function for a case.

    The response is the safety factor against yield using a closed-form
    structural model.  The nominal stress splits into an axial/membrane part
    and a bending part; axial misalignment (``k_m = 1 + 3e/t``) magnifies only
    the membrane part, and a parametric weld-toe SCF scales the total when
    geometric scatter is enabled.  Material scatter distributions are
    re-centred on the loaded base-metal properties at the case temperature.

    Returns
    -------
    tuple[list[RandomVariable], Callable[[dict[str, float]], float]]
        ``(variables, analysis_func)`` for Monte Carlo, Sobol, or FORM.
    """
    from feaweld.core.materials import load_material
    from feaweld.probabilistic.distributions import (
        geometric_tolerance_distributions, material_property_distributions,
    )

    material = load_material(case.material.base_metal)
    T = case.material.temperature

    def _nominal(getter):
        try:
            return getter(T)
        except (KeyError, ValueError):
            return None

    cfg = case.probabilistic
    variables = []
    if cfg.include_material_scatter:
        # Re-centre the generic scatter model on this material's actual
        # properties so the sampled yield/UTS/E reflect the chosen base metal.
        variables.extend(material_property_distributions(
            case.material.base_metal,
            yield_nominal=_nominal(material.sigma_y),
            uts_nominal=_nominal(material.sigma_u),
            modulus_nominal=_nominal(material.E),
        ))
    if cfg.include_geometric_tolerance:
        weld_type = "butt" if case.geometry.joint_type == JointType.BUTT else "fillet"
        variables.extend(geometric_tolerance_distributions(weld_type))
    if not variables:
        raise ValueError(
            "Probabilistic analysis needs include_material_scatter and/or "
            "include_geometric_tolerance enabled."
        )

    sigma_y_nominal = material.sigma_y(T)

    w = case.geometry.base_width
    t = case.geometry.base_thickness
    sigma_axial = abs(case.load.axial_force) / (w * t)
    sigma_bending = 6.0 * abs(case.load.bending_moment) / (w * t ** 2)

    def analysis_func(params: dict[str, float]) -> float:
        sigma_y = params.get("yield_strength", sigma_y_nominal)
        # Axial misalignment magnifies only the membrane/axial part
        # (k_m = 1 + 3 e / t); the bending part is unaffected.
        e = abs(params.get("misalignment", 0.0))
        k_m = 1.0 + 3.0 * e / t
        stress = sigma_axial * k_m + sigma_bending
        # Local toe geometry drives the stress concentration on the total.
        if "weld_toe_radius" in params or "weld_toe_angle" in params:
            from feaweld.postprocess.notch_stress import notch_stress_scf_parametric
            scf = notch_stress_scf_parametric(
                toe_radius=max(params.get("weld_toe_radius", 1.0), 0.05),
                toe_angle=params.get("weld_toe_angle", 45.0),
                plate_thickness=t,
            )
            stress *= scf
        return sigma_y / max(abs(stress), 1e-12)

    return variables, analysis_func


def run_probabilistic_case(case: AnalysisCase) -> dict[str, Any]:
    """Monte Carlo (and optional Sobol) analysis for an analysis case.

    Uses a closed-form structural response (see
    [build_probabilistic_model][feaweld.pipeline.workflow.build_probabilistic_model]) rather than re-running FEA per
    sample, so thousand-sample studies stay fast. The response is the
    safety factor against yield.
    """
    from feaweld.probabilistic.monte_carlo import MonteCarloConfig, MonteCarloEngine

    cfg = case.probabilistic
    variables, analysis_func = build_probabilistic_model(case)

    engine = MonteCarloEngine(variables, MonteCarloConfig(
        n_samples=cfg.n_samples, method=cfg.method, seed=cfg.seed,
    ))
    mc = engine.run(analysis_func)

    payload: dict[str, Any] = {
        "mean": mc.mean,
        "std": mc.std,
        "cov": mc.cov,
        "percentiles": mc.percentiles,
        "converged": mc.converged,
        "n_effective": mc.n_effective,
        "samples": mc.samples,
        "results": mc.results,
        "variable_names": [v.name for v in variables],
        "response": "safety_factor",
    }

    if cfg.sobol:
        from feaweld.probabilistic.sensitivity import sobol_indices
        payload["sobol"] = sobol_indices(
            variables, analysis_func, n_base=cfg.sobol_n_base, seed=cfg.seed,
        )

    return payload


def _run_probabilistic(case, mat_set):
    """Run probabilistic analysis (see [run_probabilistic_case][feaweld.pipeline.workflow.run_probabilistic_case])."""
    return run_probabilistic_case(case)
