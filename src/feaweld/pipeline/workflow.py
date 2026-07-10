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

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

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
    dimension: int = 2           # 2 = cross-section model, 3 = extruded solid
    base_width: float = 200.0
    base_thickness: float = 20.0
    web_height: float = 100.0
    web_thickness: float = 10.0
    weld_leg_size: float = 8.0
    groove_angle: float = 60.0   # butt welds: total included groove angle (deg)
    root_gap: float = 2.0        # butt welds: root opening (mm)
    penetration: str = "full"    # butt welds: "full" or "partial"
    length: float = 1.0  # extrusion depth (mm) when dimension == 3; quasi-2D marker otherwise

    @field_validator("dimension")
    @classmethod
    def _dimension_2_or_3(cls, value: int) -> int:
        if value not in (2, 3):
            raise ValueError(f"geometry.dimension must be 2 or 3, got {value}")
        return value

    @field_validator("penetration")
    @classmethod
    def _penetration_full_or_partial(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in ("full", "partial"):
            raise ValueError(
                f"geometry.penetration must be 'full' or 'partial', "
                f"got {value!r}"
            )
        return normalized


class MeshConfig(BaseModel):
    """Mesh generation configuration."""
    global_size: float = 2.0
    weld_toe_size: float = 0.2
    element_order: int = 2
    element_type: str = "tri"
    element_type_3d: str = "tet"
    refinement_distance: float | None = None  # None -> WeldMeshConfig default


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


class SpectrumBlock(BaseModel):
    """One constant-amplitude block of a load spectrum.

    A block is defined in either factor space (``range_factor`` /
    ``mean_factor``, fractions of the solved reference stress) or
    absolute space (``stress_range`` / ``mean_stress`` in MPa) — mixing
    the two styles within a block is rejected.
    """
    range_factor: float | None = None
    stress_range: float | None = None   # MPa
    mean_factor: float | None = None
    mean_stress: float | None = None    # MPa
    cycles: float = 1.0

    @model_validator(mode="after")
    def _one_style(self) -> "SpectrumBlock":
        has_factor = self.range_factor is not None or self.mean_factor is not None
        has_abs = self.stress_range is not None or self.mean_stress is not None
        if has_factor and has_abs:
            raise ValueError(
                "Spectrum block mixes factor keys (range_factor/mean_factor) "
                "with absolute keys (stress_range/mean_stress); use one style "
                "per block."
            )
        if self.range_factor is None and self.stress_range is None:
            raise ValueError(
                "Spectrum block needs 'range_factor' or 'stress_range'."
            )
        return self


class ResidualStressConfig(BaseModel):
    """Residual stress specification for fatigue assessment.

    At most one source may be given: a named through-thickness
    ``profile`` (see [feaweld.data.residual_stress][]), a direct surface
    ``value`` in MPa, or ``as_welded`` (yield-magnitude tensile residual
    stress).  The (possibly PWHT-relaxed) surface value always enters
    the fatigue assessment as a residual mean stress; ``superimpose``
    additionally adds the corresponding stress field to the saved
    results *after* post-processing and the fatigue assessment, so it
    reaches exported/visualized fields only — a static residual can
    shift cycle means but never a fatigue stress range.
    """
    profile: str | None = None
    value: float | None = None      # MPa
    as_welded: bool = False
    superimpose: bool = False

    @model_validator(mode="after")
    def _one_source(self) -> "ResidualStressConfig":
        n_given = sum((
            self.profile is not None,
            self.value is not None,
            self.as_welded,
        ))
        if n_given > 1:
            raise ValueError(
                "residual_stress accepts at most one of 'profile', 'value', "
                "or 'as_welded'."
            )
        return self

    @property
    def configured(self) -> bool:
        return self.profile is not None or self.value is not None or self.as_welded


class FatigueConfig(BaseModel):
    """Cyclic-loading definition and fatigue corrections.

    At most one cyclic style may be given: ``r_ratio`` (constant
    amplitude), ``blocks`` (spectrum), ``history`` (inline signal), or
    ``history_file`` (signal file, rainflow-counted).  With none given
    the fatigue assessment falls back to the legacy single-evaluation
    behavior.
    """
    r_ratio: float | None = None    # R = sigma_min / sigma_max (< 1)
    cycles: float | None = None     # design cycle count for r_ratio style
    blocks: list[SpectrumBlock] = Field(default_factory=list)
    history: list[float] = Field(default_factory=list)
    history_file: str | None = None
    history_units: Literal["load_factor", "stress"] = "load_factor"
    mean_stress_correction: Literal["none", "goodman", "gerber"] = "none"
    thickness_correction: bool = True
    thickness_exponent: float = 0.3
    surface_roughness: float | None = None  # R_a in micrometres
    environment: Literal["air", "corrosive", "seawater"] = "air"
    residual_stress: ResidualStressConfig = Field(
        default_factory=ResidualStressConfig
    )

    @model_validator(mode="after")
    def _one_cyclic_style(self) -> "FatigueConfig":
        given = [
            name
            for name, present in (
                ("r_ratio", self.r_ratio is not None),
                ("blocks", bool(self.blocks)),
                ("history", bool(self.history)),
                ("history_file", self.history_file is not None),
            )
            if present
        ]
        if len(given) > 1:
            raise ValueError(
                "Only one cyclic-loading style may be given; got "
                + " and ".join(given)
                + "."
            )
        return self


class WeldEfficiencyConfig(BaseModel):
    """Weld joint efficiency: a code-table lookup or a direct value.

    Give either all of ``standard`` / ``joint_type`` / ``examination``
    (resolved via [get_weld_efficiency][feaweld.data.weld_efficiency.get_weld_efficiency])
    or a direct ``value`` — not both.
    """
    standard: str | None = None
    joint_type: str | None = None
    examination: str | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _lookup_xor_value(self) -> "WeldEfficiencyConfig":
        has_lookup = any(
            v is not None
            for v in (self.standard, self.joint_type, self.examination)
        )
        if self.value is not None and has_lookup:
            raise ValueError(
                "weld_efficiency takes either a direct 'value' or a "
                "standard/joint_type/examination lookup, not both."
            )
        if self.value is None and not (
            self.standard and self.joint_type and self.examination
        ):
            raise ValueError(
                "weld_efficiency lookup needs 'standard', 'joint_type', "
                "and 'examination' (or give a direct 'value')."
            )
        return self


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
    notch_nominal_stress: float | None = None  # MPa; None -> linearized structural stress at the toe
    sed_control_radius: float = 0.28      # mm, R0 for structural steel
    sed_w_ref: float | None = None        # MJ/m^3 at N_ref; None -> skip SED life estimate
    linearization_points: int = 20
    weld_efficiency: WeldEfficiencyConfig | None = None


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
    fatigue: FatigueConfig = Field(default_factory=FatigueConfig)
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)
    output_dir: str = "results"

    # Directory of the YAML file this case was loaded from (set by
    # load_case); used to resolve relative paths such as
    # fatigue.history_file.  Not part of the serialized case.
    _base_dir: str | None = PrivateAttr(default=None)


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
    case = AnalysisCase(**data)
    case._base_dir = str(Path(path).resolve().parent)
    return case


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
        mesh_kwargs: dict[str, Any] = dict(
            global_size=case.mesh.global_size,
            weld_toe_size=case.mesh.weld_toe_size,
            element_order=case.mesh.element_order,
            element_type_2d=case.mesh.element_type,
            element_type_3d=case.mesh.element_type_3d,
        )
        if case.mesh.refinement_distance is not None:
            mesh_kwargs["refinement_distance"] = case.mesh.refinement_distance
        mesh_config = WeldMeshConfig(**mesh_kwargs)
        if case.geometry.dimension == 3:
            if case.geometry.length < 2.0 * case.mesh.global_size:
                result.warnings.append(
                    f"geometry.length ({case.geometry.length:g} mm) is below "
                    f"2x mesh.global_size ({case.mesh.global_size:g} mm); the "
                    "extruded solid is a sliver with barely one element "
                    "through the weld direction."
                )
            if case.mesh.weld_toe_size < case.geometry.base_thickness / 20.0:
                result.warnings.append(
                    f"mesh.weld_toe_size ({case.mesh.weld_toe_size:g} mm) is "
                    "finer than base_thickness/20 in a 3D model; expect "
                    "O(10^5+) tetrahedra. For hot-spot work a toe size of "
                    "~0.2*t is usually sufficient."
                )
            mesh = generate_mesh(joint, mesh_config, dim=3)
        else:
            mesh = generate_mesh(joint, mesh_config)
        result.mesh = mesh

        # Step 4: Solver
        from feaweld.solver.backend import get_backend
        backend = get_backend(case.solver.backend)

        # Build load case
        load_case_obj = _build_load_case(case.load, mesh)

        fea_results = _run_solver(case, backend, mesh, base, load_case_obj, result)

        # Resolve the configured residual stress (surface value), if any.
        residual_cfg = case.fatigue.residual_stress
        sigma_res: float | None = None
        residual_source: str | None = None
        if residual_cfg.configured and fea_results.stress is not None:
            T_case = case.material.temperature
            if residual_cfg.value is not None:
                sigma_res = float(residual_cfg.value)
                residual_source = "value"
            elif residual_cfg.profile is not None:
                from feaweld.data.residual_stress import surface_residual_stress
                sigma_res = surface_residual_stress(
                    residual_cfg.profile, base.sigma_y(T_case),
                )
                residual_source = f"profile:{residual_cfg.profile}"
            else:
                sigma_res = base.sigma_y(T_case)
                residual_source = "as_welded"

        # Step 4b: PWHT stress relaxation on the solved field
        residual_relaxation: float | None = None
        if case.thermal.pwht_enabled and fea_results.stress is not None:
            from feaweld.core.loads import PWHTSchedule
            schedule = PWHTSchedule(
                heating_rate=case.thermal.pwht_heating_rate,
                holding_temperature=case.thermal.pwht_temperature,
                holding_time=case.thermal.pwht_time_hours,
                cooling_rate=case.thermal.pwht_cooling_rate,
            )
            if sigma_res is not None and not case.thermal.enabled:
                # A configured residual stress is what PWHT actually
                # relaxes; the elastic load field is left untouched.
                from feaweld.solver.creep import pwht_relaxation_factor
                residual_relaxation = pwht_relaxation_factor(
                    base, schedule, sigma_res,
                )
                sigma_res *= residual_relaxation
                fea_results.metadata["pwht_residual_relaxation"] = (
                    residual_relaxation
                )
            else:
                from feaweld.solver.creep import simulate_pwht
                if not case.thermal.enabled:
                    result.warnings.append(
                        "PWHT relaxation applied to a load-stress field, not a "
                        "residual-stress field (thermal.enabled is false)."
                    )
                as_welded_vm_max = float(np.max(fea_results.stress.von_mises))
                fea_results = simulate_pwht(fea_results, base, schedule)
                fea_results.metadata["as_welded_max_von_mises"] = as_welded_vm_max

        # Step 4c: record the residual stress.  The (relaxed) surface value
        # always enters the fatigue assessment as a mean stress — a static
        # residual can shift cycle means but never a fatigue stress range —
        # so superimposing its field onto the saved stress values is
        # deferred until after post-processing and fatigue (Step 6b).
        residual_mean_for_fatigue = 0.0
        residual_meta: dict[str, Any] | None = None
        if sigma_res is not None:
            residual_meta = {
                "source": residual_source,
                "surface_value": float(sigma_res),
            }
            residual_mean_for_fatigue = float(sigma_res)
            fea_results.metadata["residual_stress"] = residual_meta

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
            weld_lines = _weld_lines_from_mesh(mesh, joint, case)
            for method in case.postprocess.stress_methods:
                try:
                    pp_result = _run_postprocess(
                        method, fea_results, mesh, case, base,
                        weld_lines=weld_lines, warnings=result.warnings,
                    )
                    result.postprocess_results[method.value] = pp_result
                except Exception as e:
                    result.errors.append(f"Post-processing {method}: {e}")

        # Step 5b: Mesh-sensitivity / singularity check (best-effort).
        # The check compares the fine solve against a coarse *linear elastic*
        # re-solve, so it is only valid when the effective fine solve was also
        # plain linear elastic.  Plasticity capping (ELASTOPLASTIC / nonlinear)
        # or PWHT relaxation would make the coarse baseline incomparable and
        # produce false negatives, so guard those out.  (A superimposed
        # residual field is no longer a concern: it is only added to the
        # stress values in Step 6b, after this check.)  The check re-meshes
        # and compares 2D sections, so extruded 3D cases skip it outright.
        if (
            case.postprocess.singularity_check
            and fea_results.stress is not None
            and case.geometry.dimension == 3
        ):
            result.warnings.append(
                "singularity check is 2D-only; skipped for dimension == 3."
            )
        elif (
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
                fatigue = _run_fatigue_assessment(
                    result.postprocess_results, case,
                    material=base,
                    base_dir=case._base_dir or Path.cwd(),
                    residual_mean=residual_mean_for_fatigue,
                    warnings=result.warnings,
                )
                # The governing method's MPa-scaled cycles feed the rainflow
                # and damage-map report figures (and `feaweld animate`).
                rainflow_cycles = fatigue.pop("_governing_cycles_mpa", None)
                if rainflow_cycles:
                    result.postprocess_results["rainflow"] = rainflow_cycles
                result.fatigue_results = fatigue
            except Exception as e:
                result.errors.append(f"Fatigue assessment: {e}")

        # Step 6b: superimpose the residual-stress field onto the saved
        # stress values.  This runs AFTER post-processing and the fatigue
        # assessment on purpose: the stress methods' reference stresses and
        # the fatigue ranges must stay load-only (the residual reaches the
        # assessment as ``residual_mean``); the superimposed field feeds the
        # saved outputs, VTK export, and report field figures only.
        if (
            sigma_res is not None
            and residual_cfg.superimpose
            and fea_results.stress is not None
        ):
            prior_vm_max = float(np.max(fea_results.stress.von_mises))
            if residual_cfg.profile is not None:
                from feaweld.data.residual_stress import (
                    residual_stress_field,
                )
                # The z/t = 0 surface of the through-thickness profile is
                # the welded plate's top face.  Every bundled joint places
                # that plate at y in [0, base_thickness]: fillet_t /
                # cruciform base plate, butt plate (plate_thickness =
                # base_thickness), the lap joint's lower plate (the upper
                # plate and its end-face toe sit above the band), and the
                # corner joint's horizontal plate (plate_thickness_h =
                # base_thickness; the vertical arm sits above the band).
                surface_y = case.geometry.base_thickness
                res_field = residual_stress_field(
                    residual_cfg.profile, mesh,
                    case.geometry.base_thickness,
                    base.sigma_y(case.material.temperature),
                    surface_coordinate=surface_y,
                )
                if residual_relaxation is not None:
                    res_field = res_field * residual_relaxation
                residual_meta["field_surface_coordinate"] = float(surface_y)
            else:
                res_field = np.zeros_like(fea_results.stress.values)
                res_field[:, 1] = sigma_res
            fea_results.stress.values[:] = (
                fea_results.stress.values + res_field
            )
            residual_meta["load_only_max_von_mises"] = prior_vm_max

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
            dimension=config.dimension,
        ),
        JointType.BUTT: lambda: ButtWeld(
            plate_width=config.base_width,
            plate_thickness=config.base_thickness,
            groove_angle=config.groove_angle,
            root_gap=config.root_gap,
            penetration=config.penetration,
            length=config.length,
            dimension=config.dimension,
        ),
        JointType.LAP: lambda: LapJoint(
            plate_thickness=config.base_thickness,
            overlap_length=config.web_height,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
            dimension=config.dimension,
        ),
        JointType.CORNER: lambda: CornerJoint(
            plate_thickness_h=config.base_thickness,
            plate_thickness_v=config.web_thickness,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
            dimension=config.dimension,
        ),
        JointType.CRUCIFORM: lambda: CruciformJoint(
            plate_thickness=config.base_thickness,
            web_thickness=config.web_thickness,
            weld_leg_size=config.weld_leg_size,
            length=config.length,
            dimension=config.dimension,
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

    # In 3D the torch travels along the extruded weld: start at the min-z
    # end of the first toe line, heading +Z.  2D keeps the legacy
    # toe-centroid start with +X travel.
    start = None
    direction = np.array([1.0, 0.0, 0.0])
    if case.geometry.dimension == 3:
        line_ids = mesh.node_sets.get("weld_toe_0")
        if line_ids is not None and len(line_ids):
            coords = mesh.nodes[line_ids]
            start = coords[np.argmin(coords[:, 2])]
            direction = np.array([0.0, 0.0, 1.0])
    if start is None:
        toe_ids = mesh.node_sets.get("weld_toe")
        if toe_ids is not None and len(toe_ids):
            start = mesh.nodes[toe_ids].mean(axis=0)
        else:
            start = mesh.nodes.mean(axis=0)
    start = np.array(start, dtype=np.float64)
    if start.shape[0] == 2:
        start = np.append(start, 0.0)

    return GoldakHeatSource(
        power=heat.power,
        a_f=heat.a_f, a_r=heat.a_r, b=heat.b, c=heat.c,
        f_f=heat.f_f, f_r=heat.f_r,
        travel_speed=heat.travel_speed,
        start_position=start,
        direction=direction,
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


def _default_weld_line(mesh, case):
    """The legacy single weld line: the ``weld_toe`` node set (or node 0)."""
    from feaweld.core.types import WeldLineDefinition

    return WeldLineDefinition(
        name="weld_toe",
        node_ids=np.array(list(mesh.node_sets.get("weld_toe", [0]))),
        plate_thickness=case.geometry.base_thickness,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )


def _toe_plate_thickness(joint, normal, default: float) -> float:
    """Thickness of the plate a weld toe lies on.

    The joint normal convention (see
    [get_weld_toe_normals][feaweld.geometry.joints.JointGeometry.get_weld_toe_normals])
    encodes the carrying plate: a toe with a dominant ``±Y`` normal sits on
    a horizontal plate surface (through-thickness along Y), one with a
    dominant ``±X`` normal on a vertical web or end face (through-thickness
    along X).  The matching thickness is the first of the joint's dataclass
    fields that exists — horizontal: ``base_thickness`` /
    ``plate_thickness`` / ``plate_thickness_h``; vertical:
    ``web_thickness`` / ``plate_thickness_v`` / ``plate_thickness`` —
    falling back to *default*.
    """
    if abs(float(normal[1])) >= abs(float(normal[0])):
        candidates = ("base_thickness", "plate_thickness", "plate_thickness_h")
    else:
        candidates = ("web_thickness", "plate_thickness_v", "plate_thickness")
    for attr in candidates:
        value = getattr(joint, attr, None)
        if value is not None:
            return float(value)
    return float(default)


def _weld_lines_from_mesh(mesh, joint, case):
    """Build the weld-line definitions consumed by post-processing.

    For 2D cases this is the legacy single ``weld_toe`` line.  For 3D
    cases each ``weld_toe_{i}`` node set becomes one line: its node ids
    are spatially ordered along the toe via
    [order_weld_line_nodes][feaweld.postprocess.hotspot.order_weld_line_nodes],
    the surface normal comes from ``joint.get_weld_toe_normals()``, and
    the plate thickness from the plate the toe lies on (see
    [_toe_plate_thickness][feaweld.pipeline.workflow._toe_plate_thickness]).

    Returns
    -------
    list of WeldLineDefinition
        At least one entry; falls back to the legacy default line when no
        per-toe node sets exist on the mesh.
    """
    if case.geometry.dimension != 3:
        return [_default_weld_line(mesh, case)]

    from feaweld.core.types import WeldLineDefinition
    from feaweld.postprocess.hotspot import order_weld_line_nodes

    lines = []
    for i, normal in enumerate(joint.get_weld_toe_normals()):
        ids = mesh.node_sets.get(f"weld_toe_{i}")
        if ids is None or len(ids) == 0:
            continue
        lines.append(WeldLineDefinition(
            name=f"weld_toe_{i}",
            node_ids=order_weld_line_nodes(
                mesh, np.asarray(ids, dtype=np.int64)
            ),
            plate_thickness=_toe_plate_thickness(
                joint, normal, case.geometry.base_thickness
            ),
            normal_direction=np.asarray(normal, dtype=np.float64),
        ))
    if not lines:
        return [_default_weld_line(mesh, case)]
    return lines


def _run_postprocess(method, fea_results, mesh, case, material,
                     weld_lines=None, warnings=None):
    """Run a single post-processing method (one branch per StressMethod).

    Parameters
    ----------
    method, fea_results, mesh, case, material
        Stress method, solved results, mesh, case, and base material.
    weld_lines : list of WeldLineDefinition, optional
        Weld lines from [_weld_lines_from_mesh][feaweld.pipeline.workflow._weld_lines_from_mesh].
        *None* rebuilds the legacy single default line (2D behavior).
    warnings : list of str, optional
        Sink for non-fatal diagnostics (poorly resolved hot-spot stations
        on 3D toe lines).
    """
    if weld_lines is None:
        weld_lines = [_default_weld_line(mesh, case)]
    is_3d = case.geometry.dimension == 3

    # Critical station: the highest-von-Mises toe node over every weld
    # line.  The line carrying it supplies the plate thickness and surface
    # normal for the point-evaluation methods (notch, SED, linearization,
    # nominal).
    vm = fea_results.stress.von_mises
    weld_line = weld_lines[0]
    toe_node = 0
    best_vm = -np.inf
    for line in weld_lines:
        ids = line.node_ids[line.node_ids < vm.shape[0]]
        if len(ids) == 0:
            continue
        candidate = int(ids[np.argmax(vm[ids])])
        if vm[candidate] > best_vm:
            best_vm = float(vm[candidate])
            toe_node = candidate
            weld_line = line

    if method.value in ("hotspot_linear", "hotspot_quadratic"):
        from feaweld.postprocess.hotspot import (
            HotSpotType, hotspot_stress_linear, hotspot_stress_quadratic,
        )

        def _stations(line):
            if method.value == "hotspot_linear":
                return hotspot_stress_linear(fea_results, line, HotSpotType.TYPE_A)
            return hotspot_stress_quadratic(fea_results, line)

        if not is_3d:
            results = _stations(weld_line)
            return {"results": results, "max_stress": max(r.hot_spot_stress for r in results) if results else 0.0}

        all_results = []
        per_line: list[tuple[str, list]] = []
        for line in weld_lines:
            stations = _stations(line)
            all_results.extend(stations)
            per_line.append((line.name, stations))
            if warnings is not None:
                n_coarse = sum(1 for r in stations if not r.well_resolved)
                if n_coarse:
                    warnings.append(
                        f"{method.value}: {n_coarse}/{len(stations)} "
                        f"station(s) on '{line.name}' not well resolved "
                        "(mesh too coarse for surface extrapolation at the "
                        "toe)"
                    )

        # Poorly resolved stations carry raw nodal values, not surface
        # extrapolations (both reference points snapped to the same or a
        # far-away node), so they must not compete for the critical line
        # while any properly extrapolated station exists.  With no
        # well-resolved station anywhere, fall back to all of them.
        any_resolved = any(r.well_resolved for r in all_results)
        if all_results and not any_resolved and warnings is not None:
            warnings.append(
                f"{method.value}: no well-resolved station on any weld "
                "line -- the mesh cannot support hot-spot surface "
                "extrapolation at the toe; max_stress/critical_line fall "
                "back to raw (unextrapolated) nodal values."
            )
        critical_line = None
        max_stress = 0.0
        for name, stations in per_line:
            candidates = (
                [r for r in stations if r.well_resolved]
                if any_resolved else stations
            )
            if not candidates:
                continue
            line_max = max(r.hot_spot_stress for r in candidates)
            if critical_line is None or line_max > max_stress:
                max_stress = line_max
                critical_line = name
        return {
            "results": all_results,
            "max_stress": max_stress if all_results else 0.0,
            "critical_line": critical_line,
            "n_stations": len(all_results),
        }

    elif method.value == "structural_dong":
        if is_3d:
            raise ValueError(
                "structural_dong currently supports 2D cross-section models "
                "only; use hot-spot or effective-notch methods for "
                "dimension == 3."
            )
        from feaweld.postprocess.dong import (
            MASTER_SN_H, dong_structural_stress, dong_fatigue_life,
        )
        dong_result = dong_structural_stress(fea_results, weld_line)
        dong_result = dong_fatigue_life(dong_result, case.geometry.base_thickness)
        out = {
            "dong_result": dong_result,
            "max_stress": float(np.max(dong_result.structural_stress)),
        }
        if dong_result.fatigue_life is not None and len(dong_result.fatigue_life):
            out["fatigue_life"] = float(np.min(dong_result.fatigue_life))
            out["sn_curve_used"] = "ASME master S-N curve (Dong)"
            out["life_exponent"] = MASTER_SN_H
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
        t = weld_line.plate_thickness
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
            "sn_curve_obj": FAT225_CURVE,
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
            "evaluated_at": int(toe_node),
        }
        if case.postprocess.sed_w_ref:
            sed_result = sed_fatigue_life(sed_result, W_ref=case.postprocess.sed_w_ref)
            out["sed_result"] = sed_result
            out["fatigue_life"] = sed_result.fatigue_life
            out["sn_curve_used"] = "SED power law (Lazzarin)"
            # N ∝ W^(-1/slope) with W ∝ σ², so the stress-space life
            # exponent is 2/slope (sed_fatigue_life default slope = 1.5).
            out["life_exponent"] = 2.0 / 1.5
        return out

    elif method.value == "linearization":
        from feaweld.postprocess.linearization import linearize_at_weld_toe
        lin = linearize_at_weld_toe(
            fea_results, toe_node,
            plate_thickness=weld_line.plate_thickness,
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
        t = weld_line.plate_thickness
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
        efficiency_cfg = case.postprocess.weld_efficiency
        joint_efficiency = 1.0
        if efficiency_cfg is not None:
            if efficiency_cfg.value is not None:
                joint_efficiency = float(efficiency_cfg.value)
            else:
                from feaweld.data.weld_efficiency import get_weld_efficiency
                joint_efficiency = get_weld_efficiency(
                    efficiency_cfg.standard,
                    efficiency_cfg.joint_type,
                    efficiency_cfg.examination,
                ).efficiency
        checks = asme_allowable_check(
            cat, S_m=S_m, S_y=S_y, joint_efficiency=joint_efficiency,
        )
        out = {
            "categorization": cat,
            "membrane": cat.membrane,
            "bending": cat.bending,
            "asme_checks": checks,
            "max_stress": cat.primary_plus_bending,
        }
        if efficiency_cfg is not None:
            out["weld_efficiency"] = joint_efficiency
        return out

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


def _run_fatigue_assessment(
    postprocess_results,
    case,
    material=None,
    base_dir=None,
    residual_mean=0.0,
    warnings=None,
):
    """Run fatigue assessment on post-processing results.

    Without a cyclic-loading definition in ``case.fatigue`` this is the
    legacy single-evaluation path: each method's ``max_stress`` is read
    off the case S-N curve once, and methods with a mandated curve keep
    their own constant-amplitude life.  With a cyclic definition the
    cycle set is scaled by each method's reference stress (its
    ``max_stress`` at the solved maximum load) and assessed with
    mean-stress correction and strength knockdowns via
    [assess_spectrum][feaweld.fatigue.assessment.assess_spectrum];
    methods with their own single-slope life law (Dong, SED) use
    [spectrum_life_power_law][feaweld.fatigue.assessment.spectrum_life_power_law]
    without corrections.

    Parameters
    ----------
    postprocess_results : dict
        Per-method dicts from [_run_postprocess][feaweld.pipeline.workflow._run_postprocess].
    case : AnalysisCase
        The analysis case (S-N spec, fatigue block, geometry).
    material : Material, optional
        Base metal; loaded from ``case.material.base_metal`` when absent
        and the spectrum path needs material properties.
    base_dir : str or Path, optional
        Directory for resolving a relative ``fatigue.history_file``.
    residual_mean : float
        Residual mean stress (MPa) added to every cycle mean — the
        (possibly PWHT-relaxed) surface residual whenever one is
        configured, regardless of ``superimpose``; inert with
        ``mean_stress_correction`` ``"none"``.
    warnings : list of str, optional
        Sink for configuration warnings (double-count / inert combos).

    Returns
    -------
    dict
        Per-method entries plus ``sn_curve``; the spectrum path adds
        ``loading`` / ``corrections`` sub-dicts and a private
        ``_governing_cycles_mpa`` key with the highest-damage method's
        MPa-scaled cycles.
    """
    from feaweld.fatigue.sn_curves import parse_sn_spec
    from feaweld.fatigue.assessment import build_cycle_set

    sn_spec = case.postprocess.sn_curve
    curve = parse_sn_spec(sn_spec)

    fat_cfg = case.fatigue
    cycle_set = build_cycle_set(
        r_ratio=fat_cfg.r_ratio,
        cycles=fat_cfg.cycles,
        blocks=[b.model_dump(exclude_none=True) for b in fat_cfg.blocks],
        history=fat_cfg.history or None,
        history_file=fat_cfg.history_file,
        history_units=fat_cfg.history_units,
        base_dir=base_dir,
    )

    if cycle_set is None:
        if (
            warnings is not None
            and fat_cfg.residual_stress.configured
            and not fat_cfg.residual_stress.superimpose
        ):
            warnings.append(
                "residual_stress is configured but inert: without a "
                "cyclic-loading definition (r_ratio/blocks/history/"
                "history_file) the legacy single-evaluation path ignores "
                "the residual mean and any mean_stress_correction."
            )
        fatigue = {"sn_curve": sn_spec}

        for method, pp_result in postprocess_results.items():
            if not isinstance(pp_result, dict):
                continue
            if pp_result.get("fatigue_life") is not None:
                # Methods with a mandated curve (notch/FAT225, Dong master
                # curve, SED power law) carry their own life estimate.
                fatigue[method] = {
                    "stress_range": pp_result.get("max_stress"),
                    "life": float(pp_result["fatigue_life"]),
                    "sn_curve": pp_result.get("sn_curve_used", sn_spec),
                }
            elif "max_stress" in pp_result:
                stress = pp_result["max_stress"]
                fatigue[method] = {"stress_range": stress, "life": curve.life(stress)}

        return fatigue

    return _assess_cycle_set(
        postprocess_results, case, cycle_set, curve,
        material=material, residual_mean=residual_mean, warnings=warnings,
    )


def _assess_cycle_set(
    postprocess_results,
    case,
    cycle_set,
    curve,
    *,
    material,
    residual_mean,
    warnings,
):
    """Spectrum branch of [_run_fatigue_assessment][feaweld.pipeline.workflow._run_fatigue_assessment]."""
    from feaweld.fatigue.assessment import (
        assess_spectrum, scale_cycles, spectrum_life_power_law,
    )
    from feaweld.fatigue.knockdown import (
        environment_factor, surface_finish_factor, thickness_correction,
    )

    fat_cfg = case.fatigue
    sn_spec = case.postprocess.sn_curve
    triples, kind, is_absolute = cycle_set
    correction = fat_cfg.mean_stress_correction

    if material is None:
        from feaweld.core.materials import load_material
        material = load_material(case.material.base_metal)
    T = case.material.temperature

    sigma_u = None
    if correction != "none" or fat_cfg.surface_roughness is not None:
        try:
            sigma_u = material.sigma_u(T)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Material '{material.name}' has no ultimate-strength data; "
                "it is required for Goodman/Gerber mean-stress correction "
                "and the surface-roughness factor."
            ) from e

    k_thickness = (
        thickness_correction(
            case.geometry.base_thickness, exponent=fat_cfg.thickness_exponent,
        )
        if fat_cfg.thickness_correction else 1.0
    )
    k_surface = (
        surface_finish_factor(fat_cfg.surface_roughness, sigma_u)
        if fat_cfg.surface_roughness is not None else 1.0
    )
    k_env = environment_factor(fat_cfg.environment)
    strength_factor = k_thickness * k_surface * k_env

    if warnings is not None:
        residual_cfg = fat_cfg.residual_stress
        if correction != "none" and residual_cfg.as_welded:
            warnings.append(
                "Goodman/Gerber mean-stress correction combined with "
                "as-welded residual stress double-counts the residual mean: "
                "design S-N curves already assume high tensile residuals."
            )
        if residual_cfg.configured and correction == "none":
            if residual_cfg.superimpose:
                warnings.append(
                    "residual_stress does not enter the fatigue "
                    "assessment: with mean_stress_correction 'none' the "
                    "residual mean is ignored, and superimpose only adds "
                    "the residual field to the saved stress output (after "
                    "post-processing)."
                )
            else:
                warnings.append(
                    "residual_stress is configured but inert: with "
                    "mean_stress_correction 'none' the residual mean does "
                    "not enter the fatigue assessment."
                )

    n_per_repeat = sum(count for _rng, _mean, count in triples)

    fatigue: dict[str, Any] = {"sn_curve": sn_spec}
    governing: tuple[float, list] | None = None

    for method, pp_result in postprocess_results.items():
        if not isinstance(pp_result, dict):
            continue
        ref = pp_result.get("max_stress")
        entry: dict[str, Any] | None = None
        cycles_mpa: list | None = None

        if pp_result.get("sn_curve_obj") is not None and ref is not None:
            # Method with a mandated S-N curve (effective notch / FAT225).
            cycles_mpa = triples if is_absolute else scale_cycles(triples, ref)
            spec_out = assess_spectrum(
                cycles_mpa, pp_result["sn_curve_obj"],
                mean_correction=correction, sigma_u=sigma_u,
                residual_mean=residual_mean, strength_factor=strength_factor,
            )
            entry = {
                "stress_range": ref,
                "life": spec_out["life_cycles"],
                "sn_curve": pp_result.get("sn_curve_used", sn_spec),
                "damage": spec_out["damage"],
                "life_repeats": spec_out["life_repeats"],
                "equivalent_stress_range": spec_out["equivalent_stress_range"],
            }
        elif (
            pp_result.get("fatigue_life") is not None
            and pp_result.get("life_exponent") is not None
        ):
            # Method with its own single-slope life law (Dong, SED).  Its
            # code mandates the law as-is: no mean-stress correction, no
            # strength knockdowns.
            life_ref = float(pp_result["fatigue_life"])
            entry = {
                "stress_range": ref,
                "life": life_ref,
                "sn_curve": pp_result.get("sn_curve_used", sn_spec),
                "corrections_applied": False,
            }
            if not is_absolute:
                factor_cycles = triples
            elif ref:
                factor_cycles = [
                    (rng / ref, mean / ref, count)
                    for rng, mean, count in triples
                ]
            else:
                # Absolute cycles but no reference stress to convert into
                # factor space -- keep the constant-amplitude estimate.
                factor_cycles = None
            if factor_cycles is not None:
                life_repeats = spectrum_life_power_law(
                    life_ref, factor_cycles, float(pp_result["life_exponent"]),
                )
                if life_repeats <= 0:
                    damage = float("inf")
                elif np.isinf(life_repeats):
                    damage = 0.0
                else:
                    damage = 1.0 / life_repeats
                entry["life"] = life_repeats * n_per_repeat
                entry["damage"] = damage
                entry["life_repeats"] = life_repeats
                if ref:
                    cycles_mpa = (
                        list(triples) if is_absolute
                        else scale_cycles(triples, ref)
                    )
        elif pp_result.get("fatigue_life") is not None:
            entry = {
                "stress_range": ref,
                "life": float(pp_result["fatigue_life"]),
                "sn_curve": pp_result.get("sn_curve_used", sn_spec),
            }
        elif ref is not None:
            cycles_mpa = triples if is_absolute else scale_cycles(triples, ref)
            spec_out = assess_spectrum(
                cycles_mpa, curve,
                mean_correction=correction, sigma_u=sigma_u,
                residual_mean=residual_mean, strength_factor=strength_factor,
            )
            entry = {
                "stress_range": ref,
                "life": spec_out["life_cycles"],
                "sn_curve": sn_spec,
                "damage": spec_out["damage"],
                "life_repeats": spec_out["life_repeats"],
                "equivalent_stress_range": spec_out["equivalent_stress_range"],
            }

        if entry is None:
            continue
        if fat_cfg.cycles is not None and "damage" in entry:
            entry["utilization"] = entry["damage"]
        fatigue[method] = entry
        if cycles_mpa is not None and "damage" in entry:
            if governing is None or entry["damage"] > governing[0]:
                governing = (entry["damage"], cycles_mpa)

    loading: dict[str, Any] = {
        "type": kind,
        "r_ratio": fat_cfg.r_ratio,
        "cycles": fat_cfg.cycles,
    }
    if kind == "history":
        loading["n_rainflow_cycles"] = len(triples)
    fatigue["loading"] = loading
    fatigue["corrections"] = {
        "mean_stress": correction,
        "k_thickness": k_thickness,
        "k_surface": k_surface,
        "k_environment": k_env,
        "k_total": strength_factor,
        "residual_mean": residual_mean,
    }
    if governing is not None:
        fatigue["_governing_cycles_mpa"] = [
            (float(rng), float(mean), float(count))
            for rng, mean, count in governing[1]
        ]

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
