"""Tests for the workflow dispatch layer (solver, loads, post-processing).

These exercise the solver-agnostic wiring in ``feaweld.pipeline.workflow``
using the :class:`FakeBackend` from ``conftest`` — no FEniCSx / CalculiX /
Gmsh required.  They pin the contract between the YAML config and each
downstream module: which solver method runs for which ``SolverType``, how a
load block becomes a ``LoadCase``, and what every ``StressMethod`` returns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    LoadType,
    SolverType,
    StressField,
    StressMethod,
)
from feaweld.core.loads import moment_to_nodal_forces
from feaweld.fatigue.sn_curves import parse_sn_spec
from feaweld.pipeline.workflow import (
    AnalysisCase,
    FatigueConfig,
    GeometryConfig,
    LoadConfig,
    MaterialConfig,
    MeshConfig,
    PostProcessConfig,
    ProbabilisticConfig,
    SolverConfig,
    ThermalConfig,
    WeldEfficiencyConfig,
    WorkflowResult,
    _build_geometry,
    _build_heat_source,
    _build_load_case,
    _run_fatigue_assessment,
    _run_postprocess,
    _run_singularity_check,
    _run_solver,
    run_analysis,
    run_probabilistic_case,
)


# ---------------------------------------------------------------------------
# Local material (temperature-dependent, with creep + hardening params)
# ---------------------------------------------------------------------------

def _material(**over) -> Material:
    kw = dict(
        name="DispatchSteel",
        density=7850.0,
        elastic_modulus={20.0: 200000.0, 500.0: 160000.0, 800.0: 70000.0},
        poisson_ratio={20.0: 0.3, 500.0: 0.3},
        yield_strength={20.0: 250.0, 500.0: 165.0, 800.0: 25.0},
        ultimate_strength={20.0: 400.0, 500.0: 310.0},
        thermal_conductivity={20.0: 51.9, 500.0: 37.2},
        specific_heat={20.0: 440.0, 500.0: 650.0},
        thermal_expansion={20.0: 11.7e-6, 500.0: 14.0e-6},
        creep_A=1e-12,
        creep_n=5.0,
        creep_m=0.0,
        hardening_modulus=1000.0,
    )
    kw.update(over)
    return Material(**kw)


# ---------------------------------------------------------------------------
# Post-processing dispatch: all 8 StressMethod values
# ---------------------------------------------------------------------------

_MAX_STRESS_METHODS = {
    StressMethod.HOTSPOT_LINEAR,
    StressMethod.HOTSPOT_QUADRATIC,
    StressMethod.LINEARIZATION,
    StressMethod.NOMINAL,
    StressMethod.STRUCTURAL_DONG,
    StressMethod.NOTCH_STRESS,
}


def _dispatch_case() -> AnalysisCase:
    return AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0,
                                weld_leg_size=8.0),
        postprocess=PostProcessConfig(sed_w_ref=1.0),
    )


@pytest.mark.parametrize("method", list(StressMethod))
def test_postprocess_all_methods_return_dict(method, grid_plate_mesh,
                                             fake_backend):
    """Every StressMethod dispatches to a module and returns a dict."""
    mat = _material()
    case = _dispatch_case()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    out = _run_postprocess(method, results, grid_plate_mesh, case, mat)
    assert isinstance(out, dict)
    # Defensive fall-through would leave a "warning" key; real dispatch never does.
    assert "warning" not in out


@pytest.mark.parametrize("method", sorted(_MAX_STRESS_METHODS, key=lambda m: m.value))
def test_postprocess_reports_max_stress(method, grid_plate_mesh, fake_backend):
    """Stress-based methods surface a finite ``max_stress``."""
    mat = _material()
    case = _dispatch_case()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    out = _run_postprocess(method, results, grid_plate_mesh, case, mat)
    assert "max_stress" in out
    assert np.isfinite(out["max_stress"])


def test_postprocess_notch_reports_fatigue_life(grid_plate_mesh, fake_backend):
    """The effective-notch method carries its own FAT225 life estimate."""
    mat = _material()
    case = _dispatch_case()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    out = _run_postprocess(StressMethod.NOTCH_STRESS, results, grid_plate_mesh,
                           case, mat)
    assert "fatigue_life" in out
    assert out["fatigue_life"] > 0.0


def test_postprocess_notch_parametric_contract(grid_plate_mesh, fake_backend):
    """The effective-notch method returns a mesh-insensitive parametric SCF.

    The SCF is the parametric weld-toe K_t (not the raw singular FE peak) and
    the reported stress is K_t times the linearized structural stress at the
    toe, so there is no full notch stress field.
    """
    from feaweld.postprocess.linearization import linearize_at_weld_toe
    from feaweld.postprocess.notch_stress import notch_stress_scf_parametric

    mat = _material()
    case = _dispatch_case()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    out = _run_postprocess(StressMethod.NOTCH_STRESS, results, grid_plate_mesh,
                           case, mat)
    assert {"max_stress", "scf", "fatigue_life", "sn_curve_used"} <= set(out)
    assert out["sn_curve_used"] == "IIW FAT225 (effective notch)"
    assert out["notch_result"].notch_stress_field is None

    expected_kt = notch_stress_scf_parametric(
        toe_radius=case.postprocess.notch_radius, toe_angle=45.0,
        plate_thickness=case.geometry.base_thickness, weld_toe_type="fillet",
    )
    assert out["scf"] == pytest.approx(expected_kt)

    # Reported stress = K_t * structural (membrane+bending) stress at the toe.
    toe_ids = grid_plate_mesh.node_sets["weld_toe"]
    vm = results.stress.von_mises
    toe_node = int(toe_ids[np.argmax(vm[toe_ids])])
    lin = linearize_at_weld_toe(
        results, toe_node, plate_thickness=case.geometry.base_thickness,
        surface_normal=np.array([0.0, 1.0, 0.0]),
        n_points=case.postprocess.linearization_points,
    )
    assert out["max_stress"] == pytest.approx(
        expected_kt * lin.membrane_plus_bending_scalar
    )


def test_fat225_two_slope_continuity():
    """FAT225 is a proper two-slope IIW curve, continuous at the 1e7 knee."""
    from feaweld.postprocess.notch_stress import FAT225_CURVE, FAT225_KNEE_STRESS

    knee = FAT225_KNEE_STRESS
    seg3, seg5 = FAT225_CURVE.segments
    # Both anchored formulas give N = 1e7 at the knee stress.
    n_from_m3 = seg3.C / knee ** seg3.m
    n_from_m5 = seg5.C / knee ** seg5.m
    assert n_from_m3 == pytest.approx(n_from_m5, rel=1e-6)
    assert n_from_m3 == pytest.approx(1e7, rel=1e-6)

    # SNCurve.life() is continuous across the knee (m=3 above, m=5 below).
    below = FAT225_CURVE.life(knee * (1.0 - 1e-9))
    above = FAT225_CURVE.life(knee * (1.0 + 1e-9))
    assert below == pytest.approx(above, rel=1e-6)
    # IIW variable-amplitude cutoff.
    assert FAT225_CURVE.cutoff_cycles == 1e9


def test_postprocess_sed_life_requires_w_ref(grid_plate_mesh, fake_backend):
    """SED life is only emitted when ``sed_w_ref`` is configured."""
    mat = _material()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    with_ref = _run_postprocess(
        StressMethod.SED, results, grid_plate_mesh,
        _dispatch_case(), mat,
    )
    assert "fatigue_life" in with_ref

    no_ref_case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        postprocess=PostProcessConfig(sed_w_ref=None),
    )
    without_ref = _run_postprocess(
        StressMethod.SED, results, grid_plate_mesh, no_ref_case, mat,
    )
    assert "fatigue_life" not in without_ref
    assert "averaged_sed" in without_ref


def test_postprocess_nominal_asme_categories(grid_plate_mesh, fake_backend):
    """The nominal method returns the four ASME categories + a categorization."""
    mat = _material()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    out = _run_postprocess(StressMethod.NOMINAL, results, grid_plate_mesh,
                           _dispatch_case(), mat)
    assert "categorization" in out
    assert set(out["asme_checks"]) == {"Pm", "PL", "Pm+Pb", "PL+Pb+Q"}
    for check in out["asme_checks"].values():
        assert {"value", "limit", "ratio", "passes"} <= set(check)


# ---------------------------------------------------------------------------
# Load-case construction
# ---------------------------------------------------------------------------

@pytest.fixture
def load_mesh():
    """A 4-node square with bottom/top node sets (top spans x = 0 and 10)."""
    nodes = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
                     dtype=np.float64)
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return FEMesh(
        nodes=nodes, elements=elements, element_type=ElementType.TRI3,
        node_sets={
            "bottom": np.array([0, 1]),
            "top": np.array([2, 3]),
            "weld_toe": np.array([1, 2]),
        },
    )


def _by_type(loads, bc_type):
    return [bc for bc in loads if bc.bc_type == bc_type]


def test_build_load_case_all_five_loads(load_mesh):
    cfg = LoadConfig(
        axial_force=2000.0, shear_force=600.0, bending_moment=5000.0,
        pressure=3.0, temperature_delta=80.0,
    )
    lc = _build_load_case(cfg, load_mesh)
    n_top = len(load_mesh.node_sets["top"])

    # A single fixed-bottom constraint.
    assert len(lc.constraints) == 1
    assert lc.constraints[0].node_set == "bottom"
    assert lc.constraints[0].bc_type == LoadType.DISPLACEMENT

    forces = _by_type(lc.loads, LoadType.FORCE)

    # Axial: per-node magnitude along +y.
    axial = next(f for f in forces if f.direction is not None
                 and np.allclose(f.direction, [0, 1, 0]))
    np.testing.assert_allclose(axial.values, [cfg.axial_force / n_top])

    # Shear: per-node magnitude along +x.
    shear = next(f for f in forces if f.direction is not None
                 and np.allclose(f.direction, [1, 0, 0]))
    np.testing.assert_allclose(shear.values, [cfg.shear_force / n_top])

    # Bending: (n, 3) couple with zero net force, net moment ≈ M.
    bending = next(f for f in forces if f.direction is None)
    assert bending.values.ndim == 2
    x = load_mesh.nodes[load_mesh.node_sets["top"]][:, 0]
    fy = bending.values[:, 1]
    assert abs(float(np.sum(fy))) < 1e-9
    net_moment = float(np.sum(fy * (x - x.mean())))
    np.testing.assert_allclose(net_moment, cfg.bending_moment, rtol=1e-9)

    # Pressure.
    pressure = _by_type(lc.loads, LoadType.PRESSURE)
    assert len(pressure) == 1
    np.testing.assert_allclose(pressure[0].values, [cfg.pressure])

    # Temperature stored as absolute (20 C reference + ΔT).
    temperature = _by_type(lc.loads, LoadType.TEMPERATURE)
    assert len(temperature) == 1
    np.testing.assert_allclose(temperature[0].values, [20.0 + cfg.temperature_delta])


def test_build_load_case_default_is_backward_compatible(load_mesh):
    """A default LoadConfig produces only the fixed-bottom constraint."""
    lc = _build_load_case(LoadConfig(), load_mesh)
    assert lc.loads == []
    assert len(lc.constraints) == 1
    assert lc.constraints[0].node_set == "bottom"


@pytest.mark.parametrize("field,bc_type", [
    ("axial_force", LoadType.FORCE),
    ("shear_force", LoadType.FORCE),
    ("bending_moment", LoadType.FORCE),
    ("pressure", LoadType.PRESSURE),
    ("temperature_delta", LoadType.TEMPERATURE),
])
def test_build_load_case_single_load(field, bc_type, load_mesh):
    """Only the configured load appears; a zero value adds nothing."""
    lc = _build_load_case(LoadConfig(**{field: 1234.0}), load_mesh)
    assert len(lc.loads) == 1
    assert lc.loads[0].bc_type == bc_type


def test_moment_to_nodal_forces_couple(load_mesh):
    bc = moment_to_nodal_forces(load_mesh, "top", 5000.0)
    assert bc.bc_type == LoadType.FORCE
    assert bc.values.shape == (2, 3)
    x = load_mesh.nodes[load_mesh.node_sets["top"]][:, 0]
    fy = bc.values[:, 1]
    assert abs(float(np.sum(fy))) < 1e-9
    np.testing.assert_allclose(float(np.sum(fy * (x - x.mean()))), 5000.0, rtol=1e-9)


def test_moment_to_nodal_forces_degenerate_axis_raises(load_mesh):
    # The weld_toe set (nodes 1 and 2) shares x = 10, so a moment about
    # axis 0 has no lever arm.
    with pytest.raises(ValueError, match="same coordinate"):
        moment_to_nodal_forces(load_mesh, "weld_toe", 5000.0, axis=0)


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------

def _solve(case, mesh, mat, backend):
    result = WorkflowResult(case=case)
    lc = _build_load_case(case.load, mesh)
    res = _run_solver(case, backend, mesh, mat, lc, result)
    return res, result


def test_run_solver_linear_elastic(load_mesh, fake_backend):
    case = AnalysisCase(solver=SolverConfig(solver_type=SolverType.LINEAR_ELASTIC),
                        load=LoadConfig(axial_force=1000.0))
    res, _ = _solve(case, load_mesh, _material(), fake_backend)
    assert fake_backend.methods_called() == ["solve_static"]
    assert res.stress is not None


@pytest.mark.parametrize("cfg", [
    SolverConfig(solver_type=SolverType.ELASTOPLASTIC),
    SolverConfig(solver_type=SolverType.LINEAR_ELASTIC, nonlinear=True),
])
def test_run_solver_elastoplastic(cfg, load_mesh, fake_backend):
    case = AnalysisCase(solver=cfg, load=LoadConfig(axial_force=1000.0))
    res, _ = _solve(case, load_mesh, _material(), fake_backend)
    assert fake_backend.methods_called() == ["solve_static"]
    assert res.metadata["plasticity"]["model"] == "j2_radial_return_postcorrection"


def test_run_solver_thermal_steady(load_mesh, fake_backend):
    case = AnalysisCase(solver=SolverConfig(solver_type=SolverType.THERMAL_STEADY))
    _solve(case, load_mesh, _material(), fake_backend)
    assert fake_backend.methods_called() == ["solve_thermal_steady"]


def test_run_solver_thermal_transient(load_mesh, fake_backend):
    case = AnalysisCase(
        solver=SolverConfig(solver_type=SolverType.THERMAL_TRANSIENT, n_time_steps=3),
    )
    _solve(case, load_mesh, _material(), fake_backend)
    assert fake_backend.methods_called() == ["solve_thermal_transient"]


def test_run_solver_thermomechanical(load_mesh, fake_backend):
    case = AnalysisCase(
        solver=SolverConfig(solver_type=SolverType.THERMOMECHANICAL, n_time_steps=4),
        load=LoadConfig(axial_force=1000.0),
    )
    _solve(case, load_mesh, _material(), fake_backend)
    calls = fake_backend.methods_called()
    assert calls.count("solve_thermal_transient") == 1
    assert calls.count("solve_static") == 4  # one static solve per time step


def test_run_solver_thermal_enabled_routes_to_thermomechanical(load_mesh, fake_backend):
    case = AnalysisCase(
        solver=SolverConfig(solver_type=SolverType.LINEAR_ELASTIC, n_time_steps=3),
        thermal=ThermalConfig(enabled=True),
        load=LoadConfig(axial_force=1000.0),
    )
    _solve(case, load_mesh, _material(), fake_backend)
    calls = fake_backend.methods_called()
    assert calls.count("solve_thermal_transient") == 1
    assert calls.count("solve_static") == 3


def test_run_solver_creep(load_mesh, fake_backend):
    case = AnalysisCase(
        solver=SolverConfig(solver_type=SolverType.CREEP, creep_temperature=550.0,
                            creep_time_hours=1.0),
        load=LoadConfig(axial_force=1000.0),
    )
    res, _ = _solve(case, load_mesh, _material(), fake_backend)
    assert fake_backend.methods_called() == ["solve_static"]
    assert "creep_relaxation" in res.metadata


# ---------------------------------------------------------------------------
# PWHT block: relaxes a solved stress field
# ---------------------------------------------------------------------------

def test_pwht_relaxation_is_non_increasing():
    from feaweld.core.loads import PWHTSchedule
    from feaweld.solver.creep import simulate_pwht

    mat = _material(creep_A=1e-20, creep_n=5.0, creep_m=0.0)
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mesh = FEMesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
                  element_type=ElementType.TET4)
    stress = np.zeros((4, 6))
    stress[:, 0] = 280.0
    results = FEAResults(mesh=mesh, stress=StressField(values=stress))

    schedule = PWHTSchedule(heating_rate=200.0, holding_temperature=620.0,
                            holding_time=2.0, cooling_rate=200.0)
    relaxed = simulate_pwht(results, mat, schedule, dt=120.0)

    vm0 = results.stress.von_mises
    vm1 = relaxed.stress.von_mises
    assert np.all(vm1 <= vm0 + 1e-6)
    assert "pwht_creep_strain" in relaxed.metadata


# ---------------------------------------------------------------------------
# Singularity check (mesh generation monkeypatched away)
# ---------------------------------------------------------------------------

def test_run_singularity_check(load_mesh, fake_backend, monkeypatch):
    # Coarser synthetic mesh returned in place of the real Gmsh generator.
    coarse = FEMesh(
        nodes=load_mesh.nodes * 2.0,
        elements=load_mesh.elements,
        element_type=ElementType.TRI3,
        node_sets={k: v.copy() for k, v in load_mesh.node_sets.items()},
    )

    def fake_generate_mesh(joint, cfg):
        return coarse

    monkeypatch.setattr("feaweld.mesh.generator.generate_mesh", fake_generate_mesh)

    case = AnalysisCase(geometry=GeometryConfig(base_width=10.0, base_thickness=10.0),
                        load=LoadConfig(axial_force=1000.0))
    mat = _material()
    fine = fake_backend.solve_static(load_mesh, mat, None)

    check = _run_singularity_check(case, fake_backend, object(), fine, mat)
    assert {"n_flagged", "n_singular", "threshold"} <= set(check)
    assert check["threshold"] == case.postprocess.singularity_threshold
    assert check["n_flagged"] >= check["n_singular"] >= 0


# ---------------------------------------------------------------------------
# Fatigue assessment aggregation
# ---------------------------------------------------------------------------

def test_run_fatigue_assessment():
    case = AnalysisCase()  # default sn_curve = IIW_FAT90
    pp = {
        "m1": {"max_stress": 100.0},
        "m2": {"max_stress": 50.0, "fatigue_life": 1e6, "sn_curve_used": "X"},
        "singularity_check": {"n_flagged": 0},
    }
    fatigue = _run_fatigue_assessment(pp, case)

    curve = parse_sn_spec(case.postprocess.sn_curve)
    np.testing.assert_allclose(fatigue["m1"]["life"], curve.life(100.0))

    assert fatigue["m2"]["life"] == 1e6
    assert fatigue["m2"]["sn_curve"] == "X"

    assert "singularity_check" not in fatigue


# ---------------------------------------------------------------------------
# Geometry config passthrough
# ---------------------------------------------------------------------------

def test_build_geometry_butt_groove_passthrough():
    """Butt-weld groove parameters flow from the YAML config to ButtWeld."""
    from feaweld.geometry.joints import ButtWeld
    from feaweld.core.types import JointType

    joint = _build_geometry(GeometryConfig(
        joint_type=JointType.BUTT, base_width=200.0, base_thickness=20.0,
        groove_angle=45.0, root_gap=1.0, penetration="partial",
    ))
    assert isinstance(joint, ButtWeld)
    assert joint.groove_angle == 45.0
    assert joint.root_gap == 1.0
    assert joint.penetration == "partial"
    assert joint.plate_width == 200.0
    assert joint.plate_thickness == 20.0


def test_geometry_config_groove_defaults_match_buttweld():
    """GeometryConfig groove defaults equal the ButtWeld dataclass defaults."""
    from feaweld.geometry.joints import ButtWeld

    cfg = GeometryConfig()
    joint = ButtWeld(plate_width=200.0, plate_thickness=20.0)
    assert cfg.groove_angle == joint.groove_angle
    assert cfg.root_gap == joint.root_gap
    assert cfg.penetration == joint.penetration


def test_geometry_penetration_validator():
    """penetration is restricted to full/partial and case-normalized.

    ButtWeld exact-matches the lowercase "full", so an unvalidated "Full"
    used to silently build the partial-penetration geometry.
    """
    assert GeometryConfig(penetration="Partial").penetration == "partial"
    assert GeometryConfig(penetration="FULL").penetration == "full"
    with pytest.raises(ValueError, match="'full' or 'partial'"):
        GeometryConfig(penetration="none")
    with pytest.raises(ValueError, match="'full' or 'partial'"):
        GeometryConfig(penetration="complete")


# ---------------------------------------------------------------------------
# Weld joint efficiency
# ---------------------------------------------------------------------------

def test_asme_allowable_check_joint_efficiency_scales_limits():
    from feaweld.postprocess.nominal import (
        StressCategorization, asme_allowable_check,
    )

    cat = StressCategorization(membrane=100.0, bending=30.0, peak=10.0,
                               total=140.0, stress_intensity=140.0)
    base = asme_allowable_check(cat, S_m=160.0, S_y=250.0)
    scaled = asme_allowable_check(cat, S_m=160.0, S_y=250.0,
                                  joint_efficiency=0.85)

    e_sm = 0.85 * 160.0
    assert scaled["Pm"]["limit"] == pytest.approx(e_sm)
    assert scaled["PL"]["limit"] == pytest.approx(1.5 * e_sm)
    assert scaled["Pm+Pb"]["limit"] == pytest.approx(1.5 * e_sm)
    assert scaled["PL+Pb+Q"]["limit"] == pytest.approx(
        max(3.0 * e_sm, 2.0 * 250.0)
    )
    assert scaled["joint_efficiency"] == pytest.approx(0.85)

    # Default efficiency (1.0) is unchanged vs the pre-efficiency behavior.
    assert "joint_efficiency" not in base
    assert base["Pm"]["limit"] == pytest.approx(160.0)
    assert base["PL+Pb+Q"]["limit"] == pytest.approx(max(3.0 * 160.0, 500.0))


def test_postprocess_nominal_weld_efficiency_value(grid_plate_mesh, fake_backend):
    """A direct weld_efficiency value scales the four ASME limits."""
    mat = _material()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    base_out = _run_postprocess(StressMethod.NOMINAL, results, grid_plate_mesh,
                                _dispatch_case(), mat)
    eff_case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        postprocess=PostProcessConfig(
            weld_efficiency=WeldEfficiencyConfig(value=0.85),
        ),
    )
    eff_out = _run_postprocess(StressMethod.NOMINAL, results, grid_plate_mesh,
                               eff_case, mat)

    assert eff_out["weld_efficiency"] == pytest.approx(0.85)
    assert "weld_efficiency" not in base_out
    for key in ("Pm", "PL", "Pm+Pb"):
        assert eff_out["asme_checks"][key]["limit"] == pytest.approx(
            0.85 * base_out["asme_checks"][key]["limit"]
        )


def test_postprocess_nominal_weld_efficiency_lookup(grid_plate_mesh, fake_backend):
    """A code-table lookup resolves E through data/weld_efficiency."""
    mat = _material()
    results = fake_backend.solve_static(grid_plate_mesh, mat, None)

    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        postprocess=PostProcessConfig(
            weld_efficiency=WeldEfficiencyConfig(
                standard="ASME_VIII_Div1", joint_type="Type_1",
                examination="Spot_RT",
            ),
        ),
    )
    out = _run_postprocess(StressMethod.NOMINAL, results, grid_plate_mesh,
                           case, mat)
    assert out["weld_efficiency"] == pytest.approx(0.85)


def test_weld_efficiency_config_validators():
    with pytest.raises(ValueError, match="not both"):
        WeldEfficiencyConfig(value=0.8, standard="ASME_VIII_Div1",
                             joint_type="Type_1", examination="Full_RT")
    with pytest.raises(ValueError, match="needs"):
        WeldEfficiencyConfig(standard="ASME_VIII_Div1")
    with pytest.raises(ValueError, match="needs"):
        WeldEfficiencyConfig()


# ---------------------------------------------------------------------------
# parse_sn_spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["IIW_FAT90", "FAT90", "90", "DNV_D", "ASME_ferritic"])
def test_parse_sn_spec_valid(spec):
    from feaweld.core.types import SNCurve
    curve = parse_sn_spec(spec)
    assert isinstance(curve, SNCurve)
    assert curve.segments


@pytest.mark.parametrize("spec", ["garbage", "XYZ_123", "IIW_FAT999"])
def test_parse_sn_spec_invalid(spec):
    with pytest.raises((ValueError, KeyError)):
        parse_sn_spec(spec)


# ---------------------------------------------------------------------------
# Probabilistic case
# ---------------------------------------------------------------------------

def _prob_case(**prob_over) -> AnalysisCase:
    prob = dict(enabled=True, n_samples=64, seed=7)
    prob.update(prob_over)
    return AnalysisCase(
        geometry=GeometryConfig(base_width=100.0, base_thickness=10.0),
        load=LoadConfig(axial_force=50000.0),
        probabilistic=ProbabilisticConfig(**prob),
    )


def test_probabilistic_flags_change_variable_count():
    both = run_probabilistic_case(_prob_case(
        include_material_scatter=True, include_geometric_tolerance=True))
    material_only = run_probabilistic_case(_prob_case(
        include_material_scatter=True, include_geometric_tolerance=False))

    assert len(both["variable_names"]) == 8
    assert len(material_only["variable_names"]) == 4
    assert len(both["variable_names"]) != len(material_only["variable_names"])


def test_probabilistic_shapes_and_legacy_keys():
    case = _prob_case()
    out = run_probabilistic_case(case)
    n = case.probabilistic.n_samples
    k = len(out["variable_names"])
    assert out["samples"].shape == (n, k)
    assert out["results"].shape == (n,)
    for key in ("mean", "std", "cov"):
        assert key in out


def test_probabilistic_sobol():
    out = run_probabilistic_case(_prob_case(sobol=True, sobol_n_base=8))
    assert "sobol" in out
    assert set(out["sobol"]) == {"first_order", "total"}
    names = set(out["variable_names"])
    assert set(out["sobol"]["first_order"]) == names
    assert set(out["sobol"]["total"]) == names


def test_probabilistic_no_sources_raises():
    with pytest.raises(ValueError):
        run_probabilistic_case(_prob_case(
            include_material_scatter=False, include_geometric_tolerance=False))


# ---------------------------------------------------------------------------
# Config backward compatibility
# ---------------------------------------------------------------------------

def test_analysis_case_empty_dict():
    case = AnalysisCase(**{})
    assert case.name == "default"
    # New capability fields are present with defaults.
    assert case.solver.creep_temperature == 550.0
    assert case.thermal.pwht_enabled is False
    assert case.probabilistic.enabled is False


def test_analysis_case_from_old_fields():
    old = {
        "name": "legacy",
        "material": {"base_metal": "A36", "weld_metal": "E70XX"},
        "geometry": {"joint_type": "fillet_t", "base_thickness": 12.0},
        "solver": {"solver_type": "linear_elastic"},
        "load": {"axial_force": 1000.0, "bending_moment": 200.0},
        "postprocess": {"stress_methods": ["hotspot_linear"], "sn_curve": "IIW_FAT80"},
    }
    case = AnalysisCase(**old)
    assert case.name == "legacy"
    assert case.geometry.base_thickness == 12.0
    assert case.postprocess.sn_curve == "IIW_FAT80"
    # Fields not present in the old dict fall back to defaults.
    assert case.thermal.enabled is False
    assert case.probabilistic.sobol is False


def test_workflow_result_warnings_default_and_success():
    result = WorkflowResult(case=AnalysisCase())
    assert result.warnings == []
    assert result.success is True
    # Warnings never flip success; only errors do.
    result.warnings.append("heads up")
    assert result.success is True
    result.errors.append("boom")
    assert result.success is False


# ---------------------------------------------------------------------------
# End-to-end run_analysis wiring (Gmsh + solver replaced by fixtures)
# ---------------------------------------------------------------------------

def _mesh_without_weld_toe() -> FEMesh:
    """A minimal plate mesh carrying only bottom/top node sets."""
    nodes = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
                     dtype=np.float64)
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return FEMesh(
        nodes=nodes, elements=elements, element_type=ElementType.TRI3,
        node_sets={"bottom": np.array([0, 1]), "top": np.array([2, 3])},
    )


def _run_full_analysis(case, mesh, fake_backend, monkeypatch):
    """Run ``run_analysis`` with a fixed mesh and the FakeBackend."""
    monkeypatch.setattr("feaweld.mesh.generator.generate_mesh",
                        lambda joint, cfg, **kwargs: mesh)
    monkeypatch.setattr("feaweld.solver.backend.get_backend",
                        lambda preference="auto": fake_backend)
    return run_analysis(case)


def test_run_analysis_warns_when_weld_toe_absent(fake_backend, monkeypatch):
    """A mesh with no weld_toe set triggers a loud, visible warning."""
    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=10.0),
        postprocess=PostProcessConfig(singularity_check=False,
                                      fatigue_assessment=False),
    )
    result = _run_full_analysis(case, _mesh_without_weld_toe(),
                                fake_backend, monkeypatch)
    assert any("weld_toe" in w and "node 0" in w for w in result.warnings)


def test_run_analysis_no_weld_toe_warning_when_present(fake_backend, monkeypatch,
                                                       grid_plate_mesh):
    """A mesh that carries a weld_toe set produces no such warning."""
    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        postprocess=PostProcessConfig(singularity_check=False,
                                      fatigue_assessment=False),
    )
    result = _run_full_analysis(case, grid_plate_mesh, fake_backend, monkeypatch)
    assert not any("weld_toe node set" in w for w in result.warnings)


def test_singularity_check_runs_for_linear_elastic(fake_backend, monkeypatch,
                                                   grid_plate_mesh):
    """The singularity check runs for a plain linear-elastic solve."""
    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        solver=SolverConfig(solver_type=SolverType.LINEAR_ELASTIC),
        postprocess=PostProcessConfig(fatigue_assessment=False),
    )
    result = _run_full_analysis(case, grid_plate_mesh, fake_backend, monkeypatch)
    assert "singularity_check" in result.postprocess_results


def test_singularity_check_skipped_when_nonlinear(fake_backend, monkeypatch,
                                                  grid_plate_mesh):
    """Plasticity capping makes the coarse baseline incomparable — skip it."""
    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        solver=SolverConfig(solver_type=SolverType.LINEAR_ELASTIC, nonlinear=True),
        postprocess=PostProcessConfig(fatigue_assessment=False),
    )
    result = _run_full_analysis(case, grid_plate_mesh, fake_backend, monkeypatch)
    assert "singularity_check" not in result.postprocess_results


def test_singularity_check_skipped_when_pwht(fake_backend, monkeypatch,
                                             grid_plate_mesh):
    """PWHT relaxation makes the coarse baseline incomparable — skip it."""
    case = AnalysisCase(
        geometry=GeometryConfig(base_width=40.0, base_thickness=20.0),
        thermal=ThermalConfig(pwht_enabled=True, pwht_time_hours=0.05,
                              pwht_heating_rate=3600.0, pwht_cooling_rate=3600.0),
        postprocess=PostProcessConfig(fatigue_assessment=False),
    )
    result = _run_full_analysis(case, grid_plate_mesh, fake_backend, monkeypatch)
    assert "singularity_check" not in result.postprocess_results


# ---------------------------------------------------------------------------
# 3D pipeline (dimension == 3, grid_solid_mesh + FakeBackend)
# ---------------------------------------------------------------------------

def _geometry_3d(**over) -> GeometryConfig:
    kw = dict(base_width=40.0, base_thickness=20.0, web_thickness=10.0,
              weld_leg_size=8.0, dimension=3, length=40.0)
    kw.update(over)
    return GeometryConfig(**kw)


def _case_3d(**over) -> AnalysisCase:
    kw = dict(
        geometry=_geometry_3d(),
        mesh=MeshConfig(global_size=10.0, weld_toe_size=2.5),
        load=LoadConfig(axial_force=2500.0),
    )
    kw.update(over)
    return AnalysisCase(**kw)


def test_geometry_dimension_validator():
    assert GeometryConfig(dimension=2).dimension == 2
    assert AnalysisCase(**{"geometry": {"dimension": 3}}).geometry.dimension == 3
    with pytest.raises(ValueError, match="2 or 3"):
        GeometryConfig(dimension=4)


def test_build_geometry_passes_dimension():
    joint = _build_geometry(_geometry_3d())
    assert joint.dimension == 3
    assert joint.length == 40.0


def test_build_load_case_3d_divides_over_top_face(grid_solid_mesh):
    lc = _build_load_case(LoadConfig(axial_force=2500.0), grid_solid_mesh)
    n_top = len(grid_solid_mesh.node_sets["top"])
    assert n_top == 25
    axial = next(bc for bc in lc.loads if bc.bc_type == LoadType.FORCE)
    np.testing.assert_allclose(axial.values, [2500.0 / 25])
    np.testing.assert_allclose(axial.direction, [0.0, 1.0, 0.0])


def test_build_heat_source_3d_starts_at_toe_line_min_z(grid_solid_mesh):
    case = _case_3d(thermal=ThermalConfig(enabled=True))
    source = _build_heat_source(case, grid_solid_mesh)
    np.testing.assert_allclose(source.start_position, [20.0, 20.0, 0.0])
    np.testing.assert_allclose(source.direction, [0.0, 0.0, 1.0])


class TestRunAnalysis3D:
    def test_full_pipeline(self, fake_backend, monkeypatch, grid_solid_mesh):
        case = _case_3d(
            postprocess=PostProcessConfig(stress_methods=[
                StressMethod.HOTSPOT_LINEAR,
                StressMethod.LINEARIZATION,
                StressMethod.NOMINAL,
                StressMethod.STRUCTURAL_DONG,
            ]),
            fatigue=FatigueConfig(r_ratio=0.1, cycles=1e5),
        )
        result = _run_full_analysis(case, grid_solid_mesh, fake_backend,
                                    monkeypatch)

        # Point/line methods produce finite stresses on the solid mesh.
        for name in ("hotspot_linear", "linearization", "nominal"):
            assert name in result.postprocess_results, result.errors
            assert np.isfinite(result.postprocess_results[name]["max_stress"])

        # Hot spot ran per station along the (single) toe line.
        hs = result.postprocess_results["hotspot_linear"]
        assert hs["critical_line"] == "weld_toe_0"
        assert hs["n_stations"] == 5
        assert len(hs["results"]) == 5

        # structural_dong is 2D-only: lands in errors, not results.
        assert "structural_dong" not in result.postprocess_results
        assert any("2D cross-section" in e for e in result.errors)

        # Singularity check auto-skips in 3D with a warning.
        assert "singularity_check" not in result.postprocess_results
        assert any("singularity check is 2D-only" in w for w in result.warnings)

        # The fatigue assessment consumes the 3D hot-spot reference stress.
        entry = result.fatigue_results["hotspot_linear"]
        assert entry["stress_range"] == pytest.approx(hs["max_stress"])
        for name in ("hotspot_linear", "linearization", "nominal"):
            assert "damage" in result.fatigue_results[name]
            assert result.fatigue_results[name]["damage"] > 0.0

    def test_mesh_size_warnings(self, fake_backend, monkeypatch,
                                grid_solid_mesh):
        case = _case_3d(
            geometry=_geometry_3d(length=5.0),
            mesh=MeshConfig(global_size=10.0, weld_toe_size=0.5),
            postprocess=PostProcessConfig(singularity_check=False,
                                          fatigue_assessment=False),
        )
        result = _run_full_analysis(case, grid_solid_mesh, fake_backend,
                                    monkeypatch)
        assert any("sliver" in w for w in result.warnings)
        assert any("tetrahedra" in w for w in result.warnings)

    def test_no_mesh_size_warnings_for_sane_config(self, fake_backend,
                                                   monkeypatch,
                                                   grid_solid_mesh):
        case = _case_3d(
            postprocess=PostProcessConfig(singularity_check=False,
                                          fatigue_assessment=False),
        )
        result = _run_full_analysis(case, grid_solid_mesh, fake_backend,
                                    monkeypatch)
        assert not any("sliver" in w for w in result.warnings)
        assert not any("tetrahedra" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 3D hot-spot reduction: poorly resolved stations must not win the max
# ---------------------------------------------------------------------------

def _solid_nid(i, j, k, n=5):
    """Node id on the 5x5x5 grid_solid_mesh."""
    return (k * n + j) * n + i


class TestHotspot3DWellResolvedExclusion:
    """Stations flagged ``well_resolved=False`` carry raw nodal values, not
    surface extrapolations; they must not compete for
    ``max_stress``/``critical_line`` while a well-resolved station exists
    (LapJoint upper end-face toe scenario)."""

    @staticmethod
    def _lines(mesh):
        from feaweld.core.types import WeldLineDefinition

        resolved_ids = np.array([_solid_nid(2, 4, k) for k in range(5)],
                                dtype=np.int64)
        edge_ids = np.array([_solid_nid(4, 4, k) for k in range(5)],
                            dtype=np.int64)
        resolved = WeldLineDefinition(
            name="weld_toe_0", node_ids=resolved_ids,
            plate_thickness=20.0,
            normal_direction=np.array([0.0, 1.0, 0.0]),
        )
        # This line sits on the x = 40 top edge with a ±X normal: its
        # 0.4t / 1.0t reference points leave the mesh (they end up above
        # the y = 20 face) and both snap back to the toe node itself, so
        # every station is well_resolved=False and sigma_hs degenerates
        # to the raw corner nodal stress.
        unresolved = WeldLineDefinition(
            name="weld_toe_1", node_ids=edge_ids,
            plate_thickness=20.0,
            normal_direction=np.array([1.0, 0.0, 0.0]),
        )
        return resolved, unresolved, edge_ids

    def test_unresolved_stations_excluded_from_critical_line(
            self, grid_solid_mesh, fake_backend):
        mat = _material()
        case = _case_3d()
        results = fake_backend.solve_static(grid_solid_mesh, mat, None)
        resolved, unresolved, edge_ids = self._lines(grid_solid_mesh)
        # Spike the end-face corner nodes so the bogus raw values would
        # have won the max under the old behavior.
        results.stress.values[edge_ids, 1] = 300.0

        sink: list[str] = []
        out = _run_postprocess(
            StressMethod.HOTSPOT_LINEAR, results, grid_solid_mesh, case,
            mat, weld_lines=[resolved, unresolved], warnings=sink,
        )

        vm = results.stress.von_mises
        # weld_toe_0 extrapolates from the 0.4t / 1.0t surface nodes.
        expected = (1.67 * vm[_solid_nid(3, 4, 0)]
                    - 0.67 * vm[_solid_nid(4, 4, 0)])
        bogus = float(vm[_solid_nid(4, 4, 0)])
        assert expected < bogus  # the raw corner value would have won

        assert out["critical_line"] == "weld_toe_0"
        assert out["max_stress"] == pytest.approx(expected)
        # All stations are still reported (nothing is dropped from
        # results/n_stations, only from the competition) ...
        assert out["n_stations"] == 10
        assert sum(1 for r in out["results"] if not r.well_resolved) == 5
        # ... with the per-line coarse-station warning, but no
        # all-unresolved fallback warning.
        assert any("weld_toe_1" in w and "not well resolved" in w
                   for w in sink)
        assert not any("cannot support" in w for w in sink)

    def test_all_unresolved_falls_back_with_loud_warning(
            self, grid_solid_mesh, fake_backend):
        mat = _material()
        case = _case_3d()
        results = fake_backend.solve_static(grid_solid_mesh, mat, None)
        _, unresolved, edge_ids = self._lines(grid_solid_mesh)

        sink: list[str] = []
        out = _run_postprocess(
            StressMethod.HOTSPOT_LINEAR, results, grid_solid_mesh, case,
            mat, weld_lines=[unresolved], warnings=sink,
        )

        vm = results.stress.von_mises
        # With no well-resolved station anywhere, the raw values are kept
        # (legacy behavior) and a loud warning is emitted.
        assert out["critical_line"] == "weld_toe_1"
        assert out["max_stress"] == pytest.approx(float(vm[edge_ids].max()))
        assert any("cannot support hot-spot surface extrapolation" in w
                   for w in sink)
