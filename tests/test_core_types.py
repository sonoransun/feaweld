"""Tests for the shared data types in feaweld.core.types."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    JointType,
    LoadCase,
    LoadHistory,
    LoadType,
    Point2D,
    Point3D,
    SNCurve,
    SNSegment,
    SNStandard,
    StressField,
    StressMethod,
    WeldSegment,
    WeldType,
)


# ---------------------------------------------------------------------------
# Case-insensitive string enums
# ---------------------------------------------------------------------------


class TestCaseInsensitiveStrEnum:
    """Lookup semantics shared by all feaweld enums."""

    def test_exact_value_lookup(self) -> None:
        assert JointType("fillet_t") is JointType.FILLET_T

    def test_uppercase_value_lookup(self) -> None:
        assert JointType("FILLET_T") is JointType.FILLET_T

    def test_mixed_case_value_lookup(self) -> None:
        assert JointType("Fillet_T") is JointType.FILLET_T

    def test_member_name_alias(self) -> None:
        """Member names that differ from values are accepted (SED alias)."""
        assert StressMethod("SED") is StressMethod.SED
        assert StressMethod.SED.value == "strain_energy_density"
        assert StressMethod("strain_energy_density") is StressMethod.SED

    def test_member_name_alias_weld_type(self) -> None:
        assert WeldType("groove_full") is WeldType.GROOVE_FULL
        assert WeldType.GROOVE_FULL.value == "groove_full_penetration"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            JointType("bogus_joint")

    def test_non_string_input_raises(self) -> None:
        with pytest.raises(ValueError):
            JointType(42)

    def test_members_compare_equal_to_value_string(self) -> None:
        assert SNStandard.IIW == "iiw"
        assert LoadType.CONVECTION == "convection"


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


class TestPoints:
    def test_point2d_to_array(self) -> None:
        arr = Point2D(1.5, -2.0).to_array()
        assert arr.shape == (2,)
        assert np.allclose(arr, [1.5, -2.0])

    def test_point3d_to_array(self) -> None:
        arr = Point3D(1.0, 2.0, 3.0).to_array()
        assert arr.shape == (3,)
        assert np.allclose(arr, [1.0, 2.0, 3.0])

    def test_points_are_frozen(self) -> None:
        p = Point3D(0.0, 0.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.x = 1.0


class TestWeldSegment:
    def test_length(self) -> None:
        seg = WeldSegment(
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(3.0, 4.0, 0.0),
            leg_size=6.0,
        )
        assert seg.length == pytest.approx(5.0)

    def test_default_weld_type_is_fillet(self) -> None:
        seg = WeldSegment(Point3D(0, 0, 0), Point3D(1, 0, 0), leg_size=6.0)
        assert seg.weld_type is WeldType.FILLET

    def test_fillet_throat_is_leg_over_sqrt2(self) -> None:
        seg = WeldSegment(Point3D(0, 0, 0), Point3D(1, 0, 0), leg_size=8.0)
        assert seg.throat == pytest.approx(8.0 / math.sqrt(2.0))

    def test_groove_throat_equals_leg_size(self) -> None:
        seg = WeldSegment(
            Point3D(0, 0, 0), Point3D(1, 0, 0),
            leg_size=10.0, weld_type=WeldType.GROOVE_FULL,
        )
        assert seg.throat == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# FEMesh validation and properties
# ---------------------------------------------------------------------------


class TestFEMesh:
    def test_3d_nodes_accepted(self, simple_plate_mesh) -> None:
        assert simple_plate_mesh.n_nodes == 4
        assert simple_plate_mesh.n_elements == 2
        assert simple_plate_mesh.ndim == 3

    def test_2d_nodes_accepted(self) -> None:
        mesh = FEMesh(
            nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        assert mesh.ndim == 2
        assert mesh.n_nodes == 3
        assert mesh.n_elements == 1

    def test_bad_node_width_raises(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 2\) or \(n, 3\)"):
            FEMesh(
                nodes=np.zeros((4, 4)),
                elements=np.array([[0, 1, 2]]),
                element_type=ElementType.TRI3,
            )

    def test_1d_nodes_raise(self) -> None:
        with pytest.raises(ValueError, match="nodes"):
            FEMesh(
                nodes=np.zeros(6),
                elements=np.array([[0, 1, 2]]),
                element_type=ElementType.TRI3,
            )

    def test_1d_elements_raise(self) -> None:
        with pytest.raises(ValueError, match="elements must be 2D"):
            FEMesh(
                nodes=np.zeros((3, 3)),
                elements=np.array([0, 1, 2]),
                element_type=ElementType.TRI3,
            )

    def test_default_sets_are_empty_dicts(self) -> None:
        mesh = FEMesh(
            nodes=np.zeros((3, 3)),
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        assert mesh.physical_groups == {}
        assert mesh.node_sets == {}
        assert mesh.element_sets == {}


# ---------------------------------------------------------------------------
# StressField invariants
# ---------------------------------------------------------------------------


class TestStressField:
    def test_wrong_component_count_raises(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
            StressField(values=np.zeros((4, 5)))

    def test_1d_values_raise(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
            StressField(values=np.zeros(6))

    def test_default_location_is_nodes(self) -> None:
        field = StressField(values=np.zeros((2, 6)))
        assert field.location == "nodes"

    def test_uniaxial_von_mises_equals_tresca(self) -> None:
        """For uniaxial stress both measures equal |sigma|."""
        vals = np.zeros((2, 6))
        vals[0, 1] = 100.0
        vals[1, 0] = -80.0
        field = StressField(values=vals)
        assert field.von_mises == pytest.approx([100.0, 80.0])
        assert field.tresca == pytest.approx([100.0, 80.0])

    def test_pure_shear_von_mises(self) -> None:
        vals = np.zeros((1, 6))
        vals[0, 3] = 50.0  # tau_xy
        field = StressField(values=vals)
        assert field.von_mises[0] == pytest.approx(50.0 * math.sqrt(3.0))

    def test_pure_shear_tresca(self) -> None:
        vals = np.zeros((1, 6))
        vals[0, 3] = 50.0
        field = StressField(values=vals)
        assert field.tresca[0] == pytest.approx(100.0)

    def test_hydrostatic_state_gives_zero_deviatoric_measures(self) -> None:
        vals = np.zeros((1, 6))
        vals[0, :3] = 75.0
        field = StressField(values=vals)
        assert field.von_mises[0] == pytest.approx(0.0, abs=1e-9)
        assert field.tresca[0] == pytest.approx(0.0, abs=1e-9)

    def test_principal_sorted_ascending_pure_shear(self) -> None:
        vals = np.zeros((1, 6))
        vals[0, 3] = 50.0
        field = StressField(values=vals)
        assert field.principal.shape == (1, 3)
        assert field.principal[0] == pytest.approx([-50.0, 0.0, 50.0])

    def test_principal_uniaxial(self) -> None:
        vals = np.zeros((1, 6))
        vals[0, 1] = 100.0
        field = StressField(values=vals)
        assert field.principal[0] == pytest.approx([0.0, 0.0, 100.0])

    def test_derived_measures_shape(self) -> None:
        field = StressField(values=np.random.default_rng(0).normal(size=(5, 6)))
        assert field.von_mises.shape == (5,)
        assert field.tresca.shape == (5,)
        assert field.principal.shape == (5, 3)


# ---------------------------------------------------------------------------
# FEAResults
# ---------------------------------------------------------------------------


class TestFEAResults:
    def test_is_transient_false_without_time_steps(self, simple_plate_mesh) -> None:
        results = FEAResults(mesh=simple_plate_mesh)
        assert results.is_transient is False

    def test_is_transient_false_for_single_step(self, simple_plate_mesh) -> None:
        results = FEAResults(mesh=simple_plate_mesh, time_steps=np.array([0.0]))
        assert results.is_transient is False

    def test_is_transient_true_for_multiple_steps(self, simple_plate_mesh) -> None:
        results = FEAResults(
            mesh=simple_plate_mesh, time_steps=np.array([0.0, 1.0, 2.0])
        )
        assert results.is_transient is True

    def test_optional_fields_default_to_none(self, simple_plate_mesh) -> None:
        results = FEAResults(mesh=simple_plate_mesh)
        assert results.displacement is None
        assert results.stress is None
        assert results.strain is None
        assert results.temperature is None
        assert results.nodal_forces is None
        assert results.time_history is None
        assert results.metadata == {}


# ---------------------------------------------------------------------------
# S-N curve life evaluation
# ---------------------------------------------------------------------------


class TestSNCurve:
    """Piecewise life evaluation: first segment whose threshold is met wins."""

    @pytest.fixture
    def two_slope_curve(self) -> SNCurve:
        return SNCurve(
            name="test",
            standard=SNStandard.IIW,
            segments=[
                SNSegment(m=3.0, C=1e12, stress_threshold=50.0),
                SNSegment(m=5.0, C=1e15, stress_threshold=10.0),
            ],
        )

    def test_zero_stress_gives_infinite_life(self, two_slope_curve) -> None:
        assert math.isinf(two_slope_curve.life(0.0))

    def test_negative_stress_gives_infinite_life(self, two_slope_curve) -> None:
        assert math.isinf(two_slope_curve.life(-100.0))

    def test_first_segment_selected_above_its_threshold(self, two_slope_curve) -> None:
        assert two_slope_curve.life(100.0) == pytest.approx(1e12 / 100.0**3)

    def test_second_segment_selected_between_thresholds(self, two_slope_curve) -> None:
        assert two_slope_curve.life(20.0) == pytest.approx(1e15 / 20.0**5)

    def test_threshold_boundary_uses_that_segment(self, two_slope_curve) -> None:
        """stress_range >= threshold is inclusive."""
        assert two_slope_curve.life(50.0) == pytest.approx(1e12 / 50.0**3)
        assert two_slope_curve.life(10.0) == pytest.approx(1e15 / 10.0**5)

    def test_below_all_thresholds_gives_infinite_life(self, two_slope_curve) -> None:
        assert math.isinf(two_slope_curve.life(9.99))

    def test_defaults(self, two_slope_curve) -> None:
        assert two_slope_curve.cutoff_cycles == pytest.approx(1e7)
        assert SNSegment(m=3.0, C=1e12).stress_threshold == 0.0

    def test_single_segment_reference_point(self, simple_sn_curve) -> None:
        """FAT90-like single-slope curve: life(90) == 2e6 by construction."""
        assert simple_sn_curve.life(90.0) == pytest.approx(2e6)


# ---------------------------------------------------------------------------
# Load containers
# ---------------------------------------------------------------------------


class TestLoadContainers:
    def test_boundary_condition_direction_defaults_to_none(self) -> None:
        bc = BoundaryCondition(
            node_set="top",
            bc_type=LoadType.FORCE,
            values=np.array([0.0, 1000.0, 0.0]),
        )
        assert bc.direction is None
        assert bc.bc_type is LoadType.FORCE

    def test_load_case_defaults_are_independent_lists(self) -> None:
        lc1 = LoadCase(name="a")
        lc2 = LoadCase(name="b")
        assert lc1.loads == [] and lc1.constraints == []
        lc1.loads.append("sentinel")
        assert lc2.loads == []

    def test_load_history_optional_fields_default_to_none(self) -> None:
        history = LoadHistory(
            time=np.array([0.0, 1.0]),
            stress_ranges=np.array([100.0, 50.0]),
        )
        assert history.mean_stress is None
        assert history.r_ratio is None
