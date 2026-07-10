"""Tests for load and boundary-condition definitions in feaweld.core.loads."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.loads import (
    MechanicalLoad,
    PWHTSchedule,
    ThermalLoad,
    WeldingHeatInput,
    moment_to_nodal_forces,
)
from feaweld.core.types import BoundaryCondition, LoadType


# ---------------------------------------------------------------------------
# Moment -> nodal force couple
# ---------------------------------------------------------------------------


class TestMomentToNodalForces:
    def test_forces_sum_to_zero(self, grid_plate_mesh) -> None:
        bc = moment_to_nodal_forces(grid_plate_mesh, "top", moment=5000.0)
        assert np.sum(bc.values) == pytest.approx(0.0, abs=1e-9)

    def test_net_moment_equals_input(self, grid_plate_mesh) -> None:
        M = 5000.0
        bc = moment_to_nodal_forces(grid_plate_mesh, "top", moment=M)
        node_ids = grid_plate_mesh.node_sets["top"]
        x = grid_plate_mesh.nodes[node_ids][:, 0]
        lever = x - x.mean()
        assert float(np.sum(bc.values[:, 1] * lever)) == pytest.approx(M)

    def test_per_node_force_distribution(self, grid_plate_mesh) -> None:
        """Top edge x = [0..40]: levers [-20..20], denom 1000."""
        bc = moment_to_nodal_forces(grid_plate_mesh, "top", moment=5000.0)
        assert bc.values[:, 1] == pytest.approx([-100.0, -50.0, 0.0, 50.0, 100.0])

    def test_output_shape_and_type(self, grid_plate_mesh) -> None:
        bc = moment_to_nodal_forces(grid_plate_mesh, "top", moment=1000.0)
        assert isinstance(bc, BoundaryCondition)
        assert bc.bc_type is LoadType.FORCE
        assert bc.node_set == "top"
        assert bc.values.shape == (5, 3)
        assert np.all(bc.values[:, 0] == 0.0)
        assert np.all(bc.values[:, 2] == 0.0)
        assert bc.direction is None

    def test_collinear_nodes_raise(self, simple_plate_mesh) -> None:
        """weld_toe nodes share x = 10, so axis 0 has no lever arm."""
        with pytest.raises(ValueError, match="same coordinate"):
            moment_to_nodal_forces(simple_plate_mesh, "weld_toe", moment=1000.0)

    def test_custom_axis_and_force_dof(self, simple_plate_mesh) -> None:
        bc = moment_to_nodal_forces(
            simple_plate_mesh, "weld_toe", moment=1000.0, axis=1, force_dof=0,
        )
        # weld_toe y = [0, 10]: levers [-5, 5], denom 50
        assert bc.values[:, 0] == pytest.approx([-100.0, 100.0])
        assert np.all(bc.values[:, 1:] == 0.0)


# ---------------------------------------------------------------------------
# MechanicalLoad
# ---------------------------------------------------------------------------


class TestMechanicalLoad:
    def test_node_set_takes_precedence_over_surface(self) -> None:
        load = MechanicalLoad(
            load_type=LoadType.FORCE, magnitude=100.0,
            surface="top_face", node_set="top_nodes",
        )
        assert load.to_boundary_condition().node_set == "top_nodes"

    def test_surface_used_when_no_node_set(self) -> None:
        load = MechanicalLoad(
            load_type=LoadType.PRESSURE, magnitude=5.0, surface="top_face",
        )
        assert load.to_boundary_condition().node_set == "top_face"

    def test_empty_target_when_neither_given(self) -> None:
        load = MechanicalLoad(load_type=LoadType.FORCE, magnitude=100.0)
        assert load.to_boundary_condition().node_set == ""

    def test_values_and_direction_passthrough(self) -> None:
        direction = np.array([0.0, 1.0, 0.0])
        load = MechanicalLoad(
            load_type=LoadType.FORCE, magnitude=250.0,
            direction=direction, node_set="top",
        )
        bc = load.to_boundary_condition()
        assert bc.bc_type is LoadType.FORCE
        assert bc.values == pytest.approx([250.0])
        assert bc.direction is direction


# ---------------------------------------------------------------------------
# ThermalLoad
# ---------------------------------------------------------------------------


class TestThermalLoad:
    def test_convection_packs_film_and_ambient(self) -> None:
        load = ThermalLoad(
            load_type=LoadType.CONVECTION, value=0.0, surface="outer",
            film_coefficient=25.0, ambient_temperature=30.0,
        )
        bc = load.to_boundary_condition()
        assert bc.values == pytest.approx([25.0, 30.0])

    def test_non_convection_packs_single_value(self) -> None:
        load = ThermalLoad(
            load_type=LoadType.TEMPERATURE, value=620.0, node_set="weld",
        )
        bc = load.to_boundary_condition()
        assert bc.bc_type is LoadType.TEMPERATURE
        assert bc.values == pytest.approx([620.0])

    def test_node_set_takes_precedence_over_surface(self) -> None:
        load = ThermalLoad(
            load_type=LoadType.HEAT_FLUX, value=1e4,
            surface="face", node_set="nodes",
        )
        assert load.to_boundary_condition().node_set == "nodes"

    def test_defaults(self) -> None:
        load = ThermalLoad(load_type=LoadType.CONVECTION, value=0.0)
        assert load.film_coefficient == 0.0
        assert load.ambient_temperature == 20.0
        assert load.to_boundary_condition().values == pytest.approx([0.0, 20.0])


# ---------------------------------------------------------------------------
# WeldingHeatInput
# ---------------------------------------------------------------------------


class TestWeldingHeatInput:
    def test_power_is_eta_v_i(self) -> None:
        heat = WeldingHeatInput(
            voltage=25.0, current=200.0, travel_speed=5.0, efficiency=0.9,
        )
        assert heat.power == pytest.approx(0.9 * 25.0 * 200.0)

    def test_heat_input_is_power_over_speed(self) -> None:
        heat = WeldingHeatInput(
            voltage=25.0, current=200.0, travel_speed=5.0, efficiency=0.9,
        )
        assert heat.heat_input == pytest.approx(heat.power / 5.0)

    def test_default_efficiency(self) -> None:
        heat = WeldingHeatInput(voltage=25.0, current=200.0, travel_speed=5.0)
        assert heat.efficiency == pytest.approx(0.8)
        assert heat.power == pytest.approx(4000.0)

    def test_zero_travel_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="travel_speed must be positive"):
            WeldingHeatInput(voltage=25.0, current=200.0, travel_speed=0.0)

    def test_negative_travel_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="travel_speed must be positive"):
            WeldingHeatInput(voltage=25.0, current=200.0, travel_speed=-1.0)


# ---------------------------------------------------------------------------
# PWHTSchedule
# ---------------------------------------------------------------------------


class TestPWHTSchedule:
    @pytest.fixture
    def schedule(self) -> PWHTSchedule:
        # 180 C/h heating to 620 C, 2 h hold, 90 C/h cooling; from 20 C
        # ambient this gives t_heat = 12000 s, t_hold = 7200 s,
        # t_cool = 24000 s -> 720 samples at dt = 60 s.
        return PWHTSchedule(
            heating_rate=180.0,
            holding_temperature=620.0,
            holding_time=2.0,
            cooling_rate=90.0,
        )

    def test_profile_length_and_start(self, schedule) -> None:
        times, temps = schedule.temperature_profile()
        assert len(times) == 720
        assert len(temps) == 720
        assert times[0] == 0.0
        assert temps[0] == pytest.approx(20.0)

    def test_heating_slope_matches_rate(self, schedule) -> None:
        times, temps = schedule.temperature_profile()
        assert temps[1] - temps[0] == pytest.approx(180.0 / 3600.0 * 60.0)

    def test_cooling_slope_matches_rate(self, schedule) -> None:
        times, temps = schedule.temperature_profile()
        # t = 30000 s is mid-cooling (cooling starts at 19200 s)
        i = int(np.searchsorted(times, 30000.0))
        assert temps[i + 1] - temps[i] == pytest.approx(-90.0 / 3600.0 * 60.0)

    def test_plateau_duration(self, schedule) -> None:
        times, temps = schedule.temperature_profile()
        # 2 h hold at dt = 60 s spans 121 samples including both boundaries
        assert int(np.isclose(temps, 620.0).sum()) == 121

    def test_peak_temperature(self, schedule) -> None:
        _, temps = schedule.temperature_profile()
        assert temps.max() == pytest.approx(620.0)

    def test_final_temperature_near_ambient(self, schedule) -> None:
        _, temps = schedule.temperature_profile()
        # last sample at 43140 s, 60 s before full cooldown to 20 C
        assert temps[-1] == pytest.approx(21.5)

    def test_custom_dt(self, schedule) -> None:
        times, temps = schedule.temperature_profile(dt=30.0)
        assert len(times) == 1440
        assert times[1] - times[0] == pytest.approx(30.0)
        assert temps.max() == pytest.approx(620.0)

    def test_non_positive_heating_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="heating_rate must be positive"):
            PWHTSchedule(
                heating_rate=0.0, holding_temperature=620.0,
                holding_time=2.0, cooling_rate=90.0,
            )

    def test_non_positive_cooling_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="cooling_rate must be positive"):
            PWHTSchedule(
                heating_rate=180.0, holding_temperature=620.0,
                holding_time=2.0, cooling_rate=-10.0,
            )
