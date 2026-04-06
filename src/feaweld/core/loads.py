"""Load case and boundary condition definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import BoundaryCondition, LoadCase, LoadType


@dataclass
class MechanicalLoad:
    """A mechanical load specification."""
    load_type: LoadType
    magnitude: float
    direction: NDArray[np.float64] | None = None  # unit vector
    surface: str | None = None  # physical group name for surface loads
    node_set: str | None = None  # node set name for point/nodal loads

    def to_boundary_condition(self) -> BoundaryCondition:
        target = self.node_set or self.surface or ""
        return BoundaryCondition(
            node_set=target,
            bc_type=self.load_type,
            values=np.array([self.magnitude]),
            direction=self.direction,
        )


@dataclass
class ThermalLoad:
    """A thermal load or boundary condition."""
    load_type: LoadType  # TEMPERATURE, HEAT_FLUX, or CONVECTION
    value: float
    surface: str | None = None
    node_set: str | None = None
    # For convection: film coefficient and ambient temperature
    film_coefficient: float = 0.0  # W/(m^2·K)
    ambient_temperature: float = 20.0  # C

    def to_boundary_condition(self) -> BoundaryCondition:
        target = self.node_set or self.surface or ""
        if self.load_type == LoadType.CONVECTION:
            vals = np.array([self.film_coefficient, self.ambient_temperature])
        else:
            vals = np.array([self.value])
        return BoundaryCondition(
            node_set=target,
            bc_type=self.load_type,
            values=vals,
        )


@dataclass
class WeldingHeatInput:
    """Welding process parameters for thermal simulation."""
    voltage: float             # V
    current: float             # A
    travel_speed: float        # mm/s
    efficiency: float = 0.8   # arc efficiency η

    # Goldak double-ellipsoid parameters (mm)
    a_f: float = 5.0   # front semi-axis
    a_r: float = 10.0  # rear semi-axis
    b: float = 5.0     # width semi-axis
    c: float = 5.0     # depth semi-axis
    f_f: float = 0.6   # front fraction
    f_r: float = 1.4   # rear fraction (f_f + f_r = 2.0)

    def __post_init__(self) -> None:
        if self.travel_speed <= 0:
            raise ValueError("travel_speed must be positive")

    @property
    def power(self) -> float:
        """Net heat input power (W)."""
        return self.efficiency * self.voltage * self.current

    @property
    def heat_input(self) -> float:
        """Linear heat input (J/mm)."""
        return self.power / self.travel_speed


@dataclass
class PWHTSchedule:
    """Post-weld heat treatment schedule."""
    heating_rate: float          # C/hour
    holding_temperature: float   # C
    holding_time: float          # hours
    cooling_rate: float          # C/hour

    def __post_init__(self) -> None:
        if self.heating_rate <= 0:
            raise ValueError("heating_rate must be positive")
        if self.cooling_rate <= 0:
            raise ValueError("cooling_rate must be positive")

    def temperature_profile(self, dt: float = 60.0) -> tuple[NDArray, NDArray]:
        """Generate time (s) vs temperature (C) arrays for the PWHT cycle.

        Args:
            dt: time step in seconds
        """
        T_ambient = 20.0

        # Heating phase
        t_heat = (self.holding_temperature - T_ambient) / (self.heating_rate / 3600.0)
        # Holding phase
        t_hold = self.holding_time * 3600.0
        # Cooling phase
        t_cool = (self.holding_temperature - T_ambient) / (self.cooling_rate / 3600.0)

        total_time = t_heat + t_hold + t_cool
        times = np.arange(0, total_time, dt)
        temps = np.empty_like(times)

        for i, t in enumerate(times):
            if t <= t_heat:
                temps[i] = T_ambient + (self.heating_rate / 3600.0) * t
            elif t <= t_heat + t_hold:
                temps[i] = self.holding_temperature
            else:
                temps[i] = self.holding_temperature - (self.cooling_rate / 3600.0) * (t - t_heat - t_hold)

        return times, temps
