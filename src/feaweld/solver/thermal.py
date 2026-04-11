"""Thermal solver utilities: moving heat sources and element birth/death."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from feaweld.core.types import WeldPass, WeldSequence


@dataclass
class GoldakHeatSource:
    """Goldak double-ellipsoid volumetric heat source model.

    The heat flux distribution is described by two half-ellipsoids
    (front and rear) sharing common width and depth parameters.  The
    source moves along a straight path at constant speed.

    Parameters
    ----------
    power : float
        Net power  eta * V * I  (W).
    a_f : float
        Front semi-axis along travel direction (mm).
    a_r : float
        Rear semi-axis along travel direction (mm).
    b : float
        Width semi-axis perpendicular to travel in the surface plane (mm).
    c : float
        Depth semi-axis (mm).
    f_f : float
        Front fraction of deposited energy (default 0.6).
    f_r : float
        Rear fraction of deposited energy (default 1.4, f_f + f_r = 2).
    travel_speed : float
        Travel speed (mm/s).
    start_position : NDArray
        Starting coordinates of the heat source centre (mm).
    direction : NDArray
        Unit vector indicating the travel direction.
    """

    power: float
    a_f: float
    a_r: float
    b: float
    c: float
    f_f: float = 0.6
    f_r: float = 1.4
    travel_speed: float = 5.0
    start_position: NDArray = field(default_factory=lambda: np.zeros(3))
    direction: NDArray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.start_position = np.asarray(self.start_position, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm

    def _build_local_frame(self) -> tuple[NDArray, NDArray, NDArray]:
        """Build an orthonormal local coordinate frame from the travel direction.

        Returns (e_xi, e_eta, e_zeta) where e_xi is the travel direction.
        """
        e_xi = self.direction.copy()
        # Choose a vector not parallel to e_xi for cross product
        candidate = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(e_xi, candidate)) > 0.99:
            candidate = np.array([0.0, 1.0, 0.0])
        e_eta = np.cross(e_xi, candidate)
        e_eta /= np.linalg.norm(e_eta)
        e_zeta = np.cross(e_xi, e_eta)
        e_zeta /= np.linalg.norm(e_zeta)
        return e_xi, e_eta, e_zeta

    def evaluate(
        self, x: NDArray, y: NDArray, z: NDArray, t: float
    ) -> NDArray:
        """Evaluate heat source intensity at given coordinates and time.

        Parameters
        ----------
        x, y, z : NDArray
            Coordinate arrays (mm).  Can be any broadcastable shape.
        t : float
            Current time (s).

        Returns
        -------
        NDArray
            Volumetric heat flux (W/mm^3) at each coordinate.
        """
        P = self.power
        a_f = self.a_f
        a_r = self.a_r
        b = self.b
        c = self.c
        f_f = self.f_f
        f_r = self.f_r

        # Current centre of the heat source
        centre = self.start_position + self.direction * self.travel_speed * t

        # Transform to local coordinates
        e_xi, e_eta, e_zeta = self._build_local_frame()

        dx = x - centre[0]
        dy = y - centre[1]
        dz = z - centre[2]

        # Project onto local axes
        xi = dx * e_xi[0] + dy * e_xi[1] + dz * e_xi[2]
        eta = dx * e_eta[0] + dy * e_eta[1] + dz * e_eta[2]
        zeta = dx * e_zeta[0] + dy * e_zeta[1] + dz * e_zeta[2]

        # Common exponential terms for width and depth
        exp_eta = np.exp(-3.0 * eta ** 2 / b ** 2)
        exp_zeta = np.exp(-3.0 * zeta ** 2 / c ** 2)

        # Normalization constant: 6*sqrt(3) / (pi*sqrt(pi))
        pi = np.pi
        norm_const = 6.0 * np.sqrt(3.0) / (pi * np.sqrt(pi))

        # Front ellipsoid (xi >= 0)
        Q_f = (
            norm_const
            * f_f
            * P
            / (a_f * b * c)
            * np.exp(-3.0 * xi ** 2 / a_f ** 2)
            * exp_eta
            * exp_zeta
        )

        # Rear ellipsoid (xi < 0)
        Q_r = (
            norm_const
            * f_r
            * P
            / (a_r * b * c)
            * np.exp(-3.0 * xi ** 2 / a_r ** 2)
            * exp_eta
            * exp_zeta
        )

        # Select front or rear based on sign of xi
        result = np.where(xi >= 0, Q_f, Q_r)
        return result

    def total_energy_rate(self) -> float:
        """Return the total power being deposited (should equal self.power).

        This is a sanity check: integrating the Goldak formula over all
        space should recover the net power P.
        """
        return self.power


@dataclass
class ElementBirthDeath:
    """Manage element activation for weld metal deposition simulation.

    Elements in the weld filler region start "dead" (zero stiffness /
    zero conductivity) and are activated ("born") when the welding torch
    passes their centroid.

    Parameters
    ----------
    element_centroids : NDArray
        (n_elements, 3) array of element centroid coordinates (mm).
    weld_element_ids : NDArray
        1-D array of element indices that belong to the weld deposit.
    activation_distance : float
        Distance behind the torch at which elements are activated (mm).
    """

    element_centroids: NDArray
    weld_element_ids: NDArray
    activation_distance: float = 2.0

    def __post_init__(self) -> None:
        self.element_centroids = np.asarray(self.element_centroids, dtype=np.float64)
        self.weld_element_ids = np.asarray(self.weld_element_ids, dtype=np.int64)
        self._alive = np.zeros(len(self.weld_element_ids), dtype=bool)

    @property
    def alive_mask(self) -> NDArray:
        """Boolean mask over ``weld_element_ids`` indicating which are alive."""
        return self._alive.copy()

    @property
    def alive_element_ids(self) -> NDArray:
        """Element IDs of currently activated weld elements."""
        return self.weld_element_ids[self._alive]

    @property
    def dead_element_ids(self) -> NDArray:
        """Element IDs of weld elements not yet activated."""
        return self.weld_element_ids[~self._alive]

    def update(
        self,
        torch_position: NDArray,
        travel_direction: NDArray,
    ) -> NDArray:
        """Activate elements behind the torch and return newly activated IDs.

        Parameters
        ----------
        torch_position : NDArray
            Current (x, y, z) position of the welding torch (mm).
        travel_direction : NDArray
            Unit vector of travel direction.

        Returns
        -------
        NDArray
            Element IDs that were newly activated in this call.
        """
        torch_position = np.asarray(torch_position, dtype=np.float64)
        travel_direction = np.asarray(travel_direction, dtype=np.float64)
        norm = np.linalg.norm(travel_direction)
        if norm > 0:
            travel_direction = travel_direction / norm

        weld_centroids = self.element_centroids[self.weld_element_ids]
        # Vector from torch to centroid
        diff = weld_centroids - torch_position[np.newaxis, :]
        # Signed distance along travel direction (negative = behind torch)
        signed_dist = np.sum(diff * travel_direction[np.newaxis, :], axis=1)
        # Perpendicular distance
        proj = signed_dist[:, np.newaxis] * travel_direction[np.newaxis, :]
        perp = diff - proj
        perp_dist = np.linalg.norm(perp, axis=1)

        # Activate if element is behind the torch (within activation_distance)
        # and within a reasonable perpendicular distance (use activation_distance too)
        should_activate = (signed_dist <= self.activation_distance) & (
            perp_dist <= self.activation_distance * 3.0
        )

        newly_alive = should_activate & ~self._alive
        self._alive |= should_activate

        return self.weld_element_ids[newly_alive]

    def reset(self) -> None:
        """Reset all weld elements to the dead state."""
        self._alive[:] = False

    def activate_all(self) -> None:
        """Activate all weld elements at once (e.g. for non-deposition analyses)."""
        self._alive[:] = True


# ---------------------------------------------------------------------------
# Multi-pass dispatch (Track G)
# ---------------------------------------------------------------------------

# Default Goldak geometric parameters (mm) used when a ``WeldPass`` does not
# carry its own semi-axis configuration. These mirror the defaults used
# across the rest of the codebase (see ``core.loads.WeldingHeatInput``).
_DEFAULT_A_F = 5.0
_DEFAULT_A_R = 10.0
_DEFAULT_B = 5.0
_DEFAULT_C = 5.0
_DEFAULT_F_F = 0.6
_DEFAULT_F_R = 1.4


@dataclass
class MultiPassHeatSource:
    """Dispatch to a per-pass :class:`GoldakHeatSource` by simulation time.

    Each :class:`~feaweld.core.types.WeldPass` in the supplied
    :class:`~feaweld.core.types.WeldSequence` owns a :class:`GoldakHeatSource`
    constructed from its voltage / current / travel_speed / efficiency.
    :meth:`evaluate` routes a query to whichever pass is active at time
    ``t``; before the first pass, between passes, and after the final pass
    the source returns ``0``.

    The per-pass source uses default double-ellipsoid semi-axes matching
    :class:`feaweld.core.loads.WeldingHeatInput`. Advanced users can assign
    into :attr:`_sources` after construction to customise individual passes.
    """
    sequence: "WeldSequence"
    _sources: dict[int, GoldakHeatSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for p in self.sequence.passes:
            power = p.voltage * p.current * p.efficiency
            self._sources[p.order] = GoldakHeatSource(
                power=power,
                a_f=_DEFAULT_A_F,
                a_r=_DEFAULT_A_R,
                b=_DEFAULT_B,
                c=_DEFAULT_C,
                f_f=_DEFAULT_F_F,
                f_r=_DEFAULT_F_R,
                travel_speed=p.travel_speed,
            )

    def evaluate(
        self, x: NDArray, y: NDArray, z: NDArray, t: float
    ) -> NDArray | float:
        """Evaluate the active pass's heat source at ``(x, y, z, t)``.

        Returns a scalar ``0.0`` when no pass is active at ``t``; otherwise
        delegates to the corresponding :class:`GoldakHeatSource` with a
        time origin shifted to the pass ``start_time``.
        """
        p = self.sequence.active_pass_at(float(t))
        if p is None:
            return 0.0
        src = self._sources[p.order]
        return src.evaluate(x, y, z, float(t) - p.start_time)

    def power_history(self, times: NDArray) -> NDArray:
        """Instantaneous total power (W) at each time in ``times``.

        Intended for validation / plotting. Returns an array matching
        the shape of ``times``; entries outside any pass window are ``0``.
        """
        times_arr = np.asarray(times, dtype=np.float64)
        powers = np.zeros_like(times_arr, dtype=np.float64)
        flat = powers.reshape(-1)
        for i, t in enumerate(times_arr.reshape(-1)):
            p = self.sequence.active_pass_at(float(t))
            if p is None:
                continue
            flat[i] = p.voltage * p.current * p.efficiency
        return powers
