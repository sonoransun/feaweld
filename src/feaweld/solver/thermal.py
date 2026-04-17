"""Thermal solver utilities: moving heat sources and element birth/death."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


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
