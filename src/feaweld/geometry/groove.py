"""Parametric 2D groove cross-section profiles for weld preparations.

Each profile returns the ordered vertices of a closed polygon describing
the *weld-metal* (positive-fill) cross-section in local ``(t, z)``
coordinates, where ``t`` is the in-plane direction perpendicular to the
weld path and ``z`` is the through-thickness direction. The polygon is
symmetric about ``t = 0`` for V/U/J/X grooves and one-sided for K grooves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


_ARC_SAMPLES = 12  # points used to discretize curved root fillets


@dataclass
class GrooveProfile(ABC):
    """Abstract base for a 2D groove cross-section in local ``(t, z)`` coords."""

    plate_thickness: float
    root_gap: float = 0.0

    @abstractmethod
    def cross_section_polygon(self) -> NDArray[np.float64]:
        """Return the closed polygon as an ``(n_pts, 2)`` array of ``(t, z)``."""

    def area(self) -> float:
        """Shoelace area of the closed polygon."""
        pts = self.cross_section_polygon()
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * float(
            abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )


# ---------------------------------------------------------------------------
# V-groove
# ---------------------------------------------------------------------------

@dataclass
class VGroove(GrooveProfile):
    """Single-V groove with a straight root face and symmetric bevels."""

    angle: float = 60.0  # total included angle (degrees)
    root_face: float = 2.0

    def cross_section_polygon(self) -> NDArray[np.float64]:
        t_plate = self.plate_thickness
        rg = self.root_gap
        rf = min(self.root_face, t_plate)
        half = np.radians(self.angle / 2.0)

        bevel_height = max(t_plate - rf, 0.0)
        half_gap = rg / 2.0
        half_top = half_gap + bevel_height * np.tan(half)

        pts = [
            (-half_gap, 0.0),              # bottom-left of root face
            (half_gap, 0.0),               # bottom-right of root face
            (half_gap, rf),                # start of right bevel
            (half_top, t_plate),           # top-right (toe)
            (-half_top, t_plate),          # top-left (toe)
            (-half_gap, rf),               # start of left bevel
        ]
        return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# U-groove
# ---------------------------------------------------------------------------

@dataclass
class UGroove(GrooveProfile):
    """Single-U groove with a circular root radius and vertical-ish bevels."""

    root_radius: float = 4.0
    bevel_angle: float = 10.0  # degrees from vertical above the root radius
    root_face: float = 2.0

    def cross_section_polygon(self) -> NDArray[np.float64]:
        t_plate = self.plate_thickness
        rg = self.root_gap
        rf = min(self.root_face, t_plate)
        r = self.root_radius
        bev = np.radians(self.bevel_angle)
        half_gap = rg / 2.0

        # Circle centre at z = rf + r, offset by half_gap in t so that the
        # arc is tangent to the vertical face of the root gap.
        cz = rf + r

        # Sample the right half of the arc from the bottom (theta = -pi/2)
        # upward until the bevel tangent line takes over. The bevel slope is
        # tan(bev) in t per unit z, so the tangent angle at the arc tip is
        # (pi/2 - bev) measured from +t axis counter-clockwise — meaning the
        # arc parameter runs from -pi/2 to -bev.
        theta = np.linspace(-np.pi / 2.0, -bev, _ARC_SAMPLES)
        right_arc_t = half_gap + r * np.cos(theta) - 0.0
        right_arc_z = cz + r * np.sin(theta)
        # Shift arc so its bottom tangent touches half_gap (not half_gap + r)
        right_arc_t = half_gap + (r + r * np.cos(theta) - r)  # simplifies
        # The above is equivalent to half_gap + r * cos(theta). We want the
        # arc's leftmost point (theta = pi -> cos = -1) to sit at t=half_gap;
        # but our arc spans -pi/2..-bev, i.e. cos in [0, cos(bev)]. Use a
        # shifted centre instead: centre at (half_gap, cz) and radius r makes
        # the arc start at (half_gap, rf) which matches the root face corner.
        right_arc_t = half_gap + r * np.cos(theta) - r * np.cos(-np.pi / 2.0)
        # cos(-pi/2) = 0, so no horizontal shift. Substitute back directly:
        right_arc_t = half_gap + r * np.cos(theta)
        right_arc_z = cz + r * np.sin(theta)

        # Top of right bevel meets the plate surface
        right_tip_t = right_arc_t[-1]
        right_tip_z = right_arc_z[-1]
        top_right_t = right_tip_t + (t_plate - right_tip_z) * np.tan(bev)
        top_right_z = t_plate

        pts: list[tuple[float, float]] = []
        # Start with root face bottom-left → bottom-right
        pts.append((-half_gap, 0.0))
        pts.append((half_gap, 0.0))
        # Right side: up root face, around arc, up bevel, across top, mirror
        pts.append((half_gap, rf))
        for i in range(_ARC_SAMPLES):
            pts.append((float(right_arc_t[i]), float(right_arc_z[i])))
        pts.append((float(top_right_t), float(top_right_z)))
        # Top edge
        pts.append((-float(top_right_t), float(top_right_z)))
        # Left side (mirrored): down bevel, around arc, down root face
        for i in range(_ARC_SAMPLES - 1, -1, -1):
            pts.append((-float(right_arc_t[i]), float(right_arc_z[i])))
        pts.append((-half_gap, rf))
        return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# J-groove
# ---------------------------------------------------------------------------

@dataclass
class JGroove(GrooveProfile):
    """Single-J groove: one plate square, the other with a radiused bevel."""

    bevel_angle: float = 30.0
    root_radius: float = 4.0
    root_face: float = 2.0

    def cross_section_polygon(self) -> NDArray[np.float64]:
        t_plate = self.plate_thickness
        rg = self.root_gap
        rf = min(self.root_face, t_plate)
        r = self.root_radius
        bev = np.radians(self.bevel_angle)
        half_gap = rg / 2.0

        # Left plate is square: weld metal left edge is a straight line
        # from (-half_gap, 0) to (-half_gap, t_plate).

        # Right plate has a J profile: arc tangent to the root face
        # centred at (half_gap + r, rf + r) going from (half_gap, rf)
        # up to the tangent with the bevel line. Arc parameter runs from
        # theta = pi (leftmost) to (pi/2 + bev).
        theta = np.linspace(np.pi, np.pi / 2.0 + bev, _ARC_SAMPLES)
        cz_t = half_gap + r
        cz_z = rf + r
        arc_t = cz_t + r * np.cos(theta)
        arc_z = cz_z + r * np.sin(theta)

        tip_t = arc_t[-1]
        tip_z = arc_z[-1]
        top_right_t = tip_t + (t_plate - tip_z) * np.tan(bev)
        top_right_z = t_plate

        pts: list[tuple[float, float]] = []
        pts.append((-half_gap, 0.0))
        pts.append((half_gap, 0.0))
        pts.append((half_gap, rf))
        for i in range(_ARC_SAMPLES):
            pts.append((float(arc_t[i]), float(arc_z[i])))
        pts.append((float(top_right_t), float(top_right_z)))
        pts.append((-half_gap, float(top_right_z)))
        return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# X-groove (double V)
# ---------------------------------------------------------------------------

@dataclass
class XGroove(GrooveProfile):
    """Double-V symmetric groove with bevels from both plate surfaces."""

    angle_top: float = 60.0
    angle_bottom: float = 60.0
    root_face: float = 3.0

    def cross_section_polygon(self) -> NDArray[np.float64]:
        t_plate = self.plate_thickness
        rg = self.root_gap
        rf = min(self.root_face, t_plate)
        half_top = np.radians(self.angle_top / 2.0)
        half_bot = np.radians(self.angle_bottom / 2.0)
        half_gap = rg / 2.0

        # Root face is centred on z = t/2, height rf
        z_rf_lo = (t_plate - rf) / 2.0
        z_rf_hi = z_rf_lo + rf

        top_height = t_plate - z_rf_hi
        bot_height = z_rf_lo

        top_half_width = half_gap + top_height * np.tan(half_top)
        bot_half_width = half_gap + bot_height * np.tan(half_bot)

        pts = [
            (-bot_half_width, 0.0),
            (bot_half_width, 0.0),
            (half_gap, z_rf_lo),
            (half_gap, z_rf_hi),
            (top_half_width, t_plate),
            (-top_half_width, t_plate),
            (-half_gap, z_rf_hi),
            (-half_gap, z_rf_lo),
        ]
        return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# K-groove (single-bevel)
# ---------------------------------------------------------------------------

@dataclass
class KGroove(GrooveProfile):
    """Single-bevel (K) groove — one plate square, the other beveled."""

    angle: float = 45.0
    root_face: float = 2.0

    def cross_section_polygon(self) -> NDArray[np.float64]:
        t_plate = self.plate_thickness
        rg = self.root_gap
        rf = min(self.root_face, t_plate)
        beta = np.radians(self.angle)
        half_gap = rg / 2.0

        bevel_height = max(t_plate - rf, 0.0)
        right_top_t = half_gap + bevel_height * np.tan(beta)

        # Weld metal: left vertical edge at t=-half_gap, right beveled edge.
        pts = [
            (-half_gap, 0.0),
            (half_gap, 0.0),
            (half_gap, rf),
            (right_top_t, t_plate),
            (-half_gap, t_plate),
        ]
        return np.asarray(pts, dtype=np.float64)
