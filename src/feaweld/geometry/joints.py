"""Parametric weld joint geometry builders using the Gmsh Python API.

Each joint type creates its cross-section geometry using OpenCascade (occ)
kernels, applies boolean fragment operations so regions share interfaces,
and registers physical groups for downstream meshing and analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from feaweld.core.types import JointType, Point3D


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class JointGeometry(ABC):
    """Abstract base class for all weld joint geometries."""

    joint_type: JointType

    @abstractmethod
    def build(self, model_name: str = "weld_joint") -> None:
        """Build the geometry in the current Gmsh model."""

    @abstractmethod
    def get_physical_groups(self) -> dict[str, int]:
        """Return mapping of region name to Gmsh physical group tag."""

    @abstractmethod
    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        """Return coordinates of weld toe locations for post-processing."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_gmsh_initialized() -> None:
    """Initialize Gmsh if it is not already running."""
    if not gmsh.is_initialized():
        gmsh.initialize()


def _add_rectangle(x0: float, y0: float, dx: float, dy: float) -> int:
    """Add an OCC rectangle and return its surface tag."""
    return gmsh.model.occ.addRectangle(x0, y0, 0.0, dx, dy)


def _add_polygon(points: list[tuple[float, float]]) -> int:
    """Add a 2D polygon (closed loop) via OCC and return the surface tag.

    *points* is a list of (x, y) vertices.  The polygon is closed
    automatically (last point connected back to first).
    """
    n = len(points)
    pt_tags = [gmsh.model.occ.addPoint(px, py, 0.0) for px, py in points]
    line_tags = []
    for i in range(n):
        line_tags.append(
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % n])
        )
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    return gmsh.model.occ.addPlaneSurface([loop])


def _fragment_surfaces(surf_tags: list[int]) -> list[list[tuple[int, int]]]:
    """Boolean-fragment a list of 2D surfaces so they share interfaces.

    Returns the list of *objectDimTags* and *objectDimTagsMap* from OCC.
    """
    obj = [(2, surf_tags[0])]
    tool = [(2, t) for t in surf_tags[1:]]
    out_dim_tags, out_map = gmsh.model.occ.fragment(obj, tool)
    gmsh.model.occ.synchronize()
    return out_dim_tags, out_map


# ---------------------------------------------------------------------------
# Fillet T-Joint
# ---------------------------------------------------------------------------

@dataclass
class FilletTJoint(JointGeometry):
    """T-joint with fillet welds at the web-to-base-plate junction.

    The 2D cross-section lies in the XY plane.  An optional extrusion along
    Z produces a 3D model (default *length* = 1.0 keeps it quasi-2D).
    """

    base_width: float
    base_thickness: float
    web_height: float
    web_thickness: float
    weld_leg_size: float
    length: float = 1.0

    joint_type: JointType = JointType.FILLET_T

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    # ---- public interface --------------------------------------------------

    def build(self, model_name: str = "weld_joint") -> None:
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)

        bw = self.base_width
        bt = self.base_thickness
        wh = self.web_height
        wt = self.web_thickness
        ls = self.weld_leg_size

        # 1. Base plate
        base = _add_rectangle(0, 0, bw, bt)

        # 2. Web -- centred on base plate
        web_x0 = (bw - wt) / 2.0
        web = _add_rectangle(web_x0, bt, wt, wh)

        # 3. Left fillet weld (right-triangle)
        lw_pts = [
            (web_x0 - ls, bt),          # toe on base plate surface
            (web_x0, bt),               # root (where web meets base)
            (web_x0, bt + ls),          # toe on web side
        ]
        weld_left = _add_polygon(lw_pts)

        # 4. Right fillet weld
        rw_x = web_x0 + wt
        rw_pts = [
            (rw_x, bt + ls),            # toe on web side
            (rw_x, bt),                 # root
            (rw_x + ls, bt),            # toe on base surface
        ]
        weld_right = _add_polygon(rw_pts)

        # 5. Boolean fragment to produce shared interfaces
        all_surfs = [base, web, weld_left, weld_right]
        out_tags, out_map = _fragment_surfaces(all_surfs)

        # After fragment the original tags are invalidated.  The *out_map*
        # gives the new (dim, tag) pairs corresponding to each original input
        # object in order.
        def _map_tags(idx: int) -> list[int]:
            return [t for d, t in out_map[idx] if d == 2]

        base_tags = _map_tags(0)
        web_tags = _map_tags(1)
        wl_tags = _map_tags(2)
        wr_tags = _map_tags(3)

        # 6. Physical groups for volumes / surfaces
        pg = {}
        pg["base_plate"] = gmsh.model.addPhysicalGroup(2, base_tags)
        gmsh.model.setPhysicalName(2, pg["base_plate"], "base_plate")

        pg["web"] = gmsh.model.addPhysicalGroup(2, web_tags)
        gmsh.model.setPhysicalName(2, pg["web"], "web")

        pg["weld_left"] = gmsh.model.addPhysicalGroup(2, wl_tags)
        gmsh.model.setPhysicalName(2, pg["weld_left"], "weld_left")

        pg["weld_right"] = gmsh.model.addPhysicalGroup(2, wr_tags)
        gmsh.model.setPhysicalName(2, pg["weld_right"], "weld_right")

        # 7. Boundary physical groups (1D edges)
        #    Bottom edge of base plate (y = 0)
        eps = 1e-6
        bottom_edges = _find_edges_on_line(axis="y", value=0.0, eps=eps,
                                           xmin=0.0, xmax=bw)
        if bottom_edges:
            pg["bottom"] = gmsh.model.addPhysicalGroup(1, bottom_edges)
            gmsh.model.setPhysicalName(1, pg["bottom"], "bottom")

        #    Top edge of web (y = bt + wh)
        top_y = bt + wh
        top_edges = _find_edges_on_line(axis="y", value=top_y, eps=eps,
                                        xmin=web_x0, xmax=web_x0 + wt)
        if top_edges:
            pg["top"] = gmsh.model.addPhysicalGroup(1, top_edges)
            gmsh.model.setPhysicalName(1, pg["top"], "top")

        self._physical_groups = pg

    def get_physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        bw = self.base_width
        bt = self.base_thickness
        wt = self.web_thickness
        ls = self.weld_leg_size
        web_x0 = (bw - wt) / 2.0

        return [
            (web_x0 - ls, bt, 0.0),          # left toe on base surface
            (web_x0, bt + ls, 0.0),           # left toe on web
            (web_x0 + wt, bt + ls, 0.0),      # right toe on web
            (web_x0 + wt + ls, bt, 0.0),      # right toe on base surface
        ]


# ---------------------------------------------------------------------------
# Butt Weld
# ---------------------------------------------------------------------------

@dataclass
class ButtWeld(JointGeometry):
    """V-groove butt weld between two plates."""

    plate_width: float
    plate_thickness: float
    groove_angle: float = 60.0   # total included angle in degrees
    root_gap: float = 2.0
    penetration: str = "full"    # "full" or "partial"
    length: float = 1.0

    joint_type: JointType = JointType.BUTT

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def build(self, model_name: str = "weld_joint") -> None:
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)

        pw = self.plate_width
        pt_ = self.plate_thickness
        rg = self.root_gap
        half_angle = np.radians(self.groove_angle / 2.0)

        # Groove half-width at top surface
        groove_half_top = (rg / 2.0) + pt_ * np.tan(half_angle)

        # Centre of the joint is at x = pw + groove_half_top
        cx = pw + groove_half_top

        # Left plate -- from x=0 to the left side of the groove
        left_plate_pts = [
            (0.0, 0.0),
            (cx - rg / 2.0, 0.0),            # bottom-right (root)
            (cx - groove_half_top, pt_),     # top-right (groove edge)
            (0.0, pt_),
        ]
        left_plate = _add_polygon(left_plate_pts)

        # Right plate
        right_plate_pts = [
            (cx + rg / 2.0, 0.0),
            (2.0 * cx, 0.0),
            (2.0 * cx, pt_),
            (cx + groove_half_top, pt_),
        ]
        right_plate = _add_polygon(right_plate_pts)

        # Weld metal -- fills the V groove (and root gap)
        if self.penetration == "full":
            weld_pts = [
                (cx - rg / 2.0, 0.0),
                (cx + rg / 2.0, 0.0),
                (cx + groove_half_top, pt_),
                (cx - groove_half_top, pt_),
            ]
        else:
            # Partial penetration: weld fills top 2/3 of thickness
            pen_depth = pt_ * 2.0 / 3.0
            pen_y = pt_ - pen_depth
            pen_half = (rg / 2.0) + pen_depth * np.tan(half_angle)
            weld_pts = [
                (cx - rg / 2.0, pen_y),
                (cx + rg / 2.0, pen_y),
                (cx + groove_half_top, pt_),
                (cx - groove_half_top, pt_),
            ]
            # Adjust plate shapes -- for simplicity we still use the full
            # groove cut and treat unfused region as part of plates.
        weld_metal = _add_polygon(weld_pts)

        all_surfs = [left_plate, right_plate, weld_metal]
        out_tags, out_map = _fragment_surfaces(all_surfs)

        def _mt(idx: int) -> list[int]:
            return [t for d, t in out_map[idx] if d == 2]

        pg: dict[str, int] = {}
        pg["plate_left"] = gmsh.model.addPhysicalGroup(2, _mt(0))
        gmsh.model.setPhysicalName(2, pg["plate_left"], "plate_left")

        pg["plate_right"] = gmsh.model.addPhysicalGroup(2, _mt(1))
        gmsh.model.setPhysicalName(2, pg["plate_right"], "plate_right")

        pg["weld_metal"] = gmsh.model.addPhysicalGroup(2, _mt(2))
        gmsh.model.setPhysicalName(2, pg["weld_metal"], "weld_metal")

        # Boundaries
        eps = 1e-6
        bottom = _find_edges_on_line(axis="y", value=0.0, eps=eps,
                                     xmin=0.0, xmax=2.0 * cx)
        if bottom:
            pg["bottom"] = gmsh.model.addPhysicalGroup(1, bottom)
            gmsh.model.setPhysicalName(1, pg["bottom"], "bottom")
        top = _find_edges_on_line(axis="y", value=pt_, eps=eps,
                                  xmin=0.0, xmax=2.0 * cx)
        if top:
            pg["top"] = gmsh.model.addPhysicalGroup(1, top)
            gmsh.model.setPhysicalName(1, pg["top"], "top")

        self._physical_groups = pg

    def get_physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        pt_ = self.plate_thickness
        rg = self.root_gap
        half_angle = np.radians(self.groove_angle / 2.0)
        groove_half_top = (rg / 2.0) + pt_ * np.tan(half_angle)
        pw = self.plate_width
        cx = pw + groove_half_top
        return [
            (cx - groove_half_top, pt_, 0.0),
            (cx + groove_half_top, pt_, 0.0),
        ]


# ---------------------------------------------------------------------------
# Lap Joint
# ---------------------------------------------------------------------------

@dataclass
class LapJoint(JointGeometry):
    """Two overlapping plates with fillet welds at the overlap edge."""

    plate_thickness: float
    overlap_length: float
    weld_leg_size: float
    length: float = 1.0

    joint_type: JointType = JointType.LAP

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def build(self, model_name: str = "weld_joint") -> None:
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)

        pt_ = self.plate_thickness
        ol = self.overlap_length
        ls = self.weld_leg_size

        # Total width occupied by geometry
        total_w = ol * 2.0  # each plate extends ol beyond centre

        # Lower plate: (0, 0) to (ol + ol, pt_)
        lower = _add_rectangle(0.0, 0.0, total_w, pt_)

        # Upper plate sits on top of the right half of the lower plate
        upper = _add_rectangle(ol, pt_, ol, pt_)

        # Fillet weld at the left edge of the upper plate where it meets
        # the lower plate surface
        weld_pts = [
            (ol - ls, pt_),
            (ol, pt_),
            (ol, pt_ + ls),
        ]
        weld = _add_polygon(weld_pts)

        all_surfs = [lower, upper, weld]
        out_tags, out_map = _fragment_surfaces(all_surfs)

        def _mt(idx: int) -> list[int]:
            return [t for d, t in out_map[idx] if d == 2]

        pg: dict[str, int] = {}
        pg["plate_lower"] = gmsh.model.addPhysicalGroup(2, _mt(0))
        gmsh.model.setPhysicalName(2, pg["plate_lower"], "plate_lower")

        pg["plate_upper"] = gmsh.model.addPhysicalGroup(2, _mt(1))
        gmsh.model.setPhysicalName(2, pg["plate_upper"], "plate_upper")

        pg["weld"] = gmsh.model.addPhysicalGroup(2, _mt(2))
        gmsh.model.setPhysicalName(2, pg["weld"], "weld")

        eps = 1e-6
        bottom = _find_edges_on_line(axis="y", value=0.0, eps=eps,
                                     xmin=0.0, xmax=total_w)
        if bottom:
            pg["bottom"] = gmsh.model.addPhysicalGroup(1, bottom)
            gmsh.model.setPhysicalName(1, pg["bottom"], "bottom")

        self._physical_groups = pg

    def get_physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        pt_ = self.plate_thickness
        ol = self.overlap_length
        ls = self.weld_leg_size
        return [
            (ol - ls, pt_, 0.0),    # toe on lower plate surface
            (ol, pt_ + ls, 0.0),    # toe on upper plate side
        ]


# ---------------------------------------------------------------------------
# Corner Joint
# ---------------------------------------------------------------------------

@dataclass
class CornerJoint(JointGeometry):
    """L-shaped corner joint with a fillet weld at the junction."""

    plate_thickness_h: float   # horizontal plate
    plate_thickness_v: float   # vertical plate
    weld_leg_size: float
    length: float = 1.0
    plate_length: float = 50.0  # arm length for both plates

    joint_type: JointType = JointType.CORNER

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def build(self, model_name: str = "weld_joint") -> None:
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)

        th = self.plate_thickness_h
        tv = self.plate_thickness_v
        ls = self.weld_leg_size
        pl = self.plate_length

        # Horizontal plate: sits at the bottom
        horiz = _add_rectangle(0.0, 0.0, pl, th)

        # Vertical plate: rises from the right end of horizontal plate
        vert = _add_rectangle(pl - tv, th, tv, pl)

        # Fillet weld at the inner corner
        weld_pts = [
            (pl - tv - ls, th),      # toe on horizontal surface
            (pl - tv, th),           # root
            (pl - tv, th + ls),      # toe on vertical surface
        ]
        weld = _add_polygon(weld_pts)

        all_surfs = [horiz, vert, weld]
        out_tags, out_map = _fragment_surfaces(all_surfs)

        def _mt(idx: int) -> list[int]:
            return [t for d, t in out_map[idx] if d == 2]

        pg: dict[str, int] = {}
        pg["plate_horizontal"] = gmsh.model.addPhysicalGroup(2, _mt(0))
        gmsh.model.setPhysicalName(2, pg["plate_horizontal"], "plate_horizontal")

        pg["plate_vertical"] = gmsh.model.addPhysicalGroup(2, _mt(1))
        gmsh.model.setPhysicalName(2, pg["plate_vertical"], "plate_vertical")

        pg["weld"] = gmsh.model.addPhysicalGroup(2, _mt(2))
        gmsh.model.setPhysicalName(2, pg["weld"], "weld")

        eps = 1e-6
        bottom = _find_edges_on_line(axis="y", value=0.0, eps=eps,
                                     xmin=0.0, xmax=pl)
        if bottom:
            pg["bottom"] = gmsh.model.addPhysicalGroup(1, bottom)
            gmsh.model.setPhysicalName(1, pg["bottom"], "bottom")

        self._physical_groups = pg

    def get_physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        th = self.plate_thickness_h
        tv = self.plate_thickness_v
        ls = self.weld_leg_size
        pl = self.plate_length
        return [
            (pl - tv - ls, th, 0.0),    # toe on horizontal plate
            (pl - tv, th + ls, 0.0),    # toe on vertical plate
        ]


# ---------------------------------------------------------------------------
# Cruciform Joint
# ---------------------------------------------------------------------------

@dataclass
class CruciformJoint(JointGeometry):
    """Cruciform joint -- web passes through base plate with 4 fillet welds."""

    plate_thickness: float
    web_thickness: float
    weld_leg_size: float
    length: float = 1.0
    base_width: float = 100.0
    web_height: float = 50.0

    joint_type: JointType = JointType.CRUCIFORM

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def build(self, model_name: str = "weld_joint") -> None:
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)

        bw = self.base_width
        pt_ = self.plate_thickness
        wt = self.web_thickness
        wh = self.web_height
        ls = self.weld_leg_size

        web_x0 = (bw - wt) / 2.0

        # Base plate
        base = _add_rectangle(0.0, 0.0, bw, pt_)

        # Upper web
        upper_web = _add_rectangle(web_x0, pt_, wt, wh)

        # Lower web (mirror below base plate)
        lower_web = _add_rectangle(web_x0, -wh, wt, wh)

        # 4 fillet welds
        # Upper-left
        ul_pts = [
            (web_x0 - ls, pt_),
            (web_x0, pt_),
            (web_x0, pt_ + ls),
        ]
        weld_ul = _add_polygon(ul_pts)

        # Upper-right
        ur_x = web_x0 + wt
        ur_pts = [
            (ur_x, pt_ + ls),
            (ur_x, pt_),
            (ur_x + ls, pt_),
        ]
        weld_ur = _add_polygon(ur_pts)

        # Lower-left
        ll_pts = [
            (web_x0, -(ls)),
            (web_x0, 0.0),
            (web_x0 - ls, 0.0),
        ]
        weld_ll = _add_polygon(ll_pts)

        # Lower-right
        lr_pts = [
            (ur_x + ls, 0.0),
            (ur_x, 0.0),
            (ur_x, -(ls)),
        ]
        weld_lr = _add_polygon(lr_pts)

        all_surfs = [base, upper_web, lower_web,
                     weld_ul, weld_ur, weld_ll, weld_lr]
        out_tags, out_map = _fragment_surfaces(all_surfs)

        def _mt(idx: int) -> list[int]:
            return [t for d, t in out_map[idx] if d == 2]

        pg: dict[str, int] = {}
        pg["base_plate"] = gmsh.model.addPhysicalGroup(2, _mt(0))
        gmsh.model.setPhysicalName(2, pg["base_plate"], "base_plate")

        pg["web_upper"] = gmsh.model.addPhysicalGroup(2, _mt(1))
        gmsh.model.setPhysicalName(2, pg["web_upper"], "web_upper")

        pg["web_lower"] = gmsh.model.addPhysicalGroup(2, _mt(2))
        gmsh.model.setPhysicalName(2, pg["web_lower"], "web_lower")

        pg["weld_upper_left"] = gmsh.model.addPhysicalGroup(2, _mt(3))
        gmsh.model.setPhysicalName(2, pg["weld_upper_left"], "weld_upper_left")

        pg["weld_upper_right"] = gmsh.model.addPhysicalGroup(2, _mt(4))
        gmsh.model.setPhysicalName(2, pg["weld_upper_right"], "weld_upper_right")

        pg["weld_lower_left"] = gmsh.model.addPhysicalGroup(2, _mt(5))
        gmsh.model.setPhysicalName(2, pg["weld_lower_left"], "weld_lower_left")

        pg["weld_lower_right"] = gmsh.model.addPhysicalGroup(2, _mt(6))
        gmsh.model.setPhysicalName(2, pg["weld_lower_right"], "weld_lower_right")

        # Boundaries
        eps = 1e-6
        top_y = pt_ + wh
        top_edges = _find_edges_on_line(axis="y", value=top_y, eps=eps,
                                        xmin=web_x0, xmax=web_x0 + wt)
        if top_edges:
            pg["top"] = gmsh.model.addPhysicalGroup(1, top_edges)
            gmsh.model.setPhysicalName(1, pg["top"], "top")

        bot_y = -wh
        bot_edges = _find_edges_on_line(axis="y", value=bot_y, eps=eps,
                                        xmin=web_x0, xmax=web_x0 + wt)
        if bot_edges:
            pg["bottom"] = gmsh.model.addPhysicalGroup(1, bot_edges)
            gmsh.model.setPhysicalName(1, pg["bottom"], "bottom")

        self._physical_groups = pg

    def get_physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        bw = self.base_width
        pt_ = self.plate_thickness
        wt = self.web_thickness
        ls = self.weld_leg_size
        web_x0 = (bw - wt) / 2.0
        ur_x = web_x0 + wt
        return [
            # Upper weld toes
            (web_x0 - ls, pt_, 0.0),
            (web_x0, pt_ + ls, 0.0),
            (ur_x, pt_ + ls, 0.0),
            (ur_x + ls, pt_, 0.0),
            # Lower weld toes
            (web_x0 - ls, 0.0, 0.0),
            (web_x0, -ls, 0.0),
            (ur_x, -ls, 0.0),
            (ur_x + ls, 0.0, 0.0),
        ]


# ---------------------------------------------------------------------------
# Edge-finding helper (used after synchronize)
# ---------------------------------------------------------------------------

def _find_edges_on_line(
    axis: str,
    value: float,
    eps: float = 1e-6,
    xmin: float = -1e10,
    xmax: float = 1e10,
    ymin: float = -1e10,
    ymax: float = 1e10,
) -> list[int]:
    """Return tags of 1D curves whose bounding box centre lies on the
    specified constant-coordinate line within tolerance *eps*.
    """
    edges: list[int] = []
    for dim, tag in gmsh.model.getEntities(1):
        bb = gmsh.model.getBoundingBox(dim, tag)  # xmin,ymin,zmin,xmax,ymax,zmax
        cx = (bb[0] + bb[3]) / 2.0
        cy = (bb[1] + bb[4]) / 2.0
        if axis == "y":
            if (abs(bb[1] - value) < eps and abs(bb[4] - value) < eps
                    and cx >= xmin - eps and cx <= xmax + eps):
                edges.append(tag)
        elif axis == "x":
            if (abs(bb[0] - value) < eps and abs(bb[3] - value) < eps
                    and cy >= ymin - eps and cy <= ymax + eps):
                edges.append(tag)
    return edges
