"""Parametric weld joint geometry builders using the Gmsh Python API.

Each joint type creates its cross-section geometry using OpenCascade (occ)
kernels, applies boolean fragment operations so regions share interfaces,
and registers physical groups for downstream meshing and analysis.

Joints build a 2D cross-section in the XY plane by default.  Setting
``dimension=3`` extrudes the fragmented section along +Z by ``length``,
registering volume physical groups under the same region names, boundary
face groups (``bottom``/``top`` plus ``front`` at z=0 and ``back`` at
z=length), and weld-toe edge groups (``weld_toe_<i>`` per toe and a
combined ``weld_toe``).
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
    """Abstract base class for all weld joint geometries.

    ``build`` is a template method: subclasses implement ``_build_section``
    (2D surface creation + boolean fragment) and ``_boundary_edge_specs``
    (constant-coordinate extents of the loaded/fixed boundaries), while the
    base class registers physical groups for either the plain 2D section
    (``dimension == 2``, the default) or the extruded 3D solid
    (``dimension == 3``).
    """

    joint_type: JointType

    # ---- template method ----------------------------------------------------

    def build(self, model_name: str = "weld_joint") -> None:
        """Build the geometry in the current Gmsh model.

        Creates the fragmented 2D cross-section via ``_build_section`` and
        registers physical groups.  With ``dimension == 3`` the section is
        first extruded along +Z by ``length``.

        Parameters
        ----------
        model_name:
            Name of the Gmsh model to create.
        """
        _ensure_gmsh_initialized()
        gmsh.model.add(model_name)
        named = self._build_section()
        if getattr(self, "dimension", 2) == 3:
            self._finalize_3d(named)
        else:
            self._finalize_2d(named)

    @abstractmethod
    def _build_section(self) -> list[tuple[str, list[int]]]:
        """Create the fragmented 2D cross-section surfaces.

        Returns
        -------
        list of tuple
            Ordered ``(region_name, surface_tags)`` pairs, one entry per
            physical region of the cross-section.
        """

    @abstractmethod
    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        """Return boundary group specs as ``(name, kwargs)`` pairs.

        Each ``kwargs`` dict holds the constant-coordinate line parameters
        (``axis``, ``value`` and coordinate extents) passed to
        ``_find_edges_on_line`` in 2D, or extended over z in [0, length]
        for face selection on the extruded 3D solid.
        """

    # ---- public interface ----------------------------------------------------

    def get_physical_groups(self) -> dict[str, int]:
        """Return mapping of region name to Gmsh physical group tag."""
        return dict(self._physical_groups)

    @abstractmethod
    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        """Return coordinates of weld toe locations for post-processing."""

    def get_weld_toe_lines(
        self,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Return weld-toe lines as ``((x, y, 0), (x, y, length))`` pairs.

        Each analytic toe point from ``get_weld_toe_points`` sweeps a
        straight line along +Z when the section is extruded.  The endpoint
        pairs are returned for ``dimension == 2`` joints as well (using the
        nominal ``length``), so station sampling along the weld works for
        both cases.

        Returns
        -------
        list of tuple
            One ``(start, end)`` pair of ``(x, y, z)`` coordinates per toe,
            in the same order as ``get_weld_toe_points``.
        """
        length = float(getattr(self, "length", 1.0))
        return [
            ((px, py, 0.0), (px, py, length))
            for px, py, _pz in self.get_weld_toe_points()
        ]

    def get_weld_toe_normals(self) -> list[tuple[float, float, float]]:
        """Return the unit outward plate-surface normal at each weld toe.

        Convention: the normal at a toe points out of the plate surface the
        toe lies on, perpendicular to the weld line (the Z axis), so that
        surface stress-extrapolation paths run along the plate surface away
        from the weld.  Toes on an upward-facing plate surface map to
        ``(0, 1, 0)``, toes on vertical web or side faces map to
        ``(±1, 0, 0)``, and toes on a downward-facing surface map to
        ``(0, -1, 0)``.

        The base implementation assumes every toe lies on an upward-facing
        plate surface.  Subclasses override where toes sit on other faces.

        Returns
        -------
        list of tuple
            One unit ``(nx, ny, nz)`` vector per weld toe, in the same
            order as ``get_weld_toe_points``.
        """
        return [(0.0, 1.0, 0.0) for _ in self.get_weld_toe_points()]

    # ---- finalization ---------------------------------------------------------

    def _finalize_2d(self, named: list[tuple[str, list[int]]]) -> None:
        """Register 2D region groups and 1D boundary edge groups."""
        pg: dict[str, int] = {}
        for name, tags in named:
            pg[name] = gmsh.model.addPhysicalGroup(2, tags)
            gmsh.model.setPhysicalName(2, pg[name], name)

        eps = 1e-6
        for name, spec in self._boundary_edge_specs():
            edges = _find_edges_on_line(eps=eps, **spec)
            if edges:
                pg[name] = gmsh.model.addPhysicalGroup(1, edges)
                gmsh.model.setPhysicalName(1, pg[name], name)

        self._physical_groups = pg

    def _finalize_3d(self, named: list[tuple[str, list[int]]]) -> None:
        """Extrude the section along +Z and register 3D physical groups.

        Registers volume groups under the 2D region names, boundary face
        groups (the joint's edge specs swept over z plus ``front``/``back``)
        and weld-toe edge groups.

        Raises
        ------
        RuntimeError
            If the extrusion returns an unexpected number of volumes or a
            volume's centre of mass does not project onto its source
            surface's centre of mass.
        """
        length = float(self.length)
        flat = [(name, tag) for name, tags in named for tag in tags]
        section_com = {
            tag: gmsh.model.occ.getCenterOfMass(2, tag) for _name, tag in flat
        }

        out = gmsh.model.occ.extrude(
            [(2, tag) for _name, tag in flat], 0.0, 0.0, length
        )
        gmsh.model.occ.synchronize()

        volume_tags = [tag for dim, tag in out if dim == 3]
        if len(volume_tags) != len(flat):
            raise RuntimeError(
                f"Extrusion bookkeeping failed: expected {len(flat)} volumes "
                f"for {len(flat)} section surfaces, got {len(volume_tags)}"
            )

        volume_of_surface: dict[int, int] = {}
        for (name, s_tag), v_tag in zip(flat, volume_tags):
            sx, sy, _sz = section_com[s_tag]
            vx, vy, _vz = gmsh.model.occ.getCenterOfMass(3, v_tag)
            tol = 1e-6 * max(1.0, abs(sx), abs(sy))
            if abs(vx - sx) > tol or abs(vy - sy) > tol:
                raise RuntimeError(
                    f"Extrusion bookkeeping failed: volume {v_tag} centre "
                    f"({vx:.6g}, {vy:.6g}) does not match source surface "
                    f"{s_tag} centre ({sx:.6g}, {sy:.6g}) for region '{name}'"
                )
            volume_of_surface[s_tag] = v_tag

        pg: dict[str, int] = {}
        for name, tags in named:
            vols = [volume_of_surface[t] for t in tags]
            pg[name] = gmsh.model.addPhysicalGroup(3, vols)
            gmsh.model.setPhysicalName(3, pg[name], name)

        eps = 1e-6
        for name, spec in self._boundary_edge_specs():
            faces = _find_faces_on_plane(eps=eps, zmin=0.0, zmax=length, **spec)
            if faces:
                pg[name] = gmsh.model.addPhysicalGroup(2, faces)
                gmsh.model.setPhysicalName(2, pg[name], name)

        for name, z_value in (("front", 0.0), ("back", length)):
            faces = _find_faces_on_plane(axis="z", value=z_value, eps=eps)
            if faces:
                pg[name] = gmsh.model.addPhysicalGroup(2, faces)
                gmsh.model.setPhysicalName(2, pg[name], name)

        all_toe_edges: list[int] = []
        for i, (p0, p1) in enumerate(self.get_weld_toe_lines()):
            edges = _find_edges_along_segment(p0, p1, eps=eps)
            if edges:
                name = f"weld_toe_{i}"
                pg[name] = gmsh.model.addPhysicalGroup(1, edges)
                gmsh.model.setPhysicalName(1, pg[name], name)
                all_toe_edges.extend(edges)
        if all_toe_edges:
            unique_edges = sorted(set(all_toe_edges))
            pg["weld_toe"] = gmsh.model.addPhysicalGroup(1, unique_edges)
            gmsh.model.setPhysicalName(1, pg["weld_toe"], "weld_toe")

        self._physical_groups = pg


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

    The 2D cross-section lies in the XY plane.  With ``dimension=3`` the
    section is extruded along Z by ``length`` (with the default
    ``dimension=2`` the *length* is only nominal).
    """

    base_width: float
    base_thickness: float
    web_height: float
    web_thickness: float
    weld_leg_size: float
    length: float = 1.0
    dimension: int = 2

    joint_type: JointType = JointType.FILLET_T

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    # ---- section geometry ----------------------------------------------------

    def _build_section(self) -> list[tuple[str, list[int]]]:
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

        return [
            ("base_plate", _map_tags(0)),
            ("web", _map_tags(1)),
            ("weld_left", _map_tags(2)),
            ("weld_right", _map_tags(3)),
        ]

    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        web_x0 = (self.base_width - self.web_thickness) / 2.0
        return [
            # Bottom edge of base plate (y = 0)
            ("bottom", {"axis": "y", "value": 0.0,
                        "xmin": 0.0, "xmax": self.base_width}),
            # Top edge of web (y = bt + wh)
            ("top", {"axis": "y",
                     "value": self.base_thickness + self.web_height,
                     "xmin": web_x0, "xmax": web_x0 + self.web_thickness}),
        ]

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

    def get_weld_toe_normals(self) -> list[tuple[float, float, float]]:
        """Base-plate toes face +Y; web-side toes face outward along ±X."""
        return [
            (0.0, 1.0, 0.0),     # left toe on base-plate top surface
            (-1.0, 0.0, 0.0),    # left toe on the web's left face
            (1.0, 0.0, 0.0),     # right toe on the web's right face
            (0.0, 1.0, 0.0),     # right toe on base-plate top surface
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
    dimension: int = 2

    joint_type: JointType = JointType.BUTT

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def _groove_geometry(self) -> tuple[float, float]:
        """Return (groove half-width at top surface, joint centre x)."""
        half_angle = np.radians(self.groove_angle / 2.0)
        groove_half_top = (self.root_gap / 2.0
                           + self.plate_thickness * np.tan(half_angle))
        cx = self.plate_width + groove_half_top
        return groove_half_top, cx

    def _build_section(self) -> list[tuple[str, list[int]]]:
        pt_ = self.plate_thickness
        rg = self.root_gap
        half_angle = np.radians(self.groove_angle / 2.0)
        groove_half_top, cx = self._groove_geometry()

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

        return [
            ("plate_left", _mt(0)),
            ("plate_right", _mt(1)),
            ("weld_metal", _mt(2)),
        ]

    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        _groove_half_top, cx = self._groove_geometry()
        return [
            ("bottom", {"axis": "y", "value": 0.0,
                        "xmin": 0.0, "xmax": 2.0 * cx}),
            ("top", {"axis": "y", "value": self.plate_thickness,
                     "xmin": 0.0, "xmax": 2.0 * cx}),
        ]

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        pt_ = self.plate_thickness
        groove_half_top, cx = self._groove_geometry()
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
    dimension: int = 2

    joint_type: JointType = JointType.LAP

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def _build_section(self) -> list[tuple[str, list[int]]]:
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

        return [
            ("plate_lower", _mt(0)),
            ("plate_upper", _mt(1)),
            ("weld", _mt(2)),
        ]

    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        return [
            ("bottom", {"axis": "y", "value": 0.0,
                        "xmin": 0.0, "xmax": self.overlap_length * 2.0}),
        ]

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        pt_ = self.plate_thickness
        ol = self.overlap_length
        ls = self.weld_leg_size
        return [
            (ol - ls, pt_, 0.0),    # toe on lower plate surface
            (ol, pt_ + ls, 0.0),    # toe on upper plate side
        ]

    def get_weld_toe_normals(self) -> list[tuple[float, float, float]]:
        """Lower-plate toe faces +Y; upper-plate edge toe faces -X."""
        return [
            (0.0, 1.0, 0.0),     # toe on the lower plate's top surface
            (-1.0, 0.0, 0.0),    # toe on the upper plate's left end face
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
    dimension: int = 2
    plate_length: float = 50.0  # arm length for both plates

    joint_type: JointType = JointType.CORNER

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def _build_section(self) -> list[tuple[str, list[int]]]:
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

        return [
            ("plate_horizontal", _mt(0)),
            ("plate_vertical", _mt(1)),
            ("weld", _mt(2)),
        ]

    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        return [
            ("bottom", {"axis": "y", "value": 0.0,
                        "xmin": 0.0, "xmax": self.plate_length}),
        ]

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        th = self.plate_thickness_h
        tv = self.plate_thickness_v
        ls = self.weld_leg_size
        pl = self.plate_length
        return [
            (pl - tv - ls, th, 0.0),    # toe on horizontal plate
            (pl - tv, th + ls, 0.0),    # toe on vertical plate
        ]

    def get_weld_toe_normals(self) -> list[tuple[float, float, float]]:
        """Horizontal-plate toe faces +Y; vertical-plate toe faces -X."""
        return [
            (0.0, 1.0, 0.0),     # toe on the horizontal plate's top surface
            (-1.0, 0.0, 0.0),    # toe on the vertical plate's inner face
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
    dimension: int = 2
    base_width: float = 100.0
    web_height: float = 50.0

    joint_type: JointType = JointType.CRUCIFORM

    def __post_init__(self) -> None:
        self._physical_groups: dict[str, int] = {}

    def _build_section(self) -> list[tuple[str, list[int]]]:
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

        return [
            ("base_plate", _mt(0)),
            ("web_upper", _mt(1)),
            ("web_lower", _mt(2)),
            ("weld_upper_left", _mt(3)),
            ("weld_upper_right", _mt(4)),
            ("weld_lower_left", _mt(5)),
            ("weld_lower_right", _mt(6)),
        ]

    def _boundary_edge_specs(self) -> list[tuple[str, dict]]:
        web_x0 = (self.base_width - self.web_thickness) / 2.0
        return [
            ("top", {"axis": "y",
                     "value": self.plate_thickness + self.web_height,
                     "xmin": web_x0, "xmax": web_x0 + self.web_thickness}),
            ("bottom", {"axis": "y", "value": -self.web_height,
                        "xmin": web_x0, "xmax": web_x0 + self.web_thickness}),
        ]

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

    def get_weld_toe_normals(self) -> list[tuple[float, float, float]]:
        """Base-plate toes face ±Y (top/underside); web toes face ±X."""
        return [
            # Upper weld toes
            (0.0, 1.0, 0.0),     # on base-plate top surface
            (-1.0, 0.0, 0.0),    # on upper web's left face
            (1.0, 0.0, 0.0),     # on upper web's right face
            (0.0, 1.0, 0.0),     # on base-plate top surface
            # Lower weld toes
            (0.0, -1.0, 0.0),    # on base-plate bottom surface
            (-1.0, 0.0, 0.0),    # on lower web's left face
            (1.0, 0.0, 0.0),     # on lower web's right face
            (0.0, -1.0, 0.0),    # on base-plate bottom surface
        ]


# ---------------------------------------------------------------------------
# Entity-finding helpers (used after synchronize)
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


def _find_faces_on_plane(
    axis: str,
    value: float,
    eps: float = 1e-6,
    xmin: float = -1e10,
    xmax: float = 1e10,
    ymin: float = -1e10,
    ymax: float = 1e10,
    zmin: float = -1e10,
    zmax: float = 1e10,
) -> list[int]:
    """Return tags of 2D surfaces lying on a constant-coordinate plane.

    A surface matches when its bounding box is degenerate at *value* along
    *axis* (within tolerance *eps*) and its bounding-box centre lies inside
    the given coordinate extents.  3D analogue of ``_find_edges_on_line``.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    faces: list[int] = []
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[axis_idx] - value) >= eps or abs(bb[axis_idx + 3] - value) >= eps:
            continue
        cx = (bb[0] + bb[3]) / 2.0
        cy = (bb[1] + bb[4]) / 2.0
        cz = (bb[2] + bb[5]) / 2.0
        if (xmin - eps <= cx <= xmax + eps
                and ymin - eps <= cy <= ymax + eps
                and zmin - eps <= cz <= zmax + eps):
            faces.append(tag)
    return faces


def _find_edges_along_segment(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    eps: float = 1e-6,
) -> list[int]:
    """Return tags of 1D curves lying on the axis-aligned segment *p0*-*p1*.

    A curve matches when its bounding box is contained in the (degenerate)
    bounding box of the segment inflated by *eps* -- i.e. it sits exactly at
    the segment's constant coordinates and spans only within the segment's
    extent.  Used to locate extruded weld-toe lines.
    """
    lo = [min(a, b) - eps for a, b in zip(p0, p1)]
    hi = [max(a, b) + eps for a, b in zip(p0, p1)]
    edges: list[int] = []
    for dim, tag in gmsh.model.getEntities(1):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if all(lo[k] <= bb[k] and bb[k + 3] <= hi[k] for k in range(3)):
            edges.append(tag)
    return edges
