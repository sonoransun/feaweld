"""True 3D solid weld joint builders (Track D2).

Builds prismatic 3D geometries for butt and fillet-T joints using OCC
primitives exposed by :mod:`gmsh`. The module imports cleanly when gmsh
is absent; calling any builder without gmsh raises a clear ImportError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from feaweld.geometry.groove import GrooveProfile, VGroove
from feaweld.geometry.weld_path import WeldPath

try:  # pragma: no cover - import-time guard
    import gmsh
    _HAS_GMSH = True
except ImportError:  # pragma: no cover - exercised only without gmsh
    _HAS_GMSH = False
    gmsh = None  # type: ignore


def _require_gmsh() -> None:
    if not _HAS_GMSH:
        raise ImportError(
            "gmsh is required for 3D joint builders. "
            "Install via the base feaweld dependency (gmsh>=4.11)."
        )


def _ensure_initialized() -> None:
    if not gmsh.isInitialized():
        gmsh.initialize()


def _add_prism_from_polygon(
    polygon_xy: np.ndarray,
    z0: float,
    length: float,
    *,
    swap_yz: bool = False,
) -> int:
    """Add a prismatic volume by extruding a 2D polygon along +z.

    ``polygon_xy`` is an ``(n, 2)`` array of ``(t, z_local)`` groove
    coordinates. When ``swap_yz=True`` the polygon is placed in the
    XY plane with ``t -> x`` and ``z_local -> y``, and extruded along
    ``+z`` over ``length`` starting at ``z = z0``.
    """
    pt_tags: list[int] = []
    if swap_yz:
        for t_val, z_val in polygon_xy:
            pt_tags.append(
                gmsh.model.occ.addPoint(float(t_val), float(z_val), float(z0))
            )
    else:
        for x_val, y_val in polygon_xy:
            pt_tags.append(
                gmsh.model.occ.addPoint(float(x_val), float(y_val), float(z0))
            )

    n = len(pt_tags)
    line_tags = [
        gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % n]) for i in range(n)
    ]
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    surf = gmsh.model.occ.addPlaneSurface([loop])
    out = gmsh.model.occ.extrude([(2, surf)], 0.0, 0.0, float(length))
    # Return the resulting 3D volume tag (first (3, tag) entry).
    for dim, tag in out:
        if dim == 3:
            return tag
    raise RuntimeError("extrude did not produce a 3D volume")


# ---------------------------------------------------------------------------
# Volumetric butt joint
# ---------------------------------------------------------------------------

@dataclass
class VolumetricButtJoint:
    """True 3D butt joint with a parametric groove cross-section.

    The two plates sit on either side of the weld, the groove void is
    cut out of them, and a weld-metal volume fills the groove. The
    resulting model contains three volumes registered as physical groups
    ``"plate_left"``, ``"plate_right"``, and ``"weld_metal"``.
    """

    plate_width: float
    plate_thickness: float
    length: float
    groove: GrooveProfile
    path: WeldPath | None = None
    name: str = "volumetric_butt"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        if self.path is not None:
            raise NotImplementedError(
                "curved volumetric butt pending track D4"
            )
        _ensure_initialized()

        pw = float(self.plate_width)
        pt_ = float(self.plate_thickness)
        length = float(self.length)

        poly = self.groove.cross_section_polygon()
        t_min = float(poly[:, 0].min())
        t_max = float(poly[:, 0].max())
        half_top = max(abs(t_min), abs(t_max))

        # Left plate: from x = -(pw + half_top) to x = -half_top
        left_box = gmsh.model.occ.addBox(
            -(pw + half_top), 0.0, 0.0, pw, pt_, length
        )
        # Right plate: from x = half_top to x = half_top + pw
        right_box = gmsh.model.occ.addBox(
            half_top, 0.0, 0.0, pw, pt_, length
        )

        # Weld metal: extrude the groove polygon along z.
        # Polygon is in (t, z_local) coordinates, with t -> x and z_local -> y.
        weld_vol = _add_prism_from_polygon(poly, z0=0.0, length=length, swap_yz=True)

        # Optional boolean fragment to share faces between plates and weld.
        # This is not strictly necessary here (the groove carves nothing from
        # the plates in this layout) but ensures coincident interfaces share
        # mesh nodes downstream.
        obj = [(3, left_box), (3, right_box)]
        tool = [(3, weld_vol)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        left_tags = [t for d, t in frag_map[0] if d == 3]
        right_tags = [t for d, t in frag_map[1] if d == 3]
        weld_tags = [t for d, t in frag_map[2] if d == 3]

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["plate_left"] = gmsh.model.addPhysicalGroup(3, left_tags)
        gmsh.model.setPhysicalName(3, pg["plate_left"], "plate_left")
        pg["plate_right"] = gmsh.model.addPhysicalGroup(3, right_tags)
        gmsh.model.setPhysicalName(3, pg["plate_right"], "plate_right")
        pg["weld_metal"] = gmsh.model.addPhysicalGroup(3, weld_tags)
        gmsh.model.setPhysicalName(3, pg["weld_metal"], "weld_metal")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in left_tags]
            + [(3, t) for t in right_tags]
            + [(3, t) for t in weld_tags]
        )

        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        poly = self.groove.cross_section_polygon()
        pt_ = float(self.plate_thickness)
        # Weld toes are the polygon vertices at z == plate_thickness.
        toe_mask = np.isclose(poly[:, 1], pt_)
        xs = poly[toe_mask, 0]
        zs_start = 0.0
        zs_end = float(self.length)
        pts: list[tuple[float, float, float]] = []
        for x in xs:
            pts.append((float(x), pt_, zs_start))
            pts.append((float(x), pt_, zs_end))
        return pts


# ---------------------------------------------------------------------------
# Volumetric fillet T-joint
# ---------------------------------------------------------------------------

@dataclass
class VolumetricFilletTJoint:
    """True 3D fillet-T joint.

    Builds a base plate plus a web plate with a triangular fillet-weld
    prism on each side of the web. Only the straight-path variant is
    implemented in D2; a non-None :class:`WeldPath` raises
    :class:`NotImplementedError`.
    """

    base_width: float
    base_thickness: float
    web_height: float
    web_thickness: float
    weld_leg_size: float
    length: float
    path: WeldPath | None = None
    name: str = "volumetric_fillet_t"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        if self.path is not None:
            raise NotImplementedError(
                "curved volumetric fillet pending track D4"
            )
        _ensure_initialized()

        bw = float(self.base_width)
        bt = float(self.base_thickness)
        wh = float(self.web_height)
        wt = float(self.web_thickness)
        ls = float(self.weld_leg_size)
        length = float(self.length)

        web_x0 = (bw - wt) / 2.0

        base = gmsh.model.occ.addBox(0.0, 0.0, 0.0, bw, bt, length)
        web = gmsh.model.occ.addBox(web_x0, bt, 0.0, wt, wh, length)

        # Left fillet weld: right-triangle prism
        left_poly = np.array(
            [
                [web_x0 - ls, bt],
                [web_x0, bt],
                [web_x0, bt + ls],
            ],
            dtype=np.float64,
        )
        weld_left = _add_prism_from_polygon(
            left_poly, z0=0.0, length=length, swap_yz=False
        )

        right_poly = np.array(
            [
                [web_x0 + wt, bt + ls],
                [web_x0 + wt, bt],
                [web_x0 + wt + ls, bt],
            ],
            dtype=np.float64,
        )
        weld_right = _add_prism_from_polygon(
            right_poly, z0=0.0, length=length, swap_yz=False
        )

        obj = [(3, base), (3, web)]
        tool = [(3, weld_left), (3, weld_right)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        base_tags = [t for d, t in frag_map[0] if d == 3]
        web_tags = [t for d, t in frag_map[1] if d == 3]
        wl_tags = [t for d, t in frag_map[2] if d == 3]
        wr_tags = [t for d, t in frag_map[3] if d == 3]

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["base_plate"] = gmsh.model.addPhysicalGroup(3, base_tags)
        gmsh.model.setPhysicalName(3, pg["base_plate"], "base_plate")
        pg["web"] = gmsh.model.addPhysicalGroup(3, web_tags)
        gmsh.model.setPhysicalName(3, pg["web"], "web")
        pg["weld_left"] = gmsh.model.addPhysicalGroup(3, wl_tags)
        gmsh.model.setPhysicalName(3, pg["weld_left"], "weld_left")
        pg["weld_right"] = gmsh.model.addPhysicalGroup(3, wr_tags)
        gmsh.model.setPhysicalName(3, pg["weld_right"], "weld_right")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in base_tags]
            + [(3, t) for t in web_tags]
            + [(3, t) for t in wl_tags]
            + [(3, t) for t in wr_tags]
        )

        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }

    def get_weld_toe_points(self) -> list[tuple[float, float, float]]:
        bw = float(self.base_width)
        bt = float(self.base_thickness)
        wt = float(self.web_thickness)
        ls = float(self.weld_leg_size)
        length = float(self.length)
        web_x0 = (bw - wt) / 2.0
        ur_x = web_x0 + wt

        toe_xy = [
            (web_x0 - ls, bt),          # left toe on base plate
            (web_x0, bt + ls),          # left toe on web
            (ur_x, bt + ls),            # right toe on web
            (ur_x + ls, bt),            # right toe on base plate
        ]
        pts: list[tuple[float, float, float]] = []
        for x, y in toe_xy:
            pts.append((x, y, 0.0))
            pts.append((x, y, length))
        return pts
