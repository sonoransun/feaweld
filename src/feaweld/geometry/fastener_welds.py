"""Fastener weld primitives (Track D3): plug, slot, stud, spot.

Each builder emits a true 3D OCC geometry via :mod:`gmsh` with physical
groups for the plate(s) and the weld metal. Imports are gated so the
module loads cleanly when gmsh is absent; calling any constructor's
:meth:`build` method without gmsh raises :class:`ImportError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import-time guard
    import gmsh
    _HAS_GMSH = True
except ImportError:  # pragma: no cover - exercised only without gmsh
    _HAS_GMSH = False
    gmsh = None  # type: ignore


def _require_gmsh() -> None:
    if not _HAS_GMSH:
        raise ImportError(
            "gmsh is required for fastener-weld builders. "
            "Install via the base feaweld dependency (gmsh>=4.11)."
        )


def _ensure_initialized() -> None:
    if not gmsh.isInitialized():
        gmsh.initialize()


def _pick_volume_tags(frag_map_entry: list[tuple[int, int]]) -> list[int]:
    return [t for d, t in frag_map_entry if d == 3]


# ---------------------------------------------------------------------------
# Plug weld
# ---------------------------------------------------------------------------

@dataclass
class PlugWeld:
    """Circular hole in a plate filled by a cylindrical weld."""

    hole_diameter: float
    plate_thickness: float
    fill_depth: float | None = None  # None = full fill
    plate_width: float = 40.0
    plate_length: float = 40.0
    name: str = "plug_weld"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        _ensure_initialized()

        pw = float(self.plate_width)
        pl = float(self.plate_length)
        pt_ = float(self.plate_thickness)
        radius = float(self.hole_diameter) / 2.0
        depth = float(self.fill_depth) if self.fill_depth is not None else pt_
        depth = min(depth, pt_)

        plate = gmsh.model.occ.addBox(-pw / 2.0, -pl / 2.0, 0.0, pw, pl, pt_)
        # Weld cylinder fills from bottom upward by ``depth``.
        weld = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, depth, radius)

        # Cut the plate by the weld cylinder to create the hole.
        cut_out, cut_map = gmsh.model.occ.cut(
            [(3, plate)], [(3, weld)], removeObject=True, removeTool=False
        )
        plate_tags = [t for d, t in cut_out if d == 3]

        obj = [(3, t) for t in plate_tags]
        tool = [(3, weld)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        plate_final = _pick_volume_tags(frag_map[0])
        weld_final = _pick_volume_tags(frag_map[len(obj)])

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["plate"] = gmsh.model.addPhysicalGroup(3, plate_final)
        gmsh.model.setPhysicalName(3, pg["plate"], "plate")
        pg["weld"] = gmsh.model.addPhysicalGroup(3, weld_final)
        gmsh.model.setPhysicalName(3, pg["weld"], "weld")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in plate_final] + [(3, t) for t in weld_final]
        )
        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }


# ---------------------------------------------------------------------------
# Slot weld
# ---------------------------------------------------------------------------

@dataclass
class SlotWeld:
    """Elongated rounded-end slot in a plate filled by weld metal."""

    slot_length: float
    slot_width: float
    plate_thickness: float
    fill_depth: float | None = None
    plate_width: float = 60.0
    plate_length: float = 60.0
    name: str = "slot_weld"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        _ensure_initialized()

        pw = float(self.plate_width)
        pl = float(self.plate_length)
        pt_ = float(self.plate_thickness)
        sl = float(self.slot_length)
        sw = float(self.slot_width)
        radius = sw / 2.0
        depth = float(self.fill_depth) if self.fill_depth is not None else pt_
        depth = min(depth, pt_)

        plate = gmsh.model.occ.addBox(-pw / 2.0, -pl / 2.0, 0.0, pw, pl, pt_)

        # Weld shape: rectangle + two half-cylinders at ends, fused.
        straight_len = max(sl - sw, 0.0)
        rect = gmsh.model.occ.addBox(
            -straight_len / 2.0, -radius, 0.0, straight_len, sw, depth
        )
        cap_left = gmsh.model.occ.addCylinder(
            -straight_len / 2.0, 0.0, 0.0, 0.0, 0.0, depth, radius
        )
        cap_right = gmsh.model.occ.addCylinder(
            straight_len / 2.0, 0.0, 0.0, 0.0, 0.0, depth, radius
        )
        fused, _ = gmsh.model.occ.fuse(
            [(3, rect)], [(3, cap_left), (3, cap_right)]
        )
        weld_vol = fused[0][1]

        cut_out, _ = gmsh.model.occ.cut(
            [(3, plate)], [(3, weld_vol)], removeObject=True, removeTool=False
        )
        plate_tags = [t for d, t in cut_out if d == 3]

        obj = [(3, t) for t in plate_tags]
        tool = [(3, weld_vol)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        plate_final = _pick_volume_tags(frag_map[0])
        weld_final = _pick_volume_tags(frag_map[len(obj)])

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["plate"] = gmsh.model.addPhysicalGroup(3, plate_final)
        gmsh.model.setPhysicalName(3, pg["plate"], "plate")
        pg["weld"] = gmsh.model.addPhysicalGroup(3, weld_final)
        gmsh.model.setPhysicalName(3, pg["weld"], "weld")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in plate_final] + [(3, t) for t in weld_final]
        )
        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }


# ---------------------------------------------------------------------------
# Stud weld
# ---------------------------------------------------------------------------

@dataclass
class StudWeld:
    """Threaded stud welded to the top surface of a plate."""

    stud_diameter: float
    stud_height: float
    plate_thickness: float
    plate_width: float = 20.0
    plate_length: float = 20.0
    name: str = "stud_weld"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        _ensure_initialized()

        pw = float(self.plate_width)
        pl = float(self.plate_length)
        pt_ = float(self.plate_thickness)
        radius = float(self.stud_diameter) / 2.0
        h = float(self.stud_height)

        plate = gmsh.model.occ.addBox(-pw / 2.0, -pl / 2.0, 0.0, pw, pl, pt_)
        stud = gmsh.model.occ.addCylinder(0.0, 0.0, pt_, 0.0, 0.0, h, radius)

        obj = [(3, plate)]
        tool = [(3, stud)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        plate_final = _pick_volume_tags(frag_map[0])
        stud_final = _pick_volume_tags(frag_map[1])

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["plate"] = gmsh.model.addPhysicalGroup(3, plate_final)
        gmsh.model.setPhysicalName(3, pg["plate"], "plate")
        pg["weld"] = gmsh.model.addPhysicalGroup(3, stud_final)
        gmsh.model.setPhysicalName(3, pg["weld"], "weld")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in plate_final] + [(3, t) for t in stud_final]
        )
        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }


# ---------------------------------------------------------------------------
# Spot weld
# ---------------------------------------------------------------------------

@dataclass
class SpotWeld:
    """Resistance spot weld between two overlapping plates (lenticular nugget)."""

    spot_diameter: float
    plate_thickness_1: float
    plate_thickness_2: float
    plate_width: float = 20.0
    plate_length: float = 20.0
    name: str = "spot_weld"

    _physical_groups: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _volume_tags: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> dict[str, Any]:
        _require_gmsh()
        _ensure_initialized()

        pw = float(self.plate_width)
        pl = float(self.plate_length)
        t1 = float(self.plate_thickness_1)
        t2 = float(self.plate_thickness_2)
        radius = float(self.spot_diameter) / 2.0

        # Lower plate sits from z = 0 to z = t1.
        plate1 = gmsh.model.occ.addBox(-pw / 2.0, -pl / 2.0, 0.0, pw, pl, t1)
        # Upper plate sits from z = t1 to z = t1 + t2.
        plate2 = gmsh.model.occ.addBox(-pw / 2.0, -pl / 2.0, t1, pw, pl, t2)

        # Lenticular nugget: approximate as a sphere centred at the interface
        # with radius chosen so the nugget half-height roughly matches
        # min(t1, t2). We use a sphere of radius = spot_diameter/2 and then
        # dilate in z to span both plates.
        sphere = gmsh.model.occ.addSphere(0.0, 0.0, t1, radius)
        z_scale = (t1 + t2) / (2.0 * radius) if radius > 0 else 1.0
        gmsh.model.occ.dilate([(3, sphere)], 0.0, 0.0, t1, 1.0, 1.0, z_scale)

        obj = [(3, plate1), (3, plate2)]
        tool = [(3, sphere)]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        plate1_final = _pick_volume_tags(frag_map[0])
        plate2_final = _pick_volume_tags(frag_map[1])
        weld_final = _pick_volume_tags(frag_map[2])

        gmsh.model.occ.synchronize()

        pg: dict[str, int] = {}
        pg["plate_1"] = gmsh.model.addPhysicalGroup(3, plate1_final)
        gmsh.model.setPhysicalName(3, pg["plate_1"], "plate_1")
        pg["plate_2"] = gmsh.model.addPhysicalGroup(3, plate2_final)
        gmsh.model.setPhysicalName(3, pg["plate_2"], "plate_2")
        pg["weld"] = gmsh.model.addPhysicalGroup(3, weld_final)
        gmsh.model.setPhysicalName(3, pg["weld"], "weld")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in plate1_final]
            + [(3, t) for t in plate2_final]
            + [(3, t) for t in weld_final]
        )
        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }
