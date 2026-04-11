"""Gmsh OCC boolean insertion of weld defects into an existing solid.

All public functions in this module operate on the **current** Gmsh model.
They accept an integer ``parent_volume_tag`` identifying the solid to carve
defects out of, plus a defect dataclass from :mod:`feaweld.defects.types`,
and return a record dict describing the (possibly updated) parent tag and
any auxiliary tags that were produced.

Gmsh is an optional runtime dependency.  If ``gmsh`` is unavailable every
function raises :class:`ImportError` on first call, allowing the rest of
:mod:`feaweld.defects` to import cleanly in environments without Gmsh.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from feaweld.defects.types import (
    ClusterPorosity,
    Defect,
    LackOfFusionDefect,
    PoreDefect,
    RootGapDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
)

try:  # pragma: no cover - exercised only when gmsh is installed
    import gmsh

    _HAS_GMSH = True
except ImportError:  # pragma: no cover - exercised only when gmsh is absent
    _HAS_GMSH = False
    gmsh = None  # type: ignore[assignment]


def _require_gmsh() -> None:
    if not _HAS_GMSH:
        raise ImportError(
            "gmsh is required for defect insertion. "
            "Install the base feaweld dependency (gmsh>=4.11)."
        )


# ---------------------------------------------------------------------------
# Individual defect insertion functions
# ---------------------------------------------------------------------------


def insert_pore(
    parent_volume_tag: int, defect: PoreDefect
) -> dict[str, Any]:
    """Insert a spherical pore into the parent volume via boolean cut.

    Returns a dict with the (possibly new) parent volume tag and a
    ``defect_type`` label for the insertion record.
    """
    _require_gmsh()
    sphere_tag = gmsh.model.occ.addSphere(
        defect.center.x,
        defect.center.y,
        defect.center.z,
        defect.diameter / 2.0,
    )
    cut_out, _ = gmsh.model.occ.cut(
        [(3, parent_volume_tag)],
        [(3, sphere_tag)],
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    new_tag = cut_out[0][1] if cut_out else parent_volume_tag
    return {"parent_tag": new_tag, "defect_type": "pore"}


def insert_slag_inclusion(
    parent_volume_tag: int, defect: SlagInclusion
) -> dict[str, Any]:
    """Insert an ellipsoidal slag inclusion as a separate material region.

    A unit sphere is dilated to the requested semi-axes, rotated by the
    defect's Euler angles, then *fragmented* with the parent so that the
    inclusion becomes its own volume (a second material) rather than a
    hollow cavity.  The parent volume and the inclusion are returned in
    the record so callers can assign physical groups.
    """
    _require_gmsh()
    cx, cy, cz = defect.center.x, defect.center.y, defect.center.z
    a, b, c = defect.semi_axes

    sphere_tag = gmsh.model.occ.addSphere(cx, cy, cz, 1.0)
    gmsh.model.occ.dilate([(3, sphere_tag)], cx, cy, cz, a, b, c)

    # Apply Euler rotations (roll-pitch-yaw about x-, y-, z-axes) about
    # the centre of the inclusion.
    rx, ry, rz = defect.orientation_euler
    if rx:
        gmsh.model.occ.rotate([(3, sphere_tag)], cx, cy, cz, 1.0, 0.0, 0.0, rx)
    if ry:
        gmsh.model.occ.rotate([(3, sphere_tag)], cx, cy, cz, 0.0, 1.0, 0.0, ry)
    if rz:
        gmsh.model.occ.rotate([(3, sphere_tag)], cx, cy, cz, 0.0, 0.0, 1.0, rz)

    frag_out, _ = gmsh.model.occ.fragment(
        [(3, parent_volume_tag)], [(3, sphere_tag)]
    )
    gmsh.model.occ.synchronize()

    volumes = [t for (d, t) in frag_out if d == 3]
    parent_tag = volumes[0] if volumes else parent_volume_tag
    inclusion_tag = volumes[1] if len(volumes) > 1 else sphere_tag
    return {
        "parent_tag": parent_tag,
        "inclusion_tag": inclusion_tag,
        "defect_type": "slag",
    }


def insert_cluster_porosity(
    parent_volume_tag: int, cluster: ClusterPorosity, seed: int = 0
) -> dict[str, Any]:
    """Insert a cluster of spherical pores around the cluster centre."""
    _require_gmsh()
    rng = np.random.default_rng(seed)
    current_tag = parent_volume_tag
    pore_records: list[dict[str, Any]] = []
    for _ in range(cluster.n_pores):
        # Sample a random position inside the bounding sphere (rejection
        # sampling: draw in the unit cube, keep if inside the unit ball).
        while True:
            pt = rng.uniform(-1.0, 1.0, size=3)
            if float(np.dot(pt, pt)) <= 1.0:
                break
        pt *= cluster.radius
        diameter = float(
            np.clip(rng.normal(cluster.size_mean, cluster.size_std), 1e-3, None)
        )
        from feaweld.core.types import Point3D  # local to avoid import cycle

        pore = PoreDefect(
            center=Point3D(
                cluster.center.x + float(pt[0]),
                cluster.center.y + float(pt[1]),
                cluster.center.z + float(pt[2]),
            ),
            diameter=diameter,
        )
        rec = insert_pore(current_tag, pore)
        current_tag = rec["parent_tag"]
        pore_records.append(rec)
    return {
        "parent_tag": current_tag,
        "defect_type": "cluster_porosity",
        "pore_records": pore_records,
    }


def insert_undercut(
    parent_volume_tag: int, defect: UndercutDefect
) -> dict[str, Any]:
    """Insert a linear undercut groove by cutting an axis-aligned box.

    The box is built along the local ``start -> end`` direction with the
    cross-section chosen from the undercut profile (V or U) and then
    rotated + translated into place.  The resulting prism is cut from
    the parent volume.
    """
    _require_gmsh()
    start = np.array([defect.start.x, defect.start.y, defect.start.z])
    end = np.array([defect.end.x, defect.end.y, defect.end.z])
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        return {"parent_tag": parent_volume_tag, "defect_type": "undercut"}

    depth = defect.depth
    if defect.profile == "V":
        width = depth
    else:  # "U"
        width = depth * 2.0

    # Create the box at the origin along +x, then rotate/translate onto
    # the segment.  The box is centred in y (width) and tucked below the
    # weld surface in z so the top face sits at z=0 in local coords.
    box_tag = gmsh.model.occ.addBox(
        0.0, -width / 2.0, -depth, length, width, depth
    )

    # Rotate from +x onto the segment direction.
    direction = axis / length
    x_axis = np.array([1.0, 0.0, 0.0])
    cross = np.cross(x_axis, direction)
    dot = float(np.dot(x_axis, direction))
    sin = float(np.linalg.norm(cross))
    if sin > 1e-12:
        angle = float(np.arctan2(sin, dot))
        axis_n = cross / sin
        gmsh.model.occ.rotate(
            [(3, box_tag)], 0.0, 0.0, 0.0,
            float(axis_n[0]), float(axis_n[1]), float(axis_n[2]), angle,
        )
    elif dot < 0.0:
        # Anti-parallel: rotate 180 about z
        gmsh.model.occ.rotate(
            [(3, box_tag)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(np.pi)
        )

    gmsh.model.occ.translate(
        [(3, box_tag)], float(start[0]), float(start[1]), float(start[2])
    )

    cut_out, _ = gmsh.model.occ.cut(
        [(3, parent_volume_tag)], [(3, box_tag)], removeTool=True
    )
    gmsh.model.occ.synchronize()
    new_tag = cut_out[0][1] if cut_out else parent_volume_tag
    return {"parent_tag": new_tag, "defect_type": "undercut"}


def insert_lack_of_fusion(
    parent_volume_tag: int, defect: LackOfFusionDefect
) -> dict[str, Any]:
    """Insert a planar lack-of-fusion flaw via disc fragmentation.

    A disc of the in-plane extents is created at the flaw origin with the
    requested normal, then *fragmented* with the parent volume so that the
    plane becomes an internal surface (which the mesher can duplicate into
    a zero-thickness crack when required).
    """
    _require_gmsh()
    ox, oy, oz = (
        defect.plane_origin.x,
        defect.plane_origin.y,
        defect.plane_origin.z,
    )
    radius = 0.5 * max(defect.extent_u, defect.extent_v, 1e-6)

    # Disc is created in the XY plane; rotate it so +z matches the normal.
    disk_tag = gmsh.model.occ.addDisk(
        ox, oy, oz, defect.extent_u / 2.0, defect.extent_v / 2.0
    )
    normal = np.asarray(defect.plane_normal, dtype=float)
    n_norm = float(np.linalg.norm(normal))
    if n_norm > 1e-12:
        normal = normal / n_norm
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, normal)
        sin = float(np.linalg.norm(cross))
        dot = float(np.dot(z_axis, normal))
        if sin > 1e-12:
            angle = float(np.arctan2(sin, dot))
            axis_n = cross / sin
            gmsh.model.occ.rotate(
                [(2, disk_tag)], ox, oy, oz,
                float(axis_n[0]), float(axis_n[1]), float(axis_n[2]), angle,
            )
        elif dot < 0.0:
            gmsh.model.occ.rotate(
                [(2, disk_tag)], ox, oy, oz, 1.0, 0.0, 0.0, float(np.pi)
            )

    frag_out, _ = gmsh.model.occ.fragment(
        [(3, parent_volume_tag)], [(2, disk_tag)]
    )
    gmsh.model.occ.synchronize()

    volumes = [t for (d, t) in frag_out if d == 3]
    surfaces = [t for (d, t) in frag_out if d == 2]
    parent_tag = volumes[0] if volumes else parent_volume_tag
    return {
        "parent_tag": parent_tag,
        "lof_surface_tags": surfaces,
        "defect_type": "lack_of_fusion",
        "_radius": radius,  # diagnostic, not used downstream
    }


def insert_surface_crack(
    parent_volume_tag: int, defect: SurfaceCrack
) -> dict[str, Any]:
    """Insert a semi-ellipsoidal surface crack via boolean cut.

    A unit sphere is dilated to match ``(half-length, depth, thickness)``
    semi-axes where ``half-length = depth / aspect_ratio`` and
    ``thickness = 0.1 * depth`` (a thin flaw).  The ellipsoid is placed at
    the midpoint of ``start -> end``.
    """
    _require_gmsh()
    start = np.array([defect.start.x, defect.start.y, defect.start.z])
    end = np.array([defect.end.x, defect.end.y, defect.end.z])
    mid = 0.5 * (start + end)

    half_length = defect.depth / max(defect.aspect_ratio, 1e-6)
    thickness = max(0.1 * defect.depth, 1e-4)

    sphere_tag = gmsh.model.occ.addSphere(
        float(mid[0]), float(mid[1]), float(mid[2]), 1.0
    )
    gmsh.model.occ.dilate(
        [(3, sphere_tag)],
        float(mid[0]), float(mid[1]), float(mid[2]),
        half_length, thickness, defect.depth,
    )

    cut_out, _ = gmsh.model.occ.cut(
        [(3, parent_volume_tag)], [(3, sphere_tag)], removeTool=True
    )
    gmsh.model.occ.synchronize()
    new_tag = cut_out[0][1] if cut_out else parent_volume_tag
    return {"parent_tag": new_tag, "defect_type": "surface_crack"}


def insert_root_gap(
    parent_volume_tag: int, defect: RootGapDefect
) -> dict[str, Any]:
    """Insert a root gap by cutting a thin rectangular slot along the weld."""
    _require_gmsh()
    start = np.array([defect.start.x, defect.start.y, defect.start.z])
    end = np.array([defect.end.x, defect.end.y, defect.end.z])
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        return {"parent_tag": parent_volume_tag, "defect_type": "root_gap"}

    box_tag = gmsh.model.occ.addBox(
        0.0,
        -defect.gap_width / 2.0,
        -defect.plate_thickness / 2.0,
        length,
        defect.gap_width,
        defect.plate_thickness,
    )

    direction = axis / length
    x_axis = np.array([1.0, 0.0, 0.0])
    cross = np.cross(x_axis, direction)
    dot = float(np.dot(x_axis, direction))
    sin = float(np.linalg.norm(cross))
    if sin > 1e-12:
        angle = float(np.arctan2(sin, dot))
        axis_n = cross / sin
        gmsh.model.occ.rotate(
            [(3, box_tag)], 0.0, 0.0, 0.0,
            float(axis_n[0]), float(axis_n[1]), float(axis_n[2]), angle,
        )
    elif dot < 0.0:
        gmsh.model.occ.rotate(
            [(3, box_tag)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(np.pi)
        )
    gmsh.model.occ.translate(
        [(3, box_tag)], float(start[0]), float(start[1]), float(start[2])
    )

    cut_out, _ = gmsh.model.occ.cut(
        [(3, parent_volume_tag)], [(3, box_tag)], removeTool=True
    )
    gmsh.model.occ.synchronize()
    new_tag = cut_out[0][1] if cut_out else parent_volume_tag
    return {"parent_tag": new_tag, "defect_type": "root_gap"}


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def insert_defect(
    parent_volume_tag: int, defect: Defect
) -> dict[str, Any]:
    """Dispatch a defect to the appropriate insertion helper.

    Dispatch uses the dataclass ``defect_type`` attribute rather than
    ``isinstance`` so duck-typed defects are accepted too.
    """
    _require_gmsh()
    dtype = getattr(defect, "defect_type", None)
    if dtype == "pore":
        return insert_pore(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "cluster_porosity":
        return insert_cluster_porosity(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "slag":
        return insert_slag_inclusion(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "undercut":
        return insert_undercut(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "lack_of_fusion":
        return insert_lack_of_fusion(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "surface_crack":
        return insert_surface_crack(parent_volume_tag, defect)  # type: ignore[arg-type]
    if dtype == "root_gap":
        return insert_root_gap(parent_volume_tag, defect)  # type: ignore[arg-type]
    raise ValueError(f"Unknown defect type: {dtype!r}")


def insert_all(
    parent_volume_tag: int, defects: list[Defect]
) -> dict[str, Any]:
    """Sequentially insert many defects into one parent volume.

    The returned dict contains the final parent volume tag and a list of
    per-defect insertion records in insertion order.  A best-effort
    ``removeAllDuplicates`` is called at the end to stitch shared faces.
    """
    _require_gmsh()
    current_tag = parent_volume_tag
    records: list[dict[str, Any]] = []
    for d in defects:
        rec = insert_defect(current_tag, d)
        current_tag = rec["parent_tag"]
        records.append(rec)
    try:  # pragma: no cover - OCC occasionally rejects duplicate removal
        gmsh.model.occ.removeAllDuplicates()
    except Exception:
        pass
    gmsh.model.occ.synchronize()
    return {"parent_tag": current_tag, "insertion_records": records}


__all__ = [
    "insert_all",
    "insert_cluster_porosity",
    "insert_defect",
    "insert_lack_of_fusion",
    "insert_pore",
    "insert_root_gap",
    "insert_slag_inclusion",
    "insert_surface_crack",
    "insert_undercut",
]
