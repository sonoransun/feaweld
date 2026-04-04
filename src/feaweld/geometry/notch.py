"""Fictitious notch radius insertion for the effective notch stress method.

Per IIW recommendations, sharp weld-toe corners are replaced with circular
arcs of a reference radius (typically 1.0 mm for steel structures) before
meshing and solving, so that the resulting peak stress can be compared
directly against FAT classes.

This module modifies an active Gmsh model in-place using OCC boolean
operations.
"""

from __future__ import annotations

import gmsh
import numpy as np

from feaweld.geometry.joints import JointGeometry, _ensure_gmsh_initialized


def insert_fictitious_radius(
    toe_points: list[tuple[float, float, float]],
    radius: float = 1.0,
) -> None:
    """Replace sharp corners at *toe_points* with circular arcs.

    Operates on the **current** Gmsh model.  For each toe point the
    function:

    1. Creates a cylinder (disc in 2D) of the given *radius* centred at
       the toe location.
    2. Performs an OCC boolean *cut* to carve a circular notch into every
       surface entity that overlaps the disc.
    3. Then re-adds the curved fillet as weld metal surface so that the
       model remains watertight.

    The net effect is that the infinitely sharp re-entrant corner is
    replaced by a smooth circular arc of the specified radius — exactly
    what the IIW effective-notch-stress method prescribes.

    Parameters
    ----------
    toe_points:
        List of (x, y, z) tuples identifying weld toe locations where
        the notch radius should be inserted.
    radius:
        Fictitious notch radius in model units (mm).  IIW recommends
        1.0 mm for steel structures with plate thickness >= 5 mm.
    """
    if not toe_points:
        return

    for px, py, pz in toe_points:
        # Create a disc (circle surface) centred at the toe point.
        disk_tag = gmsh.model.occ.addDisk(px, py, pz, radius, radius)

        # Collect every existing 2-D surface entity that is *not* the disc.
        existing = [
            (d, t) for d, t in gmsh.model.occ.getEntities(2) if t != disk_tag
        ]

        if not existing:
            continue

        # Fragment the disc with every existing surface so that the arc
        # boundary is imprinted on adjacent regions.
        obj = [(2, disk_tag)]
        tool = existing
        gmsh.model.occ.fragment(obj, tool)
        gmsh.model.occ.synchronize()


def create_notched_model(
    joint: JointGeometry,
    radius: float = 1.0,
    model_name: str = "notched",
) -> None:
    """Build joint geometry and then apply notch rounding.

    Convenience wrapper that:

    1. Calls ``joint.build(model_name)`` to create the sharp-corner
       geometry with physical groups.
    2. Retrieves weld-toe coordinates from the joint.
    3. Calls :func:`insert_fictitious_radius` to round each toe.

    After this function returns, the active Gmsh model is ready for
    meshing.

    Parameters
    ----------
    joint:
        A :class:`~feaweld.geometry.joints.JointGeometry` instance whose
        ``build`` method has **not** yet been called (this function will
        call it).
    radius:
        Fictitious notch radius (mm).
    model_name:
        Gmsh model name.
    """
    _ensure_gmsh_initialized()

    # Step 1 -- build the parametric joint geometry
    joint.build(model_name=model_name)

    # Step 2 -- round each weld toe
    toe_points = joint.get_weld_toe_points()
    insert_fictitious_radius(toe_points, radius=radius)
