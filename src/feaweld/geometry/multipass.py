"""Multi-pass weld bead layering on top of a base joint (Track G).

The :func:`build_multipass_joint` helper walks a
:class:`~feaweld.core.types.WeldSequence`, builds a prismatic bead volume
for each pass from its cross-section profile, and tags each deposited
volume as a separate physical group so a birth-death-aware solver can
activate passes in order.

The module is gmsh-gated: it imports cleanly without gmsh present, but
calling :func:`build_multipass_joint` without gmsh raises a clear
:class:`ImportError`. The implementation keeps the MVP intentionally
minimal — it assumes straight weld paths along the joint length and
stacks bead cross-sections vertically so each pass sits above the prior
one. Full curved-path multi-pass sweeping is out of scope for this MVP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from feaweld.core.types import WeldPass, WeldSequence
from feaweld.geometry.groove import GrooveProfile
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
            "gmsh required for build_multipass_joint. Install gmsh>=4.11."
        )


def _ensure_initialized() -> None:
    if not gmsh.isInitialized():
        gmsh.initialize()


def _pass_length_from_path(path: WeldPath) -> float:
    """Return the deposited length along the weld path (mm)."""
    return float(path.arc_length())


def _extrude_pass_bead(
    profile: GrooveProfile,
    z_offset: float,
    length: float,
) -> int:
    """Extrude a single pass bead polygon along ``+z``.

    The profile polygon is interpreted in local ``(t, z_local)`` coords.
    ``t`` maps to the global ``x`` axis, ``z_local`` is shifted by
    ``z_offset`` and mapped to the global ``y`` axis, and the extrusion
    direction is the global ``z`` axis over ``length``.
    """
    poly = profile.cross_section_polygon()
    pt_tags: list[int] = []
    for t_val, z_val in poly:
        pt_tags.append(
            gmsh.model.occ.addPoint(
                float(t_val), float(z_val) + float(z_offset), 0.0
            )
        )
    n = len(pt_tags)
    line_tags = [
        gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % n])
        for i in range(n)
    ]
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    surf = gmsh.model.occ.addPlaneSurface([loop])
    out = gmsh.model.occ.extrude([(2, surf)], 0.0, 0.0, float(length))
    for dim, tag in out:
        if dim == 3:
            return tag
    raise RuntimeError("extrude did not produce a 3D volume")


def build_multipass_joint(
    base_joint: Any,
    sequence: WeldSequence,
    path: WeldPath,
    per_pass_profiles: list[GrooveProfile],
) -> dict[str, Any]:
    """Build a multi-pass weld by successively depositing each pass bead.

    Parameters
    ----------
    base_joint:
        Any existing joint object exposing a ``build()`` method. Its build
        result is discarded for the MVP — we only use it to synchronise
        the gmsh OCC state so the bead volumes are fragmented against the
        plate geometry. Passing ``None`` skips the base-joint build step,
        which keeps the test path lightweight.
    sequence:
        Ordered :class:`WeldSequence` — one :class:`WeldPass` per entry in
        ``per_pass_profiles``.
    path:
        :class:`WeldPath` providing the deposition length.
    per_pass_profiles:
        One :class:`GrooveProfile` per pass, in the same order as
        ``sequence.passes``.

    Returns
    -------
    dict
        ``{"volume_tags": [...], "physical_groups": {"pass_<order>": tag}}``
        where each physical group tags a single pass's deposited volume.
    """
    _require_gmsh()
    if len(per_pass_profiles) != len(sequence.passes):
        raise ValueError(
            f"per_pass_profiles length ({len(per_pass_profiles)}) must match "
            f"sequence.passes length ({len(sequence.passes)})"
        )
    _ensure_initialized()

    if base_joint is not None:
        # Build the base model so subsequent bead extrusions share the
        # same OCC context. The return value is currently unused.
        base_joint.build()

    length = _pass_length_from_path(path)

    volume_tags: list[int] = []
    physical_groups: dict[str, int] = {}

    z_offset = 0.0
    for weld_pass, profile in zip(sequence.passes, per_pass_profiles):
        vol_tag = _extrude_pass_bead(profile, z_offset=z_offset, length=length)
        volume_tags.append(vol_tag)

        # Advance the z offset so the next pass stacks on top of this
        # one. This is an MVP approximation: real multi-pass beads would
        # follow the residual groove height from the prior pass.
        poly = profile.cross_section_polygon()
        z_offset += float(poly[:, 1].max() - poly[:, 1].min())

    gmsh.model.occ.synchronize()

    for weld_pass, vol_tag in zip(sequence.passes, volume_tags):
        group_name = f"pass_{weld_pass.order}"
        pg_tag = gmsh.model.addPhysicalGroup(3, [vol_tag])
        gmsh.model.setPhysicalName(3, pg_tag, group_name)
        physical_groups[group_name] = pg_tag

    return {
        "volume_tags": volume_tags,
        "physical_groups": physical_groups,
    }
