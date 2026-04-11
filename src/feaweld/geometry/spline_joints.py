"""Spline-swept butt weld builder (Track D4).

Implements :class:`SplineButtWeld`, a butt weld whose weld-metal solid is
swept along a 3D curved :class:`~feaweld.geometry.weld_path.WeldPath`
using gmsh OCC ``addPipe``. The module imports cleanly without gmsh
installed; calling :meth:`SplineButtWeld.build` without gmsh raises a
clear :class:`ImportError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
            "gmsh is required for SplineButtWeld. "
            "Install gmsh>=4.11."
        )


def _ensure_initialized() -> None:  # pragma: no cover - requires gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()


@dataclass
class SplineButtWeld:
    """Butt weld swept along a 3D curved :class:`WeldPath`.

    The groove cross-section is swept along ``path`` using gmsh's OCC
    pipe operation. A bounding plate box enclosing the path is then
    fragmented against the swept weld metal so both share a common
    interface.

    Parameters
    ----------
    plate_width:
        Plate extent perpendicular to the weld path (in-plane).
    plate_thickness:
        Plate thickness (through-thickness extent).
    path:
        The curved weld path.
    groove:
        The 2D groove cross-section to sweep.
    name:
        Human-readable name (used for model naming / debugging).
    """

    plate_width: float
    plate_thickness: float
    path: WeldPath
    groove: GrooveProfile
    name: str = "spline_butt_weld"

    _physical_groups: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _volume_tags: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Curvature validation
    # ------------------------------------------------------------------

    def check_curvature(self, safety_factor: float = 1.2) -> None:
        """Raise :class:`ValueError` if the path curvature is too sharp.

        A swept profile self-intersects when the path's radius of
        curvature is smaller than the profile's half-width. The minimum
        radius of curvature is estimated from 50 densely-sampled points
        along the path via consecutive tangent changes.
        """
        n_samples = 50
        total = self.path.arc_length()
        if total <= 0.0:
            raise ValueError("WeldPath has zero arc length")

        ss = np.linspace(0.0, total, n_samples)
        # Convert arc-length samples to parameter u for tangent evaluation.
        u_vals = np.empty_like(ss)
        for i, s in enumerate(ss):
            u_vals[i] = self.path._u_for_arc_length(float(s), total)

        tangents = np.asarray(
            [self.path.tangent(float(u)) for u in u_vals], dtype=np.float64
        )
        positions = np.asarray(
            [self.path.evaluate_u(float(u)) for u in u_vals], dtype=np.float64
        )

        # Estimate local curvature from finite-difference of unit tangents
        # kappa ~ |dT/ds|. Use central differences where possible.
        min_radius = np.inf
        for i in range(1, n_samples - 1):
            dT = tangents[i + 1] - tangents[i - 1]
            dS = float(
                np.linalg.norm(positions[i + 1] - positions[i])
                + np.linalg.norm(positions[i] - positions[i - 1])
            )
            if dS < 1e-12:
                continue
            kappa = float(np.linalg.norm(dT)) / dS
            if kappa < 1e-12:
                continue
            r = 1.0 / kappa
            if r < min_radius:
                min_radius = r

        poly = self.groove.cross_section_polygon()
        half_width = float(np.max(np.abs(poly[:, 0])))
        required = half_width * float(safety_factor)

        if np.isfinite(min_radius) and min_radius < required:
            raise ValueError(
                f"WeldPath minimum radius of curvature {min_radius:.4g} is "
                f"smaller than groove half-width * safety_factor "
                f"({required:.4g}); the swept profile would self-intersect."
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build the spline butt weld in the current gmsh model.

        Returns a dictionary with ``volume_tags`` (list of ``(dim, tag)``
        entries) and ``physical_groups`` (dict mapping name -> tag).
        """
        _require_gmsh()
        self.check_curvature()
        _ensure_initialized()

        # ----- 1. Build the weld-path wire -----
        wire_tag = self.path.to_gmsh_wire(gmsh.model)

        # ----- 2. Build the groove cross-section polygon -----
        poly = self.groove.cross_section_polygon()  # (n, 2) in (t, z_local)

        # ----- 3. Frenet frame at s=0 and origin point -----
        T, N, B = self.path.frenet_frame(0.0)
        origin = np.asarray(self.path.evaluate_u(0.0), dtype=np.float64)

        # Map (t_local, z_local) -> origin + t_local * N + z_local * B.
        # The profile lies in the plane perpendicular to T at origin.
        pt_tags: list[int] = []
        for t_local, z_local in poly:
            p = (
                origin
                + float(t_local) * np.asarray(N, dtype=np.float64)
                + float(z_local) * np.asarray(B, dtype=np.float64)
            )
            pt_tags.append(
                gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2]))
            )

        n_pts = len(pt_tags)
        line_tags = [
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % n_pts])
            for i in range(n_pts)
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        profile_surface = gmsh.model.occ.addPlaneSurface([loop])

        # ----- 4. Pipe sweep along the wire -----
        pipe_out = gmsh.model.occ.addPipe([(2, profile_surface)], wire_tag)
        weld_volume_tags = [tag for dim, tag in pipe_out if dim == 3]
        if not weld_volume_tags:
            raise RuntimeError("addPipe did not produce a 3D volume")

        # ----- 5. Bounding plate box that contains the path -----
        samples = self.path.sample(64, by="s")
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        margin = 2.0 * float(self.plate_thickness) + float(self.plate_width)
        box_x0 = float(mins[0]) - margin
        box_y0 = float(mins[1]) - margin
        box_z0 = float(mins[2]) - margin
        box_dx = float(maxs[0] - mins[0]) + 2.0 * margin
        box_dy = float(maxs[1] - mins[1]) + 2.0 * margin
        box_dz = float(maxs[2] - mins[2]) + 2.0 * margin
        plate_box = gmsh.model.occ.addBox(
            box_x0, box_y0, box_z0, box_dx, box_dy, box_dz
        )

        # ----- 6. Fragment so plate and weld share a common interface -----
        obj = [(3, plate_box)]
        tool = [(3, t) for t in weld_volume_tags]
        frag_out, frag_map = gmsh.model.occ.fragment(obj, tool)

        plate_tags = [t for d, t in frag_map[0] if d == 3]
        weld_tags: list[int] = []
        for i in range(1, len(frag_map)):
            weld_tags.extend(t for d, t in frag_map[i] if d == 3)

        # Remove duplicates while preserving order
        seen: set[int] = set()
        weld_unique: list[int] = []
        for t in weld_tags:
            if t not in seen:
                seen.add(t)
                weld_unique.append(t)
        plate_only = [t for t in plate_tags if t not in seen]

        gmsh.model.occ.synchronize()

        # ----- 7. Physical groups -----
        pg: dict[str, int] = {}
        if plate_only:
            pg["plate"] = gmsh.model.addPhysicalGroup(3, plate_only)
            gmsh.model.setPhysicalName(3, pg["plate"], "plate")
        if weld_unique:
            pg["weld_metal"] = gmsh.model.addPhysicalGroup(3, weld_unique)
            gmsh.model.setPhysicalName(3, pg["weld_metal"], "weld_metal")

        # Weld-toe curves: 1D entities on the weld-metal volumes.
        toe_curve_tags: list[int] = []
        for t in weld_unique:
            try:
                boundary = gmsh.model.getBoundary(
                    [(3, t)], oriented=False, recursive=False
                )
            except Exception:  # pragma: no cover - defensive
                boundary = []
            for bdim, btag in boundary:
                if bdim == 2:
                    try:
                        edges = gmsh.model.getBoundary(
                            [(2, btag)], oriented=False, recursive=False
                        )
                    except Exception:  # pragma: no cover - defensive
                        edges = []
                    for edim, etag in edges:
                        if edim == 1 and etag not in toe_curve_tags:
                            toe_curve_tags.append(etag)
        if toe_curve_tags:
            pg["weld_toe"] = gmsh.model.addPhysicalGroup(1, toe_curve_tags)
            gmsh.model.setPhysicalName(1, pg["weld_toe"], "weld_toe")

        self._physical_groups = pg
        self._volume_tags = (
            [(3, t) for t in plate_only] + [(3, t) for t in weld_unique]
        )

        return {
            "volume_tags": list(self._volume_tags),
            "physical_groups": dict(self._physical_groups),
        }

    # ------------------------------------------------------------------
    # Toe sample points
    # ------------------------------------------------------------------

    def get_weld_toe_points(
        self, n_samples: int = 16
    ) -> list[tuple[float, float, float]]:
        """Return ``n_samples`` points along the weld toe for refinement.

        Toe points are computed by evaluating the path at evenly spaced
        arc-length positions and offsetting by the largest ``|t|`` at
        ``z == plate_thickness`` vertices of the groove polygon, using
        the local Frenet frame at each sample. Both left and right toes
        are returned, interleaved.
        """
        poly = self.groove.cross_section_polygon()
        tol = 1e-9
        top_mask = np.abs(poly[:, 1] - float(self.plate_thickness)) < tol
        if not np.any(top_mask):
            # Fall back to the maximum-z row
            max_z = float(poly[:, 1].max())
            top_mask = np.abs(poly[:, 1] - max_z) < tol
        top_ts = poly[top_mask, 0]
        top_zs = poly[top_mask, 1]

        total = self.path.arc_length()
        ss = np.linspace(0.0, total, max(int(n_samples), 2))
        out: list[tuple[float, float, float]] = []
        for s in ss:
            u = self.path._u_for_arc_length(float(s), total)
            T, N, B = self.path.frenet_frame(float(u))
            origin = np.asarray(
                self.path.evaluate_u(float(u)), dtype=np.float64
            )
            N_arr = np.asarray(N, dtype=np.float64)
            B_arr = np.asarray(B, dtype=np.float64)
            for t_local, z_local in zip(top_ts, top_zs):
                p = (
                    origin
                    + float(t_local) * N_arr
                    + float(z_local) * B_arr
                )
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out
