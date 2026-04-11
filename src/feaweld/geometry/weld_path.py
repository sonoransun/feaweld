"""3D weld path primitive supporting linear, B-spline, and Catmull-Rom curves.

The :class:`WeldPath` class parameterizes a curve through a list of control
points and exposes both parametric (``u`` in ``[0, 1]``) and arc-length
(``s`` in ``[0, arc_length()]``) evaluation as well as the Frenet frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import BSpline, make_interp_spline

from feaweld.core.types import Point3D


SplineMode = Literal["linear", "bspline", "catmull_rom"]


# Catmull-Rom basis matrix (uniform / centripetal-with-alpha=0 form, tension=0.5)
_CATMULL_ROM_M = 0.5 * np.array(
    [
        [0.0, 2.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [2.0, -5.0, 4.0, -1.0],
        [-1.0, 3.0, -3.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class WeldPath:
    """3D weld path defined by control points with optional smoothing.

    Parameters
    ----------
    control_points:
        Ordered list of 3D control points.
    mode:
        Interpolation mode: ``"linear"``, ``"bspline"`` or ``"catmull_rom"``.
    degree:
        Polynomial degree for ``"bspline"`` mode (ignored otherwise).
    """

    control_points: list[Point3D]
    mode: SplineMode = "bspline"
    degree: int = 3

    _pts: NDArray[np.float64] = field(init=False, repr=False)
    _bspline: BSpline | None = field(init=False, repr=False, default=None)
    _arc_length: float | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if len(self.control_points) < 2:
            raise ValueError("WeldPath requires at least 2 control points")
        self._pts = self._pts_array()

        if self.mode == "bspline":
            n = len(self.control_points)
            k = min(self.degree, n - 1)
            if k < 1:
                raise ValueError(
                    "bspline mode needs at least 2 control points and degree >= 1"
                )
            u_knots = np.linspace(0.0, 1.0, n)
            self._bspline = make_interp_spline(u_knots, self._pts, k=k)
        elif self.mode == "linear":
            self._bspline = None
        elif self.mode == "catmull_rom":
            if len(self.control_points) < 2:
                raise ValueError("catmull_rom mode needs at least 2 control points")
            self._bspline = None
        else:
            raise ValueError(f"unknown mode: {self.mode!r}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pts_array(self) -> NDArray[np.float64]:
        return np.asarray(
            [[p.x, p.y, p.z] for p in self.control_points], dtype=np.float64
        )

    def _catmull_rom_eval(
        self, u: NDArray[np.float64], derivative: int = 0
    ) -> NDArray[np.float64]:
        """Evaluate the Catmull-Rom spline (or its derivative) at ``u``."""
        pts = self._pts
        n = pts.shape[0]
        # n-1 segments; parameter u in [0, 1] maps to segment index
        seg_count = n - 1
        u_clip = np.clip(u, 0.0, 1.0)
        seg_float = u_clip * seg_count
        seg_idx = np.floor(seg_float).astype(int)
        seg_idx = np.clip(seg_idx, 0, seg_count - 1)
        local_t = seg_float - seg_idx

        # Build phantom end points by linear extrapolation
        p_pad = np.empty((n + 2, 3), dtype=np.float64)
        p_pad[1:-1] = pts
        p_pad[0] = 2.0 * pts[0] - pts[1]
        p_pad[-1] = 2.0 * pts[-1] - pts[-2]

        result = np.empty((local_t.shape[0], 3), dtype=np.float64)
        for i, (idx, t) in enumerate(zip(seg_idx, local_t)):
            p0 = p_pad[idx]
            p1 = p_pad[idx + 1]
            p2 = p_pad[idx + 2]
            p3 = p_pad[idx + 3]
            P = np.stack([p0, p1, p2, p3], axis=0)  # (4, 3)
            if derivative == 0:
                t_vec = np.array([1.0, t, t * t, t * t * t])
            elif derivative == 1:
                # d/du (parameter u) of position: chain rule factor seg_count
                t_vec = np.array([0.0, 1.0, 2.0 * t, 3.0 * t * t]) * seg_count
            elif derivative == 2:
                t_vec = np.array([0.0, 0.0, 2.0, 6.0 * t]) * (seg_count ** 2)
            else:
                raise ValueError("derivative must be 0, 1 or 2")
            result[i] = t_vec @ _CATMULL_ROM_M @ P
        return result

    def _linear_eval(
        self, u: NDArray[np.float64], derivative: int = 0
    ) -> NDArray[np.float64]:
        pts = self._pts
        n = pts.shape[0]
        seg_count = n - 1
        u_clip = np.clip(u, 0.0, 1.0)
        seg_float = u_clip * seg_count
        seg_idx = np.floor(seg_float).astype(int)
        seg_idx = np.clip(seg_idx, 0, seg_count - 1)
        local_t = seg_float - seg_idx
        result = np.empty((local_t.shape[0], 3), dtype=np.float64)
        for i, (idx, t) in enumerate(zip(seg_idx, local_t)):
            p0 = pts[idx]
            p1 = pts[idx + 1]
            if derivative == 0:
                result[i] = (1.0 - t) * p0 + t * p1
            elif derivative == 1:
                result[i] = (p1 - p0) * seg_count
            elif derivative == 2:
                result[i] = np.zeros(3)
            else:
                raise ValueError("derivative must be 0, 1 or 2")
        return result

    def _eval(
        self, u: float | NDArray, derivative: int = 0
    ) -> NDArray[np.float64]:
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if self.mode == "linear":
            out = self._linear_eval(u_arr, derivative=derivative)
        elif self.mode == "bspline":
            assert self._bspline is not None
            if derivative == 0:
                out = np.asarray(self._bspline(u_arr), dtype=np.float64)
            else:
                out = np.asarray(
                    self._bspline.derivative(derivative)(u_arr), dtype=np.float64
                )
        elif self.mode == "catmull_rom":
            out = self._catmull_rom_eval(u_arr, derivative=derivative)
        else:  # pragma: no cover - guarded in __post_init__
            raise ValueError(f"unknown mode: {self.mode!r}")

        if np.isscalar(u) or (isinstance(u, np.ndarray) and u.ndim == 0):
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate_u(self, u: float | NDArray) -> NDArray:
        """Evaluate the path at parametric coordinate(s) ``u`` in ``[0, 1]``."""
        return self._eval(u, derivative=0)

    def evaluate_s(self, s: float | NDArray) -> NDArray:
        """Evaluate the path at arc-length coordinate(s) ``s``."""
        total = self.arc_length()
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        u_vals = np.empty_like(s_arr)
        for i, si in enumerate(s_arr):
            u_vals[i] = self._u_for_arc_length(float(si), total)
        out = self._eval(u_vals, derivative=0)
        if np.isscalar(s) or (isinstance(s, np.ndarray) and s.ndim == 0):
            return out[0] if out.ndim == 2 else out
        return out

    def tangent(self, u: float | NDArray) -> NDArray:
        """Unit tangent vector(s) at ``u``."""
        d1 = self._eval(u, derivative=1)
        return _safe_normalize(d1)

    def normal(self, u: float | NDArray) -> NDArray:
        """Principal normal vector(s) at ``u``.

        Falls back to an arbitrary stable perpendicular when the second
        derivative vanishes (e.g. straight paths).
        """
        d1 = self._eval(u, derivative=1)
        d2 = self._eval(u, derivative=2)
        return _principal_normal(d1, d2)

    def binormal(self, u: float | NDArray) -> NDArray:
        t = self.tangent(u)
        n = self.normal(u)
        if t.ndim == 1:
            return _safe_normalize(np.cross(t, n))
        return _safe_normalize(np.cross(t, n))

    def frenet_frame(
        self, u: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return (T, N, B) unit vectors at ``u``."""
        t = self.tangent(u)
        n = self.normal(u)
        b = _safe_normalize(np.cross(t, n))
        # Re-orthogonalize n to guarantee N ⊥ T exactly
        n = _safe_normalize(np.cross(b, t))
        return t, n, b

    # ------------------------------------------------------------------
    # Arc length and sampling
    # ------------------------------------------------------------------

    def arc_length(self) -> float:
        """Return the total arc length of the path (cached)."""
        if self._arc_length is None:
            if self.mode == "linear":
                pts = self._pts
                diffs = np.diff(pts, axis=0)
                self._arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
            else:
                def speed(u: float) -> float:
                    d = self._eval(np.array([u]), derivative=1)[0]
                    return float(np.linalg.norm(d))

                val, _ = quad(speed, 0.0, 1.0, limit=200)
                self._arc_length = float(val)
        return self._arc_length

    def _u_for_arc_length(self, s: float, total: float) -> float:
        """Find ``u`` in [0, 1] whose arc length from 0 matches ``s`` via bisection."""
        if total <= 0.0:
            return 0.0
        s = max(0.0, min(s, total))

        def arc_to(u_hi: float) -> float:
            if u_hi <= 0.0:
                return 0.0
            if self.mode == "linear":
                pts = self._eval(np.linspace(0.0, u_hi, 128), derivative=0)
                return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

            def speed(u: float) -> float:
                d = self._eval(np.array([u]), derivative=1)[0]
                return float(np.linalg.norm(d))

            val, _ = quad(speed, 0.0, u_hi, limit=100)
            return float(val)

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if arc_to(mid) < s:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi)

    def sample(self, n: int, by: Literal["u", "s"] = "u") -> NDArray:
        """Sample ``n`` points along the path, by parameter or by arc length."""
        if n < 2:
            raise ValueError("sample requires n >= 2")
        if by == "u":
            us = np.linspace(0.0, 1.0, n)
            return self._eval(us, derivative=0)
        if by == "s":
            total = self.arc_length()
            ss = np.linspace(0.0, total, n)
            return self.evaluate_s(ss)
        raise ValueError(f"unknown sampling mode: {by!r}")

    # ------------------------------------------------------------------
    # Gmsh interoperability
    # ------------------------------------------------------------------

    def to_gmsh_wire(self, model) -> int:  # pragma: no cover - requires gmsh
        """Build a Gmsh OCC B-spline wire from the control points.

        The method is a convenience wrapper; it imports :mod:`gmsh` lazily
        and raises :class:`ImportError` if it is not available.
        """
        try:
            import gmsh  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "gmsh is required for WeldPath.to_gmsh_wire(); install the "
                "'geometry' extra to enable it"
            ) from exc
        import gmsh as _gmsh

        pt_tags = [
            _gmsh.model.occ.addPoint(float(p.x), float(p.y), float(p.z))
            for p in self.control_points
        ]
        if self.mode == "linear":
            line_tags = [
                _gmsh.model.occ.addLine(pt_tags[i], pt_tags[i + 1])
                for i in range(len(pt_tags) - 1)
            ]
            return _gmsh.model.occ.addWire(line_tags)
        wire_curve = _gmsh.model.occ.addBSpline(pt_tags)
        return _gmsh.model.occ.addWire([wire_curve])


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def _safe_normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        if n < 1e-15:
            return np.zeros_like(v)
        return v / n
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(norms < 1e-15, 1.0, norms)
    out = v / safe
    out[np.squeeze(norms, axis=-1) < 1e-15] = 0.0
    return out


def _arbitrary_perpendicular(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a unit vector perpendicular to ``t`` (a non-zero 3-vector)."""
    if t.ndim == 1:
        # Pick the axis most orthogonal to t
        ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        p = np.cross(t, ref)
        return _safe_normalize(p)
    out = np.empty_like(t)
    for i in range(t.shape[0]):
        out[i] = _arbitrary_perpendicular(t[i])
    return out


def _principal_normal(
    d1: NDArray[np.float64], d2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the principal (Frenet) normal from first and second derivatives."""
    if d1.ndim == 1:
        t = _safe_normalize(d1)
        # Project d2 onto the plane perpendicular to t
        if float(np.linalg.norm(d2)) < 1e-12:
            return _arbitrary_perpendicular(t)
        proj = d2 - np.dot(d2, t) * t
        if float(np.linalg.norm(proj)) < 1e-12:
            return _arbitrary_perpendicular(t)
        return _safe_normalize(proj)
    # Vectorized case
    out = np.empty_like(d1)
    for i in range(d1.shape[0]):
        out[i] = _principal_normal(d1[i], d2[i])
    return out
