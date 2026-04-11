"""Dataclasses describing weld defects used by reporting and FAT downgrade.

Geometry primitives here are standalone descriptions. Insertion of a defect
into a Gmsh model is handled by Track E2 and is not performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import Point3D


class Defect(Protocol):
    """Protocol for a weld defect that can be reported and downgraded."""

    defect_type: str

    def volume(self) -> float: ...
    def critical_dimension(self) -> float: ...
    def description(self) -> str: ...


def _segment_length(start: Point3D, end: Point3D) -> float:
    a = np.asarray([start.x, start.y, start.z])
    b = np.asarray([end.x, end.y, end.z])
    return float(np.linalg.norm(b - a))


@dataclass
class PoreDefect:
    """Single spherical gas pore."""

    center: Point3D
    diameter: float
    defect_type: str = "pore"

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * (self.diameter / 2.0) ** 3

    def critical_dimension(self) -> float:
        return self.diameter

    def description(self) -> str:
        return f"Pore d={self.diameter:.2f}mm at {self.center}"


@dataclass
class ClusterPorosity:
    """Cluster of small pores inside a bounding sphere."""

    center: Point3D
    radius: float
    n_pores: int
    size_mean: float
    size_std: float
    defect_type: str = "cluster_porosity"

    def volume(self) -> float:
        pore_volume = (4.0 / 3.0) * np.pi * (self.size_mean / 2.0) ** 3
        return float(self.n_pores) * pore_volume

    def critical_dimension(self) -> float:
        return 2.0 * self.radius

    def description(self) -> str:
        return (
            f"cluster_porosity n={self.n_pores} "
            f"mean_d={self.size_mean:.2f}mm in sphere r={self.radius:.2f}mm"
        )


@dataclass
class SlagInclusion:
    """Ellipsoidal slag inclusion with a stiffness-ratio surrogate."""

    center: Point3D
    semi_axes: tuple[float, float, float]
    orientation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    modulus_ratio: float = 0.1
    defect_type: str = "slag"

    def volume(self) -> float:
        a, b, c = self.semi_axes
        return (4.0 / 3.0) * np.pi * a * b * c

    def critical_dimension(self) -> float:
        return max(self.semi_axes) * 2.0

    def description(self) -> str:
        a, b, c = self.semi_axes
        return (
            f"slag ellipsoid ({a:.2f},{b:.2f},{c:.2f})mm "
            f"E_ratio={self.modulus_ratio:.2f}"
        )


@dataclass
class UndercutDefect:
    """Linear toe undercut with V- or U-profile cross-section."""

    start: Point3D
    end: Point3D
    depth: float
    profile: Literal["V", "U"] = "V"
    defect_type: str = "undercut"

    def volume(self) -> float:
        length = _segment_length(self.start, self.end)
        if self.profile == "V":
            return 0.5 * length * self.depth ** 2
        return length * self.depth * self.depth

    def critical_dimension(self) -> float:
        return self.depth

    def description(self) -> str:
        length = _segment_length(self.start, self.end)
        return (
            f"undercut {self.profile}-profile depth={self.depth:.2f}mm "
            f"length={length:.2f}mm"
        )


@dataclass
class LackOfFusionDefect:
    """Planar lack-of-fusion flaw described by origin, normal, and in-plane extents."""

    plane_origin: Point3D
    plane_normal: NDArray
    extent_u: float
    extent_v: float
    defect_type: str = "lack_of_fusion"

    def volume(self) -> float:
        return 0.0

    def critical_dimension(self) -> float:
        return max(self.extent_u, self.extent_v)

    def description(self) -> str:
        return (
            f"lack_of_fusion {self.extent_u:.2f}x{self.extent_v:.2f}mm "
            f"at {self.plane_origin}"
        )


@dataclass
class RootGapDefect:
    """Excess root opening along a weld line."""

    start: Point3D
    end: Point3D
    gap_width: float
    plate_thickness: float
    defect_type: str = "root_gap"

    def volume(self) -> float:
        length = _segment_length(self.start, self.end)
        return length * self.gap_width * self.plate_thickness

    def critical_dimension(self) -> float:
        return self.gap_width

    def description(self) -> str:
        length = _segment_length(self.start, self.end)
        return (
            f"root_gap width={self.gap_width:.2f}mm length={length:.2f}mm "
            f"t={self.plate_thickness:.2f}mm"
        )


@dataclass
class SurfaceCrack:
    """Semi-elliptical surface crack at a weld toe."""

    start: Point3D
    end: Point3D
    depth: float
    aspect_ratio: float = 0.3
    defect_type: str = "surface_crack"

    def volume(self) -> float:
        return 0.0

    def critical_dimension(self) -> float:
        return self.depth

    def description(self) -> str:
        length = _segment_length(self.start, self.end)
        return (
            f"surface_crack depth={self.depth:.2f}mm length={length:.2f}mm "
            f"a/c={self.aspect_ratio:.2f}"
        )
