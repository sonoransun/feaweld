"""Shared data types used across all feaweld modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y])


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])


@dataclass
class WeldSegment:
    """A straight-line segment of a weld, defined by two endpoints."""
    start: Point3D
    end: Point3D
    leg_size: float  # fillet weld leg size (mm)
    weld_type: WeldType = field(default_factory=lambda: WeldType.FILLET)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end.to_array() - self.start.to_array()))

    @property
    def throat(self) -> float:
        """Effective throat thickness for fillet welds (leg / sqrt(2))."""
        if self.weld_type == WeldType.FILLET:
            return self.leg_size / np.sqrt(2.0)
        return self.leg_size  # for groove welds, leg_size = throat


class WeldType(str, Enum):
    FILLET = "fillet"
    GROOVE_FULL = "groove_full_penetration"
    GROOVE_PARTIAL = "groove_partial_penetration"
    PLUG = "plug"
    SLOT = "slot"


class JointType(str, Enum):
    FILLET_T = "fillet_t"
    BUTT = "butt"
    LAP = "lap"
    CORNER = "corner"
    CRUCIFORM = "cruciform"


# ---------------------------------------------------------------------------
# Weld group section properties (for Blodgett hand calculations)
# ---------------------------------------------------------------------------

class WeldGroupShape(str, Enum):
    """Standard weld group configurations from Blodgett's tables."""
    LINE = "line"           # Single straight line
    PARALLEL = "parallel"   # Two parallel lines
    C_SHAPE = "c_shape"     # Channel shape (3-sided)
    L_SHAPE = "l_shape"     # Angle shape (2-sided)
    BOX = "box"             # Rectangular (4-sided)
    CIRCULAR = "circular"   # Full circle
    I_SHAPE = "i_shape"     # I-beam flanges
    T_SHAPE = "t_shape"     # T-shape
    U_SHAPE = "u_shape"     # U-shape (3-sided, open top)


@dataclass
class WeldGroupProperties:
    """Section properties of a weld group treated as a line."""
    A_w: float          # Total weld length (mm) — "area" for weld-as-line
    S_x: float          # Section modulus about x-axis (mm^2)
    S_y: float          # Section modulus about y-axis (mm^2)
    J_w: float          # Polar moment of inertia as line (mm^3)
    x_bar: float = 0.0  # Centroid x (mm)
    y_bar: float = 0.0  # Centroid y (mm)
    I_x: float = 0.0    # Moment of inertia about x (mm^2)
    I_y: float = 0.0    # Moment of inertia about y (mm^2)


# ---------------------------------------------------------------------------
# Mesh representation
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"
    HEX20 = "hex20"


@dataclass
class FEMesh:
    """Finite element mesh representation, solver-agnostic."""
    nodes: NDArray[np.float64]         # (n_nodes, 3) coordinates
    elements: NDArray[np.int64]        # (n_elements, n_nodes_per_elem) connectivity
    element_type: ElementType
    physical_groups: dict[str, NDArray[np.int64]] = field(default_factory=dict)
    node_sets: dict[str, NDArray[np.int64]] = field(default_factory=dict)
    element_sets: dict[str, NDArray[np.int64]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def ndim(self) -> int:
        return self.nodes.shape[1]


# ---------------------------------------------------------------------------
# FEA results
# ---------------------------------------------------------------------------

@dataclass
class StressField:
    """Stress tensor components at nodes or integration points."""
    values: NDArray[np.float64]  # (n_points, 6) for 3D: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
    location: str = "nodes"     # "nodes" or "gauss_points"

    @property
    def von_mises(self) -> NDArray[np.float64]:
        s = self.values
        return np.sqrt(
            0.5 * (
                (s[:, 0] - s[:, 1]) ** 2
                + (s[:, 1] - s[:, 2]) ** 2
                + (s[:, 2] - s[:, 0]) ** 2
                + 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
            )
        )

    @property
    def tresca(self) -> NDArray[np.float64]:
        """Maximum shear stress (Tresca) — stress intensity per ASME."""
        n = self.values.shape[0]
        result = np.empty(n)
        for i in range(n):
            tensor = np.array([
                [self.values[i, 0], self.values[i, 3], self.values[i, 5]],
                [self.values[i, 3], self.values[i, 1], self.values[i, 4]],
                [self.values[i, 5], self.values[i, 4], self.values[i, 2]],
            ])
            eigvals = np.linalg.eigvalsh(tensor)
            result[i] = eigvals[-1] - eigvals[0]
        return result

    @property
    def principal(self) -> NDArray[np.float64]:
        """Principal stresses (n_points, 3) sorted ascending."""
        n = self.values.shape[0]
        principals = np.empty((n, 3))
        for i in range(n):
            tensor = np.array([
                [self.values[i, 0], self.values[i, 3], self.values[i, 5]],
                [self.values[i, 3], self.values[i, 1], self.values[i, 4]],
                [self.values[i, 5], self.values[i, 4], self.values[i, 2]],
            ])
            principals[i] = np.sort(np.linalg.eigvalsh(tensor))
        return principals


@dataclass
class FEAResults:
    """Complete results from an FEA solve, solver-agnostic."""
    mesh: FEMesh
    displacement: NDArray[np.float64] | None = None  # (n_nodes, 3)
    stress: StressField | None = None
    strain: NDArray[np.float64] | None = None         # (n_nodes, 6)
    temperature: NDArray[np.float64] | None = None     # (n_nodes,) or (n_timesteps, n_nodes)
    nodal_forces: NDArray[np.float64] | None = None    # (n_nodes, 3) — reaction forces
    time_steps: NDArray[np.float64] | None = None      # for transient analyses
    time_history: dict[str, list[NDArray]] | None = None  # field name → list of arrays per timestep
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_transient(self) -> bool:
        return self.time_steps is not None and len(self.time_steps) > 1


# ---------------------------------------------------------------------------
# S-N curve specification
# ---------------------------------------------------------------------------

class SNStandard(str, Enum):
    IIW = "iiw"
    DNV = "dnv"
    ASME = "asme"


@dataclass
class SNCurve:
    """S-N curve definition: N = (C / S^m) for each segment."""
    name: str
    standard: SNStandard
    segments: list[SNSegment]
    cutoff_cycles: float = 1e7  # endurance limit cycles

    def life(self, stress_range: float) -> float:
        """Compute fatigue life N for a given stress range S."""
        for seg in self.segments:
            if stress_range >= seg.stress_threshold:
                return seg.C / (stress_range ** seg.m)
        return float("inf")


@dataclass
class SNSegment:
    """One segment of a piecewise S-N curve."""
    m: float           # slope exponent
    C: float           # intercept constant (N · S^m = C)
    stress_threshold: float = 0.0  # minimum stress range for this segment (MPa)


# ---------------------------------------------------------------------------
# Load definitions
# ---------------------------------------------------------------------------

class LoadType(str, Enum):
    FORCE = "force"
    PRESSURE = "pressure"
    DISPLACEMENT = "displacement"
    TEMPERATURE = "temperature"
    HEAT_FLUX = "heat_flux"
    CONVECTION = "convection"


@dataclass
class BoundaryCondition:
    """A boundary condition applied to a node set or surface."""
    node_set: str           # name referencing FEMesh.node_sets
    bc_type: LoadType
    values: NDArray[np.float64]  # component values (depends on type)
    direction: NDArray[np.float64] | None = None  # unit vector for directional loads


@dataclass
class LoadCase:
    """A collection of loads and boundary conditions for one analysis step."""
    name: str
    loads: list[BoundaryCondition] = field(default_factory=list)
    constraints: list[BoundaryCondition] = field(default_factory=list)


@dataclass
class LoadHistory:
    """Time-varying load history for fatigue analysis."""
    time: NDArray[np.float64]          # (n_points,) time values
    stress_ranges: NDArray[np.float64]  # (n_points,) stress range values
    mean_stress: NDArray[np.float64] | None = None  # (n_points,) for mean stress correction
    r_ratio: NDArray[np.float64] | None = None  # stress ratio R = σ_min / σ_max


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------

class StressMethod(str, Enum):
    NOMINAL = "nominal"
    HOTSPOT_LINEAR = "hotspot_linear"
    HOTSPOT_QUADRATIC = "hotspot_quadratic"
    STRUCTURAL_DONG = "structural_dong"
    NOTCH_STRESS = "notch_stress"
    SED = "strain_energy_density"
    LINEARIZATION = "linearization"
    BLODGETT = "blodgett"


class SolverType(str, Enum):
    LINEAR_ELASTIC = "linear_elastic"
    ELASTOPLASTIC = "elastoplastic"
    THERMAL_STEADY = "thermal_steady"
    THERMAL_TRANSIENT = "thermal_transient"
    THERMOMECHANICAL = "thermomechanical"
    CREEP = "creep"


@dataclass
class WeldLineDefinition:
    """Defines a weld line on the mesh for post-processing."""
    name: str
    node_ids: NDArray[np.int64]           # ordered node IDs along weld toe
    plate_thickness: float                 # plate thickness t (mm)
    normal_direction: NDArray[np.float64]  # unit vector normal to plate surface
    weld_side: str = "toe"                 # "toe" or "root"
