"""Shared data types used across all feaweld modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

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
    # 3D groove preparations (Track D)
    V_GROOVE = "v_groove"
    U_GROOVE = "u_groove"
    J_GROOVE = "j_groove"
    X_GROOVE = "x_groove"
    K_GROOVE = "k_groove"
    # Fastener welds
    PLUG = "plug"
    SLOT = "slot"
    STUD = "stud"
    SPOT = "spot"
    # Spline / volumetric
    SPLINE_BUTT = "spline_butt"
    VOLUMETRIC_FILLET = "volumetric_fillet"


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

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError(
                f"FEMesh.nodes must have shape (n, 2) or (n, 3), got {self.nodes.shape}"
            )
        if self.elements.ndim != 2:
            raise ValueError(
                f"FEMesh.elements must be 2D, got shape {self.elements.shape}"
            )

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

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != 6:
            raise ValueError(
                f"StressField.values must have shape (n, 6), got {self.values.shape}"
            )

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

    @property
    def hydrostatic(self) -> NDArray[np.float64]:
        """Hydrostatic (mean normal) stress: (σ_xx + σ_yy + σ_zz) / 3."""
        s = self.values
        return (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0

    @property
    def deviatoric(self) -> NDArray[np.float64]:
        """Deviatoric stress tensor in Voigt form, (n_points, 6)."""
        s = self.values
        p = self.hydrostatic
        dev = s.copy()
        dev[:, 0] -= p
        dev[:, 1] -= p
        dev[:, 2] -= p
        return dev

    @property
    def octahedral_shear(self) -> NDArray[np.float64]:
        """Octahedral shear stress: τ_oct = sqrt(2/3) * ||deviatoric||.

        Equivalent to (sqrt(2)/3) * σ_vm.
        """
        return np.sqrt(2.0) / 3.0 * self.von_mises

    @property
    def invariants(self) -> NDArray[np.float64]:
        """Principal invariants (I_1, I_2, I_3) of the full stress tensor."""
        s = self.values
        I1 = s[:, 0] + s[:, 1] + s[:, 2]
        I2 = (
            s[:, 0] * s[:, 1] + s[:, 1] * s[:, 2] + s[:, 0] * s[:, 2]
            - s[:, 3] ** 2 - s[:, 4] ** 2 - s[:, 5] ** 2
        )
        I3 = (
            s[:, 0] * s[:, 1] * s[:, 2]
            + 2.0 * s[:, 3] * s[:, 4] * s[:, 5]
            - s[:, 0] * s[:, 4] ** 2
            - s[:, 1] * s[:, 5] ** 2
            - s[:, 2] * s[:, 3] ** 2
        )
        return np.stack([I1, I2, I3], axis=1)

    @property
    def signed_von_mises(self) -> NDArray[np.float64]:
        """Von Mises equivalent stress signed by the sign of the max principal.

        Useful for mean-stress-aware fatigue: positive under tension-dominated
        multi-axial states, negative under compression-dominated ones.
        """
        vm = self.von_mises
        max_principal = self.principal[:, -1]
        sign = np.where(max_principal >= 0.0, 1.0, -1.0)
        return sign * vm


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
        if stress_range <= 0:
            return float("inf")
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
    # Optional spline path for arc-length-parameterized stress extraction.
    # Kept as Any to avoid an import cycle with geometry.weld_path; runtime
    # callers type-check via duck typing on .evaluate / .tangent / .arc_length.
    path: Any | None = None


# ---------------------------------------------------------------------------
# Multi-pass welding sequence (Track G)
# ---------------------------------------------------------------------------

@dataclass
class WeldPass:
    """A single pass in a multi-pass welding sequence.

    Attributes
    ----------
    order:
        1-based pass index used as the dispatch key.
    pass_type:
        Label indicating the pass role ("root", "fill" or "cap").
    bead_area:
        Deposited cross-section area (mm^2), used to estimate volume.
    start_time:
        Simulation time (s) at which the pass begins depositing.
    duration:
        Pass duration (s); the pass is considered active over
        [start_time, start_time + duration).
    voltage, current, travel_speed, efficiency:
        Arc process parameters feeding the Goldak source for this pass.
    path_description:
        Free-form label for the per-pass trajectory (MVP placeholder).
    """
    order: int
    pass_type: Literal["root", "fill", "cap"] = "fill"
    bead_area: float = 0.0
    start_time: float = 0.0
    duration: float = 0.0
    voltage: float = 25.0
    current: float = 200.0
    travel_speed: float = 5.0
    efficiency: float = 0.8
    path_description: str = "straight"


@dataclass
class WeldSequence:
    """Ordered sequence of weld passes with preheat / interpass constraints."""
    passes: list[WeldPass] = field(default_factory=list)
    preheat_temp: float = 20.0         # degC
    interpass_temp_max: float = 250.0  # degC
    interpass_cooldown_s: float = 0.0  # cooldown time between passes (s)

    def __post_init__(self) -> None:
        # Validate monotone start_times.
        for i in range(1, len(self.passes)):
            if self.passes[i].start_time < self.passes[i - 1].start_time - 1e-9:
                raise ValueError(
                    f"WeldPass start_time must be monotonically non-decreasing "
                    f"at order {self.passes[i].order}"
                )

    def total_duration(self) -> float:
        """Return total simulated duration spanning all passes."""
        if not self.passes:
            return 0.0
        return max(p.start_time + p.duration for p in self.passes)

    def active_pass_at(self, t: float) -> WeldPass | None:
        """Return the pass active at time ``t``, or ``None``."""
        for p in self.passes:
            if p.start_time <= t < p.start_time + p.duration:
                return p
        return None
