"""Temperature-dependent material property database for weld analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

import yaml

_DATA_DIR = Path(__file__).parent.parent / "data" / "materials"


@dataclass
class Material:
    """A material with temperature-dependent properties.

    Properties are stored as (temperature, value) pairs and interpolated
    using cubic spline for smooth evaluation at arbitrary temperatures.
    """
    name: str
    density: float  # kg/m^3 (assumed constant)
    category: str | None = None  # e.g. "carbon_steel", "stainless", "filler_metal"

    # Temperature-dependent properties: dict of temperature_C → value
    elastic_modulus: dict[float, float] = field(default_factory=dict)      # MPa
    poisson_ratio: dict[float, float] = field(default_factory=dict)
    yield_strength: dict[float, float] = field(default_factory=dict)      # MPa
    ultimate_strength: dict[float, float] = field(default_factory=dict)   # MPa
    thermal_conductivity: dict[float, float] = field(default_factory=dict)  # W/(m·K)
    specific_heat: dict[float, float] = field(default_factory=dict)       # J/(kg·K)
    thermal_expansion: dict[float, float] = field(default_factory=dict)   # 1/K

    # Creep parameters (Norton-Bailey: ε̇_cr = A · σ^n · t^m)
    creep_A: float = 0.0
    creep_n: float = 1.0
    creep_m: float = 0.0

    # Hardening parameters (isotropic: σ_y = σ_y0 + H * ε_p^n_hard)
    hardening_modulus: float = 0.0   # H (MPa)
    hardening_exponent: float = 1.0  # n_hard

    def __post_init__(self) -> None:
        self._interpolators: dict[str, interp1d] = {}

    def _get_interpolator(self, prop_name: str):
        if prop_name not in self._interpolators:
            data: dict[float, float] = getattr(self, prop_name)
            temps = np.array(sorted(data.keys()))
            vals = np.array([data[t] for t in temps])
            if len(temps) == 0:
                raise ValueError(
                    f"Material '{self.name}' property '{prop_name}' has no data points"
                )
            if len(temps) == 1:
                # Single data point: return constant function
                val = float(vals[0])
                self._interpolators[prop_name] = lambda T, _v=val: _v
                return self._interpolators[prop_name]
            kind = "cubic" if len(temps) >= 4 else "linear"
            self._interpolators[prop_name] = interp1d(
                temps, vals, kind=kind, fill_value="extrapolate"
            )
        return self._interpolators[prop_name]

    def E(self, T: float) -> float:
        """Elastic modulus at temperature T (C)."""
        return float(self._get_interpolator("elastic_modulus")(T))

    def nu(self, T: float) -> float:
        """Poisson's ratio at temperature T (C)."""
        return float(self._get_interpolator("poisson_ratio")(T))

    def sigma_y(self, T: float) -> float:
        """Yield strength at temperature T (C)."""
        return float(self._get_interpolator("yield_strength")(T))

    def sigma_u(self, T: float) -> float:
        """Ultimate tensile strength at temperature T (C)."""
        return float(self._get_interpolator("ultimate_strength")(T))

    def k(self, T: float) -> float:
        """Thermal conductivity at temperature T (C)."""
        return float(self._get_interpolator("thermal_conductivity")(T))

    def cp(self, T: float) -> float:
        """Specific heat at temperature T (C)."""
        return float(self._get_interpolator("specific_heat")(T))

    def alpha(self, T: float) -> float:
        """Coefficient of thermal expansion at temperature T (C)."""
        return float(self._get_interpolator("thermal_expansion")(T))

    def lame_lambda(self, T: float) -> float:
        """First Lame parameter at temperature T."""
        E_val = self.E(T)
        nu_val = self.nu(T)
        if nu_val >= 0.5:
            raise ValueError(
                f"Poisson ratio {nu_val:.4f} at T={T} C is >= 0.5 "
                "(incompressible); Lame lambda is undefined."
            )
        return E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

    def lame_mu(self, T: float) -> float:
        """Shear modulus (second Lame parameter) at temperature T."""
        return self.E(T) / (2 * (1 + self.nu(T)))

    def elasticity_tensor_2d(self, T: float, plane: str = "stress") -> NDArray:
        """2D elasticity matrix (3x3) for plane stress or plane strain."""
        E_val = self.E(T)
        nu_val = self.nu(T)
        if plane == "stress":
            if abs(nu_val) >= 1.0:
                raise ValueError(
                    f"|Poisson ratio| = {abs(nu_val):.4f} >= 1.0 at T={T} C; "
                    "plane-stress elasticity tensor is undefined."
                )
            factor = E_val / (1 - nu_val**2)
            return factor * np.array([
                [1, nu_val, 0],
                [nu_val, 1, 0],
                [0, 0, (1 - nu_val) / 2],
            ])
        else:  # plane strain
            if nu_val >= 0.5:
                raise ValueError(
                    f"Poisson ratio {nu_val:.4f} at T={T} C is >= 0.5; "
                    "plane-strain elasticity tensor is undefined."
                )
            factor = E_val / ((1 + nu_val) * (1 - 2 * nu_val))
            return factor * np.array([
                [1 - nu_val, nu_val, 0],
                [nu_val, 1 - nu_val, 0],
                [0, 0, (1 - 2 * nu_val) / 2],
            ])

    def elasticity_tensor_3d(self, T: float) -> NDArray:
        """3D elasticity matrix (6x6) in Voigt notation."""
        lam = self.lame_lambda(T)
        mu = self.lame_mu(T)
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return C


@dataclass
class MaterialSet:
    """Groups materials for different weld zones."""
    base_metal: Material
    weld_metal: Material
    haz: Material  # heat-affected zone

    def for_region(self, region_name: str) -> Material:
        mapping = {
            "base": self.base_metal,
            "base_metal": self.base_metal,
            "weld": self.weld_metal,
            "weld_metal": self.weld_metal,
            "haz": self.haz,
            "heat_affected_zone": self.haz,
        }
        return mapping[region_name.lower()]


def load_material(name: str, data_dir: Path | None = None) -> Material:
    """Load a material from a YAML file."""
    search_dir = data_dir or _DATA_DIR
    path = search_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Material file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Convert lists of [temp, value] pairs to dicts
    props = {}
    for key in [
        "elastic_modulus", "poisson_ratio", "yield_strength",
        "ultimate_strength", "thermal_conductivity", "specific_heat",
        "thermal_expansion",
    ]:
        if key in data:
            raw = data[key]
            if isinstance(raw, list):
                props[key] = {float(pair[0]): float(pair[1]) for pair in raw}
            elif isinstance(raw, dict):
                props[key] = {float(k): float(v) for k, v in raw.items()}
            else:
                # Single value: use at 20C and 500C (assume constant)
                props[key] = {20.0: float(raw), 500.0: float(raw)}

    return Material(
        name=data.get("name", name),
        density=float(data.get("density", 7850)),
        category=data.get("category", None),
        creep_A=float(data.get("creep_A", 0)),
        creep_n=float(data.get("creep_n", 1)),
        creep_m=float(data.get("creep_m", 0)),
        hardening_modulus=float(data.get("hardening_modulus", 0)),
        hardening_exponent=float(data.get("hardening_exponent", 1)),
        **props,
    )


def list_available_materials(data_dir: Path | None = None) -> list[str]:
    """List names of available material YAML files."""
    search_dir = data_dir or _DATA_DIR
    return sorted(p.stem for p in search_dir.glob("*.yaml"))


def search_materials(query: str, data_dir: Path | None = None) -> list[str]:
    """Search materials by case-insensitive substring match on file name and YAML name."""
    search_dir = data_dir or _DATA_DIR
    q = query.lower()
    results = []
    for p in search_dir.glob("*.yaml"):
        if q in p.stem.lower():
            results.append(p.stem)
            continue
        # Also check the 'name' field inside the YAML
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            if q in data.get("name", "").lower():
                results.append(p.stem)
        except Exception:
            pass
    return sorted(results)


def list_material_categories(data_dir: Path | None = None) -> dict[str, list[str]]:
    """List available materials grouped by category.

    Returns dict mapping category names to lists of material file names.
    Materials without a category are grouped under "uncategorized".
    """
    search_dir = data_dir or _DATA_DIR
    categories: dict[str, list[str]] = {}
    for p in sorted(search_dir.glob("*.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            cat = data.get("category", "uncategorized") or "uncategorized"
        except Exception:
            cat = "uncategorized"
        categories.setdefault(cat, []).append(p.stem)
    return categories
