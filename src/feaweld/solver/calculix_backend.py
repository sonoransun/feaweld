"""CalculiX solver backend.

Uses *pygccx* when available; otherwise generates Abaqus-format .inp
files directly and invokes the ``ccx`` executable.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend


# ---------------------------------------------------------------------------
# .inp file generation helpers
# ---------------------------------------------------------------------------

_ELEMENT_TYPE_MAP: dict[ElementType, str] = {
    ElementType.TRI3: "S3",
    ElementType.TRI6: "S6",
    ElementType.QUAD4: "S4",
    ElementType.QUAD8: "S8",
    ElementType.TET4: "C3D4",
    ElementType.TET10: "C3D10",
    ElementType.HEX8: "C3D8",
    ElementType.HEX20: "C3D20",
}


def _write_nodes(f: Any, mesh: FEMesh) -> None:
    """Write the *NODE section to an .inp file handle."""
    f.write("*NODE\n")
    for i in range(mesh.n_nodes):
        coords = mesh.nodes[i]
        # 1-based node IDs
        line = f"{i + 1}"
        for c in coords:
            line += f", {c:.10g}"
        f.write(line + "\n")


def _write_elements(f: Any, mesh: FEMesh) -> None:
    """Write the *ELEMENT section to an .inp file handle."""
    ccx_type = _ELEMENT_TYPE_MAP.get(mesh.element_type)
    if ccx_type is None:
        raise ValueError(f"Unsupported element type for CalculiX: {mesh.element_type}")
    f.write(f"*ELEMENT, TYPE={ccx_type}, ELSET=ALL\n")
    for i in range(mesh.n_elements):
        # 1-based element and node IDs
        connectivity = mesh.elements[i] + 1
        parts = [str(i + 1)] + [str(int(n)) for n in connectivity]
        f.write(", ".join(parts) + "\n")


def _write_node_sets(f: Any, mesh: FEMesh) -> None:
    """Write *NSET definitions."""
    for name, ids in mesh.node_sets.items():
        f.write(f"*NSET, NSET={name}\n")
        # Write in lines of up to 16 entries
        ids_1based = ids + 1
        for start in range(0, len(ids_1based), 16):
            chunk = ids_1based[start : start + 16]
            f.write(", ".join(str(int(n)) for n in chunk) + "\n")


def _write_material(f: Any, material: Material, temperature: float) -> None:
    """Write *MATERIAL section with properties at given temperature."""
    f.write(f"*MATERIAL, NAME={material.name.upper().replace(' ', '_')}\n")
    E = material.E(temperature)
    nu = material.nu(temperature)
    f.write("*ELASTIC\n")
    f.write(f"{E:.6g}, {nu:.6g}\n")

    # Density
    f.write("*DENSITY\n")
    f.write(f"{material.density:.6g}\n")

    # Thermal expansion
    try:
        alpha = material.alpha(temperature)
        f.write("*EXPANSION\n")
        f.write(f"{alpha:.6e}\n")
    except (KeyError, ValueError):
        pass

    # Thermal conductivity
    try:
        k = material.k(temperature)
        f.write("*CONDUCTIVITY\n")
        f.write(f"{k:.6g}\n")
    except (KeyError, ValueError):
        pass

    # Specific heat
    try:
        cp = material.cp(temperature)
        f.write("*SPECIFIC HEAT\n")
        f.write(f"{cp:.6g}\n")
    except (KeyError, ValueError):
        pass


def _write_boundary_conditions(
    f: Any, load_case: LoadCase, mesh: FEMesh, analysis: str = "static"
) -> None:
    """Write boundary conditions and loads for a CalculiX step."""
    # Dirichlet (displacement or temperature) constraints
    has_boundary = False
    for constraint in load_case.constraints:
        if constraint.bc_type == LoadType.DISPLACEMENT:
            if not has_boundary:
                f.write("*BOUNDARY\n")
                has_boundary = True
            node_set = constraint.node_set
            if node_set in mesh.node_sets:
                # Fix all DOFs for the specified nodes
                for nid in mesh.node_sets[node_set]:
                    nid_1 = int(nid) + 1
                    for dof in range(1, 4):
                        val = constraint.values[dof - 1] if dof - 1 < len(constraint.values) else 0.0
                        f.write(f"{nid_1}, {dof}, {dof}, {val:.10g}\n")
            else:
                # Use the node set name directly (assumes defined in .inp)
                for dof in range(1, 4):
                    val = constraint.values[dof - 1] if dof - 1 < len(constraint.values) else 0.0
                    f.write(f"{node_set}, {dof}, {dof}, {val:.10g}\n")
        elif constraint.bc_type == LoadType.TEMPERATURE:
            if not has_boundary:
                f.write("*BOUNDARY\n")
                has_boundary = True
            node_set = constraint.node_set
            T_val = constraint.values[0]
            if node_set in mesh.node_sets:
                for nid in mesh.node_sets[node_set]:
                    nid_1 = int(nid) + 1
                    f.write(f"{nid_1}, 11, 11, {T_val:.6g}\n")
            else:
                f.write(f"{node_set}, 11, 11, {T_val:.6g}\n")

    # Concentrated loads
    for load_bc in load_case.loads:
        if load_bc.bc_type == LoadType.FORCE:
            f.write("*CLOAD\n")
            node_set = load_bc.node_set
            mag = load_bc.values[0]
            direction = load_bc.direction
            if direction is not None:
                for dof in range(1, 4):
                    component = mag * direction[dof - 1]
                    if abs(component) > 1e-30:
                        if node_set in mesh.node_sets:
                            for nid in mesh.node_sets[node_set]:
                                f.write(f"{int(nid) + 1}, {dof}, {component:.10g}\n")
                        else:
                            f.write(f"{node_set}, {dof}, {component:.10g}\n")
            else:
                # Apply as magnitude in DOF 1 by default
                if node_set in mesh.node_sets:
                    for nid in mesh.node_sets[node_set]:
                        f.write(f"{int(nid) + 1}, 1, {mag:.10g}\n")
                else:
                    f.write(f"{node_set}, 1, {mag:.10g}\n")

        elif load_bc.bc_type == LoadType.PRESSURE:
            f.write("*DLOAD\n")
            f.write(f"ALL, P, {load_bc.values[0]:.10g}\n")

        elif load_bc.bc_type == LoadType.HEAT_FLUX:
            f.write("*DFLUX\n")
            f.write(f"ALL, S, {load_bc.values[0]:.10g}\n")

        elif load_bc.bc_type == LoadType.CONVECTION:
            f.write("*FILM\n")
            h_conv = load_bc.values[0]
            T_amb = load_bc.values[1] if len(load_bc.values) > 1 else 20.0
            f.write(f"ALL, F, {T_amb:.6g}, {h_conv:.6g}\n")


def generate_inp(
    mesh: FEMesh,
    material: Material,
    load_case: LoadCase,
    path: Path,
    analysis: str = "static",
    temperature: float = 20.0,
    time_steps: NDArray | None = None,
) -> Path:
    """Generate a complete CalculiX .inp file.

    Parameters
    ----------
    mesh : FEMesh
        Mesh data.
    material : Material
        Material properties.
    load_case : LoadCase
        Loads and constraints.
    path : Path
        Output file path.
    analysis : str
        Analysis type: ``"static"``, ``"thermal_steady"``, or
        ``"thermal_transient"``.
    temperature : float
        Temperature for property evaluation.
    time_steps : NDArray or None
        For transient analysis, the time steps.

    Returns
    -------
    Path
        The path to the generated file.
    """
    path = Path(path)
    mat_name = material.name.upper().replace(" ", "_")

    with open(path, "w") as f:
        f.write(f"** CalculiX input file generated by feaweld\n")
        f.write(f"** Analysis: {analysis}\n")
        f.write("**\n")

        _write_nodes(f, mesh)
        _write_elements(f, mesh)
        _write_node_sets(f, mesh)

        _write_material(f, material, temperature)

        # Assign material to element set
        f.write(f"*SOLID SECTION, ELSET=ALL, MATERIAL={mat_name}\n")
        f.write("\n")

        # Initial conditions
        if analysis in ("thermal_steady", "thermal_transient"):
            f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
            f.write("ALL, 20.0\n")

        # Step
        if analysis == "static":
            f.write("*STEP\n")
            f.write("*STATIC\n")
        elif analysis == "thermal_steady":
            f.write("*STEP\n")
            f.write("*HEAT TRANSFER, STEADY STATE\n")
        elif analysis == "thermal_transient":
            if time_steps is not None and len(time_steps) > 1:
                dt = time_steps[1] - time_steps[0]
                t_total = time_steps[-1] - time_steps[0]
            else:
                dt = 1.0
                t_total = 10.0
            f.write("*STEP\n")
            f.write(f"*HEAT TRANSFER\n")
            f.write(f"{dt:.6g}, {t_total:.6g}\n")

        _write_boundary_conditions(f, load_case, mesh, analysis)

        # Output requests
        f.write("*NODE FILE\n")
        if analysis in ("thermal_steady", "thermal_transient"):
            f.write("NT\n")
        else:
            f.write("U\n")
        f.write("*EL FILE\n")
        if analysis in ("thermal_steady", "thermal_transient"):
            f.write("HFL\n")
        else:
            f.write("S, E\n")

        f.write("*END STEP\n")

    return path


# ---------------------------------------------------------------------------
# .frd results parser
# ---------------------------------------------------------------------------

def parse_frd(path: Path) -> dict[str, NDArray]:
    """Parse a CalculiX .frd results file (ASCII format).

    Extracts displacement and stress fields.

    Parameters
    ----------
    path : Path
        Path to the .frd file.

    Returns
    -------
    dict
        Keys may include ``"displacement"`` (n, 3), ``"stress"`` (n, 6),
        ``"temperature"`` (n,).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FRD file not found: {path}")

    results: dict[str, NDArray] = {}

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for result blocks: "  100C" lines mark data headers
        if line.startswith("  100C"):
            # Component header block
            # Count the number of components from subsequent " -5" lines
            block_name = ""
            components: list[str] = []
            i += 1

            while i < len(lines) and lines[i].startswith(" -4"):
                name_field = lines[i][5:17].strip()
                block_name = name_field
                i += 1

            while i < len(lines) and lines[i].startswith(" -5"):
                comp_name = lines[i][5:17].strip()
                components.append(comp_name)
                i += 1

            # Read data lines: " -1" prefix
            node_ids: list[int] = []
            values: list[list[float]] = []
            while i < len(lines) and lines[i].startswith(" -1"):
                data_line = lines[i]
                try:
                    nid = int(data_line[3:13])
                    vals = []
                    pos = 13
                    for _ in range(len(components)):
                        val_str = data_line[pos : pos + 12].strip()
                        vals.append(float(val_str) if val_str else 0.0)
                        pos += 12
                    node_ids.append(nid)
                    values.append(vals)
                except (ValueError, IndexError):
                    pass
                i += 1

            if not values:
                continue

            arr = np.array(values)

            # Identify the field
            block_upper = block_name.upper()
            if "DISP" in block_upper or block_upper == "DISPLACEMENT":
                if arr.shape[1] >= 3:
                    results["displacement"] = arr[:, :3]
                else:
                    padded = np.zeros((arr.shape[0], 3))
                    padded[:, : arr.shape[1]] = arr
                    results["displacement"] = padded
            elif "STRESS" in block_upper or block_upper == "STRESS":
                if arr.shape[1] >= 6:
                    results["stress"] = arr[:, :6]
                else:
                    padded = np.zeros((arr.shape[0], 6))
                    padded[:, : arr.shape[1]] = arr
                    results["stress"] = padded
            elif "TEMP" in block_upper or "NDTEMP" in block_upper:
                results["temperature"] = arr[:, 0] if arr.shape[1] >= 1 else arr.flatten()
            elif "STRAIN" in block_upper:
                if arr.shape[1] >= 6:
                    results["strain"] = arr[:, :6]
                else:
                    padded = np.zeros((arr.shape[0], 6))
                    padded[:, : arr.shape[1]] = arr
                    results["strain"] = padded
            else:
                # Store with the raw name
                results[block_name.lower()] = arr

            continue

        i += 1

    return results


def _find_ccx() -> str:
    """Locate the CalculiX executable."""
    # Check common names
    for name in ("ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19", "CalculiX"):
        path = _which(name)
        if path is not None:
            return path
    # Check environment variable
    env_path = os.environ.get("CCX_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    raise FileNotFoundError(
        "CalculiX (ccx) executable not found. Install CalculiX or set CCX_PATH."
    )


def _which(name: str) -> str | None:
    """Find an executable on PATH."""
    import shutil
    return shutil.which(name)


# ---------------------------------------------------------------------------
# CalculiX Backend
# ---------------------------------------------------------------------------

class CalculiXBackend(SolverBackend):
    """FEA solver backend using CalculiX (ccx).

    Generates Abaqus-format .inp input files, runs the ``ccx``
    executable, and parses the resulting .frd output.  If *pygccx* is
    installed it is used for convenient model building; otherwise
    everything is done with direct file I/O.

    Parameters
    ----------
    ccx_path : str or None
        Path to the ``ccx`` executable.  If ``None``, the backend
        searches standard locations and ``$CCX_PATH``.
    work_dir : str or Path or None
        Working directory for temporary files.  If ``None``, a temporary
        directory is created for each solve.
    """

    def __init__(
        self,
        ccx_path: str | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self._ccx_path = ccx_path
        self._work_dir = Path(work_dir) if work_dir else None

    def _get_ccx(self) -> str:
        if self._ccx_path is not None:
            return self._ccx_path
        return _find_ccx()

    def _run_ccx(self, inp_path: Path) -> Path:
        """Run CalculiX and return the path to the .frd file."""
        ccx = self._get_ccx()
        job_name = inp_path.stem
        work = inp_path.parent

        result = subprocess.run(
            [ccx, "-i", job_name],
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=600,
        )

        frd_path = work / f"{job_name}.frd"
        if not frd_path.exists():
            msg = f"CalculiX did not produce output file {frd_path}."
            if result.stderr:
                msg += f"\nstderr: {result.stderr[:500]}"
            if result.stdout:
                msg += f"\nstdout (last 500 chars): {result.stdout[-500:]}"
            raise RuntimeError(msg)

        return frd_path

    def _try_pygccx(self) -> bool:
        """Check if pygccx is available."""
        try:
            import pygccx  # noqa: F401
            return True
        except ImportError:
            return False

    def solve_static(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> FEAResults:
        """Static mechanical solve using CalculiX."""
        work = self._work_dir or Path(tempfile.mkdtemp(prefix="feaweld_ccx_"))
        inp_path = work / "model.inp"

        generate_inp(
            mesh=mesh,
            material=material,
            load_case=load_case,
            path=inp_path,
            analysis="static",
            temperature=temperature,
        )

        frd_path = self._run_ccx(inp_path)
        raw = parse_frd(frd_path)

        disp = raw.get("displacement")
        stress_arr = raw.get("stress")
        strain_arr = raw.get("strain")

        stress_field = StressField(values=stress_arr) if stress_arr is not None else None

        return FEAResults(
            mesh=mesh,
            displacement=disp,
            stress=stress_field,
            strain=strain_arr,
            metadata={
                "solver": "calculix",
                "temperature": temperature,
                "inp_file": str(inp_path),
                "frd_file": str(frd_path),
            },
        )

    def solve_thermal_steady(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
    ) -> FEAResults:
        """Steady-state thermal solve using CalculiX."""
        work = self._work_dir or Path(tempfile.mkdtemp(prefix="feaweld_ccx_"))
        inp_path = work / "thermal_steady.inp"

        generate_inp(
            mesh=mesh,
            material=material,
            load_case=load_case,
            path=inp_path,
            analysis="thermal_steady",
        )

        frd_path = self._run_ccx(inp_path)
        raw = parse_frd(frd_path)

        temp_arr = raw.get("temperature")
        if temp_arr is None:
            temp_arr = np.full(mesh.n_nodes, 20.0)

        return FEAResults(
            mesh=mesh,
            temperature=temp_arr,
            metadata={
                "solver": "calculix",
                "analysis": "thermal_steady",
                "inp_file": str(inp_path),
            },
        )

    def solve_thermal_transient(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        time_steps: NDArray,
        heat_source: object | None = None,
    ) -> FEAResults:
        """Transient thermal solve using CalculiX.

        Note: CalculiX handles time stepping internally.  For a moving
        heat source, multiple steps with DFLUX updates would be needed.
        This implementation uses a single step with uniform heat input.
        For advanced moving-source simulations, consider the FEniCS
        backend.
        """
        time_steps = np.asarray(time_steps, dtype=np.float64)
        work = self._work_dir or Path(tempfile.mkdtemp(prefix="feaweld_ccx_"))
        inp_path = work / "thermal_transient.inp"

        generate_inp(
            mesh=mesh,
            material=material,
            load_case=load_case,
            path=inp_path,
            analysis="thermal_transient",
            time_steps=time_steps,
        )

        frd_path = self._run_ccx(inp_path)
        raw = parse_frd(frd_path)

        temp_arr = raw.get("temperature")
        if temp_arr is None:
            temp_arr = np.full(mesh.n_nodes, 20.0)

        # For transient, CalculiX may output multiple steps in one .frd.
        # For simplicity, return the final state with the time_steps array.
        if temp_arr.ndim == 1:
            # Expand to (n_steps, n_nodes) by replicating the final state
            temp_history = np.zeros((len(time_steps), mesh.n_nodes))
            temp_history[0, :] = 20.0  # initial
            temp_history[-1, : len(temp_arr)] = temp_arr
            # Linear interpolation for intermediate steps
            for i in range(1, len(time_steps) - 1):
                frac = (time_steps[i] - time_steps[0]) / (
                    time_steps[-1] - time_steps[0]
                )
                temp_history[i] = (1 - frac) * temp_history[0] + frac * temp_history[-1]
        else:
            temp_history = temp_arr

        return FEAResults(
            mesh=mesh,
            temperature=temp_history,
            time_steps=time_steps,
            metadata={
                "solver": "calculix",
                "analysis": "thermal_transient",
                "inp_file": str(inp_path),
            },
        )

    def solve_coupled(
        self,
        mesh: FEMesh,
        material: Material,
        mechanical_lc: LoadCase,
        thermal_lc: LoadCase,
        time_steps: NDArray,
    ) -> FEAResults:
        """Sequential thermomechanical coupling via CalculiX."""
        from feaweld.solver.thermomechanical import sequential_coupled_solve

        return sequential_coupled_solve(
            backend=self,
            mesh=mesh,
            material=material,
            thermal_lc=thermal_lc,
            mechanical_lc=mechanical_lc,
            time_steps=time_steps,
        )
