"""Export utilities: VTK files, PNG screenshots, and glTF 3-D exports.

PyVista is imported lazily inside each function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, FEMesh, StressField


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def export_vtk(
    mesh: FEMesh,
    results: FEAResults,
    filename: str,
) -> None:
    """Write a VTK file containing all available result fields.

    The file format is determined by the extension:

    * ``.vtk``  — legacy VTK format
    * ``.vtu``  — VTK XML UnstructuredGrid (recommended)

    Parameters
    ----------
    mesh:
        The FE mesh.
    results:
        FEA results to embed.
    filename:
        Output file path.
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for VTK export. "
            "Install with: pip install pyvista"
        ) from exc

    from feaweld.visualization.stress_plots import (
        _VTK_CELL_TYPE,
        _mesh_to_pyvista_grid,
    )

    grid = _mesh_to_pyvista_grid(mesh)

    # Attach all available fields.
    if results.displacement is not None:
        grid.point_data["displacement"] = results.displacement

    if results.stress is not None:
        component_names = [
            "stress_xx", "stress_yy", "stress_zz",
            "stress_xy", "stress_yz", "stress_xz",
        ]
        for i, name in enumerate(component_names):
            grid.point_data[name] = results.stress.values[:, i]
        grid.point_data["von_mises"] = results.stress.von_mises
        grid.point_data["tresca"] = results.stress.tresca
        principals = results.stress.principal
        grid.point_data["principal_1"] = principals[:, 2]
        grid.point_data["principal_2"] = principals[:, 1]
        grid.point_data["principal_3"] = principals[:, 0]

    if results.strain is not None:
        strain_names = [
            "strain_xx", "strain_yy", "strain_zz",
            "strain_xy", "strain_yz", "strain_xz",
        ]
        for i, name in enumerate(strain_names):
            grid.point_data[name] = results.strain[:, i]

    if results.temperature is not None:
        temp = results.temperature
        if temp.ndim == 1:
            grid.point_data["temperature"] = temp
        else:
            # Transient: store the last time step.
            grid.point_data["temperature"] = temp[-1]

    if results.nodal_forces is not None:
        grid.point_data["reaction_force"] = results.nodal_forces

    # Ensure parent directory exists.
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    grid.save(filename)


# ---------------------------------------------------------------------------
# PNG screenshot
# ---------------------------------------------------------------------------

def export_png(
    plotter: Any,
    filename: str,
    resolution: tuple[int, int] = (1920, 1080),
) -> None:
    """Save a screenshot of an existing PyVista plotter.

    Parameters
    ----------
    plotter:
        A ``pyvista.Plotter`` instance (already configured with meshes).
    filename:
        Output PNG path.
    resolution:
        ``(width, height)`` in pixels.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plotter.window_size = list(resolution)
    plotter.screenshot(filename)


# ---------------------------------------------------------------------------
# glTF (3-D web) export
# ---------------------------------------------------------------------------

def export_gltf(
    plotter: Any,
    filename: str,
) -> None:
    """Export the scene to a glTF 2.0 file.

    Parameters
    ----------
    plotter:
        A ``pyvista.Plotter`` instance.
    filename:
        Output ``.gltf`` file path.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plotter.export_gltf(filename)
