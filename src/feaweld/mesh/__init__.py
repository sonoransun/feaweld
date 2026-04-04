"""Mesh generation, quality analysis, and format conversion."""

from feaweld.mesh.convert import (
    femesh_to_dolfinx,
    femesh_to_meshio,
    femesh_to_vtk,
    meshio_to_femesh,
)
from feaweld.mesh.generator import WeldMeshConfig, generate_mesh
from feaweld.mesh.quality import aspect_ratio, jacobian_quality, mesh_quality_report

__all__ = [
    "WeldMeshConfig",
    "aspect_ratio",
    "femesh_to_dolfinx",
    "femesh_to_meshio",
    "femesh_to_vtk",
    "generate_mesh",
    "jacobian_quality",
    "mesh_quality_report",
    "meshio_to_femesh",
]
