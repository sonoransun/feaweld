"""Weld defect data model, acceptance criteria, and FAT downgrade helpers."""

from __future__ import annotations

from feaweld.defects.knockdown import (
    KnockdownResult,
    defect_fat_downgrade,
    lof_fat_downgrade,
    porosity_fat_downgrade,
    slag_fat_downgrade,
    undercut_fat_downgrade,
)
from feaweld.defects.loader import load_acceptance_criteria
from feaweld.defects.population import (
    sample_iso5817_population,
    validate_population,
)
from feaweld.defects.types import (
    ClusterPorosity,
    Defect,
    LackOfFusionDefect,
    PoreDefect,
    RootGapDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
)

# Gmsh-dependent insertion helpers.  The insertion module itself guards
# the ``import gmsh`` call so it is safe to import even without gmsh
# installed — individual helper calls raise ImportError at call time
# instead.  We still wrap the import in try/except defensively in case
# a future helper gains a hard top-level gmsh dependency.
try:  # pragma: no cover - import is exercised when gmsh is present
    from feaweld.defects.insertion import (
        insert_all,
        insert_cluster_porosity,
        insert_defect,
        insert_lack_of_fusion,
        insert_pore,
        insert_root_gap,
        insert_slag_inclusion,
        insert_surface_crack,
        insert_undercut,
    )

    _HAS_INSERTION = True
except ImportError:  # pragma: no cover - hit only when gmsh is missing
    _HAS_INSERTION = False

__all__ = [
    "ClusterPorosity",
    "Defect",
    "KnockdownResult",
    "LackOfFusionDefect",
    "PoreDefect",
    "RootGapDefect",
    "SlagInclusion",
    "SurfaceCrack",
    "UndercutDefect",
    "defect_fat_downgrade",
    "load_acceptance_criteria",
    "lof_fat_downgrade",
    "porosity_fat_downgrade",
    "sample_iso5817_population",
    "slag_fat_downgrade",
    "undercut_fat_downgrade",
    "validate_population",
]

if _HAS_INSERTION:
    __all__ += [
        "insert_all",
        "insert_cluster_porosity",
        "insert_defect",
        "insert_lack_of_fusion",
        "insert_pore",
        "insert_root_gap",
        "insert_slag_inclusion",
        "insert_surface_crack",
        "insert_undercut",
    ]
