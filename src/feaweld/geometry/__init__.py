"""Weld joint geometry builders and notch-radius utilities."""

from feaweld.geometry.groove import (
    GrooveProfile,
    JGroove,
    KGroove,
    UGroove,
    VGroove,
    XGroove,
)
from feaweld.geometry.weld_path import WeldPath

__all__ = [
    "GrooveProfile",
    "JGroove",
    "KGroove",
    "UGroove",
    "VGroove",
    "WeldPath",
    "XGroove",
]

# Gmsh-backed modules are optional: importing them eagerly would crash on
# systems that don't have gmsh installed. Guard the imports so the pure
# Python primitives (WeldPath, groove profiles) remain usable.
try:  # pragma: no cover - import-time guard
    from feaweld.geometry.joints import (  # noqa: F401
        ButtWeld,
        CornerJoint,
        CruciformJoint,
        FilletTJoint,
        JointGeometry,
        LapJoint,
    )

    __all__ += [
        "ButtWeld",
        "CornerJoint",
        "CruciformJoint",
        "FilletTJoint",
        "JointGeometry",
        "LapJoint",
    ]
except ImportError:
    pass

try:  # pragma: no cover - import-time guard
    from feaweld.geometry.notch import (  # noqa: F401
        create_notched_model,
        insert_fictitious_radius,
    )

    __all__ += ["create_notched_model", "insert_fictitious_radius"]
except ImportError:
    pass

try:  # pragma: no cover - import-time guard
    from feaweld.geometry.joints3d import (  # noqa: F401
        VolumetricButtJoint,
        VolumetricFilletTJoint,
    )
    from feaweld.geometry.fastener_welds import (  # noqa: F401
        PlugWeld,
        SlotWeld,
        SpotWeld,
        StudWeld,
    )

    __all__ += [
        "PlugWeld",
        "SlotWeld",
        "SpotWeld",
        "StudWeld",
        "VolumetricButtJoint",
        "VolumetricFilletTJoint",
    ]
except ImportError:
    pass

try:  # pragma: no cover - import-time guard
    from feaweld.geometry.spline_joints import SplineButtWeld  # noqa: F401

    __all__ += ["SplineButtWeld"]
except ImportError:
    pass
