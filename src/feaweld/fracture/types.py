"""Phase-field fracture result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FractureResult:
    """Aggregated results from a phase-field fracture solve.

    Attributes
    ----------
    displacement:
        Final converged displacement field ``(n_nodes, 2)`` for plane-strain
        TRI3, or ``(n_nodes, 3)`` for future 3D support.
    damage:
        Final damage field ``(n_nodes,)`` in the range ``[0, 1]``.
    reaction_force:
        Scalar reaction force at the loaded node set, evaluated at the
        final load step.
    load_steps:
        Load-multiplier history ``(n_steps,)`` — fractions of ``max_load``
        applied at each step.
    reaction_history:
        Reaction force at each load step ``(n_steps,)``.
    damage_history:
        List of length ``n_steps`` of damage field snapshots.
    converged:
        True if the final staggered iteration converged within the tolerance.
    metadata:
        Free-form dict for provenance / diagnostics.
    """

    displacement: np.ndarray
    damage: np.ndarray
    reaction_force: float
    load_steps: np.ndarray
    reaction_history: np.ndarray
    damage_history: list[np.ndarray]
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)
