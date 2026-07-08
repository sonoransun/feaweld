"""Tests for fictitious notch-radius insertion (effective notch stress).

These drive the live Gmsh model via ``feaweld.geometry.notch`` and so require
Gmsh; each test runs inside a clean, self-finalizing Gmsh session so it does
not leak model state into the rest of the suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gmsh")

import gmsh  # noqa: E402

from feaweld.geometry.joints import FilletTJoint  # noqa: E402
from feaweld.geometry.notch import (  # noqa: E402
    create_notched_model,
    insert_fictitious_radius,
)

pytestmark = pytest.mark.requires_gmsh


@pytest.fixture
def gmsh_session():
    """Provide a fresh Gmsh session and finalize it afterwards."""
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        yield gmsh
    finally:
        if gmsh.is_initialized():
            gmsh.finalize()


def test_insert_fictitious_radius_empty_is_noop(gmsh_session):
    """An empty toe-point list must not touch the model or raise."""
    gmsh_session.model.add("empty")
    insert_fictitious_radius([])
    # No surfaces were created.
    assert gmsh_session.model.occ.getEntities(2) == []


def test_create_notched_model_leaves_2d_entities(gmsh_session):
    """Notching a fillet T-joint yields a meshable 2D model."""
    joint = FilletTJoint(
        base_width=60.0, base_thickness=10.0, web_height=40.0,
        web_thickness=8.0, weld_leg_size=6.0,
    )
    create_notched_model(joint, radius=1.0)

    surfaces = gmsh_session.model.occ.getEntities(2)
    assert len(surfaces) > 0
