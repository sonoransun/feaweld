"""Smoke tests for feaweld documentation asset generators.

Parameterizes over every mermaid diagram, concept image, and animation
that ships with the docs, asserting that running the generator
produces a non-empty file within a sensible size bound. Generation is
gated behind ``matplotlib`` / ``imageio`` ``importorskip`` so the
baseline suite stays green when the ``[docs]`` extras are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


MERMAID_STEMS = [
    "architecture_pipeline",
    "solver_backend_hierarchy",
    "constitutive_hierarchy",
    "neural_operator_pipeline",
    "active_learning_loop",
    "digital_twin_bayesian_loop",
    "defect_insertion_workflow",
    "multipass_welding_sequence",
]

CONCEPT_IMAGE_STEMS = [
    "jax_backend_flow",
    "j2_radial_return",
    "crystal_plasticity_fcc",
    "phase_field_schematic",
    "deeponet_architecture",
    "bayesian_ensemble_uq",
    "active_learning_acquisition",
    "fft_homogenization_rve",
    "enkf_state_space",
    "normalizing_flow_density",
    "spline_weld_path",
    "groove_profile_gallery",
    "volumetric_joint_render",
    "fastener_welds_gallery",
    "defect_type_gallery",
    "fat_downgrade_curves",
    "iso5817_population_sample",
    "multiaxial_critical_plane",
    "j_integral_qfunction",
    "ctod_extrapolation",
    "volumetric_sed_control",
    "multipass_bead_stacking",
    "rainflow_multiaxial_projection",
]

ANIMATION_STEMS = [
    "phase_field_crack_propagation",
    "multipass_thermal_cycle",
    "goldak_heat_source_sweep",
    "active_learning_convergence",
    "monte_carlo_convergence",
    "enkf_crack_tracking",
    "bayesian_posterior_update",
    "rainflow_cycle_counting",
    "cyclic_stress_field",
]


# ---------------------------------------------------------------------------
# Mermaid diagrams
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", MERMAID_STEMS)
def test_mermaid_diagram_exists_and_valid(stem: str) -> None:
    """Each mermaid .mmd file exists, is non-empty, and starts with a
    valid diagram-type header.
    """
    path = _ROOT / "docs" / "diagrams" / f"{stem}.mmd"
    assert path.exists(), f"missing diagram: {path}"
    content = path.read_text()
    assert len(content) >= 50, f"diagram too short: {path}"
    headers = ("flowchart", "classDiagram", "stateDiagram-v2",
               "sequenceDiagram", "graph")
    first = content.strip().splitlines()[0].strip()
    assert any(first.startswith(h) for h in headers), (
        f"{path} does not start with a valid mermaid header: {first!r}"
    )


def test_mermaid_generator_reruns_cleanly() -> None:
    """Regenerating mermaid diagrams does not raise and produces all 8 files."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gen_mermaid", _ROOT / "scripts" / "generate_docs_mermaid.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    paths = mod.main()
    assert len(paths) == len(MERMAID_STEMS)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Concept images (SVG + PNG)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", CONCEPT_IMAGE_STEMS)
def test_concept_image_exists(stem: str) -> None:
    """Each concept image ships as an SVG + PNG pair, within size budget."""
    svg = _ROOT / "docs" / "images" / f"{stem}.svg"
    png = _ROOT / "docs" / "images" / f"{stem}.png"
    assert svg.exists(), f"missing SVG: {svg}"
    assert png.exists(), f"missing PNG: {png}"
    svg_kb = svg.stat().st_size / 1024
    png_kb = png.stat().st_size / 1024
    assert 1 <= svg_kb <= 6000, f"{svg} out of size budget ({svg_kb:.1f} KB)"
    assert 1 <= png_kb <= 6000, f"{png} out of size budget ({png_kb:.1f} KB)"


@pytest.mark.slow
def test_concept_image_generator_reruns_cleanly() -> None:
    """Regenerating concept images does not raise. Requires matplotlib."""
    pytest.importorskip("matplotlib")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gen_concept",
        _ROOT / "scripts" / "generate_docs_concept_images.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Use a tiny subset so the test stays fast even on slow CI.
    paths = mod.main(["--only", "j2_radial_return", "groove_profile_gallery"])
    assert len(paths) == 4  # 2 stems × (svg + png)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Animations (GIF + MP4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", ANIMATION_STEMS)
def test_animation_exists(stem: str) -> None:
    gif = _ROOT / "docs" / "animations" / f"{stem}.gif"
    mp4 = _ROOT / "docs" / "animations" / f"{stem}.mp4"
    assert gif.exists(), f"missing GIF: {gif}"
    assert mp4.exists(), f"missing MP4: {mp4}"
    gif_mb = gif.stat().st_size / 1024 / 1024
    mp4_mb = mp4.stat().st_size / 1024 / 1024
    assert gif_mb < 5.0, f"{gif} exceeds 5 MB budget ({gif_mb:.2f} MB)"
    assert mp4_mb < 10.0, f"{mp4} exceeds 10 MB budget ({mp4_mb:.2f} MB)"


@pytest.mark.slow
def test_animation_generator_reruns_cleanly() -> None:
    """Regenerating a single animation does not raise. Requires matplotlib + imageio."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gen_anim", _ROOT / "scripts" / "generate_docs_animations.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Pick the cheapest animation.
    paths = mod.main(["--only", "bayesian_posterior_update"])
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# CONCEPTS.md master index
# ---------------------------------------------------------------------------

def test_concepts_md_references_every_asset() -> None:
    """CONCEPTS.md must reference every mermaid block, image, and animation
    so the catalog stays in sync with the asset set on disk.
    """
    path = _ROOT / "docs" / "CONCEPTS.md"
    assert path.exists()
    text = path.read_text()

    missing = []
    for stem in CONCEPT_IMAGE_STEMS:
        if stem not in text:
            missing.append(f"images/{stem}")
    for stem in ANIMATION_STEMS:
        if stem not in text:
            missing.append(f"animations/{stem}")
    assert not missing, f"CONCEPTS.md does not reference: {missing}"


def test_readme_has_advanced_concepts_section() -> None:
    path = _ROOT / "README.md"
    assert path.exists()
    text = path.read_text()
    assert "<!-- ADVANCED_CONCEPTS_START -->" in text
    assert "<!-- ADVANCED_CONCEPTS_END -->" in text
    assert "docs/CONCEPTS.md" in text
