"""Tests for the temperature-dependent material database in feaweld.core.materials."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.materials import (
    Material,
    MaterialSet,
    list_available_materials,
    list_material_categories,
    load_material,
    search_materials,
)


def _material(**props) -> Material:
    return Material(name="probe", density=7850.0, **props)


# ---------------------------------------------------------------------------
# Property interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_single_point_is_constant_everywhere(self) -> None:
        mat = _material(elastic_modulus={20.0: 200000.0})
        assert mat.E(20.0) == pytest.approx(200000.0)
        assert mat.E(-50.0) == pytest.approx(200000.0)
        assert mat.E(900.0) == pytest.approx(200000.0)

    def test_two_points_interpolate_linearly(self) -> None:
        mat = _material(elastic_modulus={0.0: 100.0, 100.0: 200.0})
        assert mat.E(50.0) == pytest.approx(150.0)

    def test_two_points_extrapolate_linearly(self) -> None:
        """fill_value='extrapolate' extends the line — no clamping."""
        mat = _material(elastic_modulus={0.0: 100.0, 100.0: 200.0})
        assert mat.E(200.0) == pytest.approx(300.0)
        assert mat.E(-100.0) == pytest.approx(0.0)

    def test_three_points_use_linear_not_cubic(self) -> None:
        """Quadratic data through 3 knots: midpoint follows the chord."""
        mat = _material(yield_strength={0.0: 0.0, 100.0: 10000.0, 200.0: 40000.0})
        assert mat.sigma_y(50.0) == pytest.approx(5000.0)

    def test_four_points_use_cubic(self) -> None:
        """Quadratic data through 4 knots: cubic spline reproduces T^2 exactly."""
        data = {t: t**2 for t in (0.0, 100.0, 200.0, 300.0)}
        mat = _material(yield_strength=data)
        assert mat.sigma_y(150.0) == pytest.approx(22500.0)

    def test_cubic_extrapolates_the_spline(self) -> None:
        data = {t: t**2 for t in (0.0, 100.0, 200.0, 300.0)}
        mat = _material(yield_strength=data)
        assert mat.sigma_y(350.0) == pytest.approx(122500.0)

    def test_empty_property_raises(self) -> None:
        mat = _material()
        with pytest.raises(ValueError, match="has no data points"):
            mat.E(20.0)

    def test_accessors_exact_at_knots(self, steel_material) -> None:
        assert steel_material.E(20.0) == pytest.approx(200000.0)
        assert steel_material.nu(500.0) == pytest.approx(0.30)
        assert steel_material.sigma_y(500.0) == pytest.approx(165.0)
        assert steel_material.sigma_u(20.0) == pytest.approx(400.0)
        assert steel_material.k(20.0) == pytest.approx(51.9)
        assert steel_material.cp(500.0) == pytest.approx(650.0)
        assert steel_material.alpha(20.0) == pytest.approx(11.7e-6)


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------


class TestElasticity:
    @pytest.fixture
    def mat(self) -> Material:
        return _material(
            elastic_modulus={20.0: 200000.0},
            poisson_ratio={20.0: 0.3},
        )

    def test_lame_lambda(self, mat) -> None:
        expected = 200000.0 * 0.3 / (1.3 * 0.4)
        assert mat.lame_lambda(20.0) == pytest.approx(expected)

    def test_lame_lambda_incompressible_raises(self) -> None:
        bad = _material(
            elastic_modulus={20.0: 200000.0}, poisson_ratio={20.0: 0.5}
        )
        with pytest.raises(ValueError, match="incompressible"):
            bad.lame_lambda(20.0)

    def test_lame_mu(self, mat) -> None:
        assert mat.lame_mu(20.0) == pytest.approx(200000.0 / 2.6)

    def test_plane_stress_entries(self, mat) -> None:
        C = mat.elasticity_tensor_2d(20.0, plane="stress")
        factor = 200000.0 / (1 - 0.3**2)
        assert C.shape == (3, 3)
        assert C[0, 0] == pytest.approx(factor)
        assert C[1, 1] == pytest.approx(factor)
        assert C[0, 1] == pytest.approx(factor * 0.3)
        assert C[2, 2] == pytest.approx(factor * 0.35)
        assert C[0, 2] == 0.0 and C[2, 0] == 0.0

    def test_plane_stress_nu_ge_one_raises(self) -> None:
        bad = _material(
            elastic_modulus={20.0: 200000.0}, poisson_ratio={20.0: 1.0}
        )
        with pytest.raises(ValueError, match="plane-stress"):
            bad.elasticity_tensor_2d(20.0, plane="stress")

    def test_plane_strain_entries(self, mat) -> None:
        """Plane-strain matrix carries the 3D Lame constants directly."""
        C = mat.elasticity_tensor_2d(20.0, plane="strain")
        lam = mat.lame_lambda(20.0)
        mu = mat.lame_mu(20.0)
        assert C[0, 0] == pytest.approx(lam + 2 * mu)
        assert C[0, 1] == pytest.approx(lam)
        assert C[2, 2] == pytest.approx(mu)

    def test_plane_strain_nu_ge_half_raises(self) -> None:
        bad = _material(
            elastic_modulus={20.0: 200000.0}, poisson_ratio={20.0: 0.5}
        )
        with pytest.raises(ValueError, match="plane-strain"):
            bad.elasticity_tensor_2d(20.0, plane="strain")

    def test_3d_tensor_symmetric(self, mat) -> None:
        C = mat.elasticity_tensor_3d(20.0)
        assert C.shape == (6, 6)
        assert np.allclose(C, C.T)

    def test_3d_tensor_voigt_entries(self, mat) -> None:
        C = mat.elasticity_tensor_3d(20.0)
        lam = mat.lame_lambda(20.0)
        mu = mat.lame_mu(20.0)
        assert np.allclose(np.diag(C)[:3], lam + 2 * mu)
        assert C[0, 1] == pytest.approx(lam)
        assert C[2, 0] == pytest.approx(lam)
        assert np.allclose(np.diag(C)[3:], mu)
        assert np.allclose(C[:3, 3:], 0.0)


# ---------------------------------------------------------------------------
# MaterialSet
# ---------------------------------------------------------------------------


class TestMaterialSet:
    @pytest.fixture
    def mat_set(self, steel_material) -> MaterialSet:
        weld = Material(name="weld", density=7850.0)
        haz = Material(name="haz", density=7850.0)
        return MaterialSet(base_metal=steel_material, weld_metal=weld, haz=haz)

    def test_region_aliases(self, mat_set) -> None:
        assert mat_set.for_region("base") is mat_set.base_metal
        assert mat_set.for_region("base_metal") is mat_set.base_metal
        assert mat_set.for_region("weld") is mat_set.weld_metal
        assert mat_set.for_region("weld_metal") is mat_set.weld_metal
        assert mat_set.for_region("haz") is mat_set.haz
        assert mat_set.for_region("heat_affected_zone") is mat_set.haz

    def test_region_lookup_is_case_insensitive(self, mat_set) -> None:
        assert mat_set.for_region("BASE") is mat_set.base_metal
        assert mat_set.for_region("Weld_Metal") is mat_set.weld_metal

    def test_unknown_region_raises_key_error(self, mat_set) -> None:
        with pytest.raises(KeyError):
            mat_set.for_region("flange")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadMaterial:
    def test_list_of_pairs_format(self, tmp_path) -> None:
        (tmp_path / "pairs.yaml").write_text(
            "name: Pairs\n"
            "density: 7000\n"
            "elastic_modulus:\n"
            "  - [20, 200000]\n"
            "  - [500, 160000]\n"
        )
        mat = load_material("pairs", data_dir=tmp_path)
        assert mat.name == "Pairs"
        assert mat.density == pytest.approx(7000.0)
        assert mat.elastic_modulus == {20.0: 200000.0, 500.0: 160000.0}
        assert mat.E(260.0) == pytest.approx(180000.0)

    def test_mapping_format(self, tmp_path) -> None:
        (tmp_path / "mapping.yaml").write_text(
            "elastic_modulus:\n"
            "  20: 200000\n"
            "  500: 160000\n"
        )
        mat = load_material("mapping", data_dir=tmp_path)
        assert mat.elastic_modulus == {20.0: 200000.0, 500.0: 160000.0}

    def test_scalar_format_becomes_constant(self, tmp_path) -> None:
        """A bare scalar is duplicated at 20 C and 500 C."""
        (tmp_path / "scalar.yaml").write_text("poisson_ratio: 0.3\n")
        mat = load_material("scalar", data_dir=tmp_path)
        assert mat.poisson_ratio == {20.0: 0.3, 500.0: 0.3}
        assert mat.nu(1000.0) == pytest.approx(0.3)

    def test_defaults_when_fields_absent(self, tmp_path) -> None:
        (tmp_path / "bare.yaml").write_text("elastic_modulus: 1.0\n")
        mat = load_material("bare", data_dir=tmp_path)
        assert mat.name == "bare"
        assert mat.density == pytest.approx(7850.0)
        assert mat.category is None
        assert mat.creep_A == 0.0
        assert mat.creep_n == 1.0
        assert mat.hardening_modulus == 0.0
        assert mat.hardening_exponent == 1.0

    def test_creep_and_hardening_passthrough(self, tmp_path) -> None:
        (tmp_path / "creep.yaml").write_text(
            "creep_A: 1.0e-20\n"
            "creep_n: 5.0\n"
            "creep_m: 0.1\n"
            "hardening_modulus: 1500\n"
            "hardening_exponent: 0.5\n"
        )
        mat = load_material("creep", data_dir=tmp_path)
        assert mat.creep_A == pytest.approx(1e-20)
        assert mat.creep_n == pytest.approx(5.0)
        assert mat.creep_m == pytest.approx(0.1)
        assert mat.hardening_modulus == pytest.approx(1500.0)
        assert mat.hardening_exponent == pytest.approx(0.5)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="no_such_material"):
            load_material("no_such_material", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Bundled data registry
# ---------------------------------------------------------------------------


class TestBundledMaterials:
    def test_a36_smoke(self) -> None:
        mat = load_material("A36")
        assert mat.name == "ASTM A36"
        assert mat.category == "carbon_steel"
        assert mat.density == pytest.approx(7850.0)
        assert mat.E(20.0) == pytest.approx(200000.0)
        assert mat.sigma_y(20.0) == pytest.approx(250.0)
        assert mat.creep_n == pytest.approx(5.0)

    def test_list_available_materials(self) -> None:
        names = list_available_materials()
        assert len(names) >= 40
        assert "A36" in names
        assert names == sorted(names)

    def test_search_is_case_insensitive(self) -> None:
        assert search_materials("s355") == search_materials("S355")
        assert "S355" in search_materials("s355")

    def test_search_matches_inner_name_field(self) -> None:
        """'ASTM A36' only appears inside the YAML, not in the file stem."""
        assert "A36" in search_materials("astm a36")

    def test_list_material_categories(self) -> None:
        cats = list_material_categories()
        assert "carbon_steel" in cats
        assert "A36" in cats["carbon_steel"]
        assert sum(len(v) for v in cats.values()) == len(list_available_materials())
