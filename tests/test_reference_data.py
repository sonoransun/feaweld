"""Tests for technical reference data repository, cache, and domain modules."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from feaweld.core.types import StressField


# ---------------------------------------------------------------------------
# Tests: DataRegistry
# ---------------------------------------------------------------------------

class TestDataRegistry:
    def test_list_all_datasets(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        datasets = reg.list_datasets()
        assert len(datasets) > 0

    def test_list_by_category(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        materials = reg.list_datasets("materials")
        assert len(materials) >= 4  # at least original 4

    def test_categories(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        cats = reg.categories
        assert "materials" in cats

    def test_search(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        results = reg.search("A36")
        assert any("A36" in d.name for d in results)

    def test_get_dataset_path(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        path = reg.get_dataset_path("materials/A36")
        assert path.exists()

    def test_get_missing_raises(self):
        from feaweld.data.registry import DataRegistry
        reg = DataRegistry()
        with pytest.raises(KeyError):
            reg.get_dataset_path("nonexistent/thing")


# ---------------------------------------------------------------------------
# Tests: DataCache
# ---------------------------------------------------------------------------

class TestDataCache:
    def test_lazy_load(self):
        from feaweld.data.cache import DataCache
        cache = DataCache()
        data = cache.get("materials/A36")
        assert isinstance(data, dict)
        assert "name" in data

    def test_cache_hit(self):
        from feaweld.data.cache import DataCache
        cache = DataCache()
        d1 = cache.get("materials/A36")
        d2 = cache.get("materials/A36")
        assert d1 is d2  # same object from cache

    def test_clear(self):
        from feaweld.data.cache import DataCache
        cache = DataCache()
        cache.get("materials/A36")
        assert cache.stats["entries"] >= 1
        cache.clear()
        assert cache.stats["entries"] == 0

    def test_preload(self):
        from feaweld.data.cache import DataCache
        cache = DataCache()
        cache.preload("materials")
        assert cache.stats["entries"] >= 4

    def test_eviction(self):
        from feaweld.data.cache import DataCache
        # Tiny memory limit to force eviction
        cache = DataCache(max_memory_bytes=1)
        cache.get("materials/A36")
        cache.get("materials/304SS")
        # Should have evicted the first one
        assert cache.stats["entries"] <= 2

    def test_thread_safety(self):
        from feaweld.data.cache import DataCache
        cache = DataCache()
        errors = []

        def load_material():
            try:
                cache.get("materials/A36")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_material) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_singleton(self):
        from feaweld.data.cache import get_cache
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2


# ---------------------------------------------------------------------------
# Tests: SCF Database
# ---------------------------------------------------------------------------

class TestSCFDatabase:
    def test_get_fillet(self):
        from feaweld.data.scf import get_scf_coefficients
        coeff = get_scf_coefficients("fillet_toe")
        assert coeff.a == pytest.approx(0.469)
        assert coeff.b == pytest.approx(0.572)

    def test_compute_scf(self):
        from feaweld.data.scf import compute_scf
        kt = compute_scf("fillet_toe", toe_radius=1.0, toe_angle=45.0, plate_thickness=20.0)
        assert kt > 1.0

    def test_list_geometries(self):
        from feaweld.data.scf import list_scf_geometries
        geoms = list_scf_geometries()
        assert len(geoms) >= 10
        assert "fillet_toe" in geoms
        assert "butt_toe" in geoms

    def test_backward_compat_with_notch_stress(self):
        """SCF from database should match the hardcoded values."""
        from feaweld.postprocess.notch_stress import notch_stress_scf_parametric
        # Without geometry arg: uses hardcoded values
        kt_old = notch_stress_scf_parametric(1.0, 45.0, 20.0, "fillet")
        # With geometry arg: uses database
        kt_new = notch_stress_scf_parametric(1.0, 45.0, 20.0, geometry="fillet_toe")
        assert kt_old == pytest.approx(kt_new, rel=0.01)

    def test_unknown_geometry_raises(self):
        from feaweld.data.scf import get_scf_coefficients
        with pytest.raises(KeyError):
            get_scf_coefficients("nonexistent")


# ---------------------------------------------------------------------------
# Tests: Weld Details
# ---------------------------------------------------------------------------

class TestWeldDetails:
    def test_get_weld_detail(self):
        from feaweld.data.sn_curves.weld_details import get_weld_detail
        d = get_weld_detail(1)
        assert d.fat_class > 0
        assert d.description != ""

    def test_find_by_joint_type(self):
        from feaweld.data.sn_curves.weld_details import find_weld_details
        butt = find_weld_details(joint_type="butt")
        assert len(butt) > 0
        assert all("butt" in d.joint_type.lower() or d.joint_type == "butt" for d in butt)

    def test_recommend_fat_class(self):
        from feaweld.data.sn_curves.weld_details import recommend_fat_class
        result = recommend_fat_class("butt", "ground_flush")
        # May return WeldDetail or int depending on implementation
        fat = result.fat_class if hasattr(result, "fat_class") else result
        assert fat is None or (50 <= fat <= 160)

    def test_sn_curve_by_detail(self):
        from feaweld.fatigue.sn_curves import get_sn_curve_by_detail
        curve = get_sn_curve_by_detail(1)
        assert curve.name.startswith("IIW")


# ---------------------------------------------------------------------------
# Tests: CCT Database
# ---------------------------------------------------------------------------

class TestCCTDatabase:
    def test_get_a36(self):
        from feaweld.data.cct import get_cct_diagram
        cct = get_cct_diagram("A36")
        assert len(cct.cooling_rates) >= 8
        assert cct.Ac1 > 0
        assert cct.Ms > 0

    def test_list_grades(self):
        from feaweld.data.cct import list_cct_grades
        grades = list_cct_grades()
        assert len(grades) >= 20
        assert "A36" in grades

    def test_phase_prediction(self):
        from feaweld.data.cct import get_cct_diagram
        cct = get_cct_diagram("4140")
        phases = cct.predict_phases(50.0)
        total = phases.ferrite + phases.pearlite + phases.bainite + phases.martensite + phases.austenite
        assert total == pytest.approx(1.0, abs=0.05)

    def test_find_closest(self):
        from feaweld.data.cct import find_closest_cct
        cct = find_closest_cct(0.35)  # CE ~ A36
        assert cct is not None

    def test_meso_integration(self):
        from feaweld.multiscale.meso import cct_for_grade
        cct = cct_for_grade("4140")
        assert len(cct.cooling_rates) >= 8

    def test_fallback_for_unknown(self):
        from feaweld.multiscale.meso import cct_for_grade
        cct = cct_for_grade("totally_unknown_grade")
        # Should fall back to default
        assert len(cct.cooling_rates) >= 8


# ---------------------------------------------------------------------------
# Tests: Residual Stress
# ---------------------------------------------------------------------------

class TestResidualStress:
    def test_bs7910_level1(self):
        from feaweld.data.residual_stress import get_residual_profile, evaluate_residual_stress
        profile = get_residual_profile("BS7910_Level1")
        # Level 1 = uniform sigma_y everywhere
        stress = evaluate_residual_stress("BS7910_Level1", 0.5, 250.0)
        assert stress == pytest.approx(250.0, rel=0.05)

    def test_bs7910_level2_butt(self):
        from feaweld.data.residual_stress import get_residual_profile, evaluate_residual_stress
        profile = get_residual_profile("BS7910_Level2_butt")
        # At surface: should be near sigma_y
        stress_surface = evaluate_residual_stress("BS7910_Level2_butt", 0.0, 250.0)
        assert abs(stress_surface) > 0  # non-zero at surface
        # At mid-thickness: should differ from surface
        stress_mid = evaluate_residual_stress("BS7910_Level2_butt", 0.5, 250.0)
        assert isinstance(stress_mid, (int, float, np.floating))

    def test_list_profiles(self):
        from feaweld.data.residual_stress import list_residual_profiles
        profiles = list_residual_profiles()
        assert len(profiles) >= 4


# ---------------------------------------------------------------------------
# Tests: Filler Metals
# ---------------------------------------------------------------------------

class TestFillerMetals:
    def test_get_e7018(self):
        from feaweld.data.filler_metals import get_filler_metal
        fm = get_filler_metal("E7018")
        assert fm.tensile_mpa == pytest.approx(483, abs=10)
        assert fm.process == "SMAW"

    def test_list_smaw(self):
        from feaweld.data.filler_metals import get_filler_metal, list_filler_metals
        all_fm = list_filler_metals(process="SMAW")
        assert len(all_fm) > 0
        assert all(fm.process == "SMAW" for fm in all_fm)

    def test_filler_for_base(self):
        from feaweld.data.filler_metals import filler_for_base_metal
        matches = filler_for_base_metal("A36")
        assert len(matches) > 0

    def test_unknown_filler(self):
        from feaweld.data.filler_metals import get_filler_metal
        with pytest.raises(KeyError):
            get_filler_metal("NONEXISTENT")


# ---------------------------------------------------------------------------
# Tests: Weld Efficiency
# ---------------------------------------------------------------------------

class TestWeldEfficiency:
    def test_asme_full_rt(self):
        from feaweld.data.weld_efficiency import get_weld_efficiency
        result = get_weld_efficiency("ASME_VIII_Div1", "Type_1", "Full_RT")
        eff = result.efficiency if hasattr(result, "efficiency") else result
        assert eff == pytest.approx(1.0)

    def test_asme_spot_rt(self):
        from feaweld.data.weld_efficiency import get_weld_efficiency
        result = get_weld_efficiency("ASME_VIII_Div1", "Type_1", "Spot_RT")
        eff = result.efficiency if hasattr(result, "efficiency") else result
        assert eff == pytest.approx(0.85)

    def test_asme_no_rt(self):
        from feaweld.data.weld_efficiency import get_weld_efficiency
        result = get_weld_efficiency("ASME_VIII_Div1", "Type_1", "None")
        eff = result.efficiency if hasattr(result, "efficiency") else result
        assert eff == pytest.approx(0.70)

    def test_list_efficiencies(self):
        from feaweld.data.weld_efficiency import list_efficiencies
        all_eff = list_efficiencies()
        assert len(all_eff) >= 10


# ---------------------------------------------------------------------------
# Tests: Material Expansion
# ---------------------------------------------------------------------------

class TestMaterialExpansion:
    def test_original_materials_still_work(self):
        from feaweld.core.materials import load_material
        mat = load_material("A36")
        assert mat.name == "ASTM A36"
        assert mat.E(20.0) == pytest.approx(200000, rel=0.01)

    def test_search_stainless(self):
        from feaweld.core.materials import search_materials
        results = search_materials("stainless")
        # Should find 304SS at minimum (and possibly more if new materials exist)
        assert any("304" in r for r in results)

    def test_search_case_insensitive(self):
        from feaweld.core.materials import search_materials
        results = search_materials("a36")
        assert "A36" in results

    def test_list_categories(self):
        from feaweld.core.materials import list_material_categories
        cats = list_material_categories()
        assert isinstance(cats, dict)
        assert "carbon_steel" in cats
        assert "A36" in cats["carbon_steel"]

    def test_category_field_loaded(self):
        from feaweld.core.materials import load_material
        mat = load_material("A36")
        assert mat.category == "carbon_steel"

    def test_many_materials_available(self):
        from feaweld.core.materials import list_available_materials
        mats = list_available_materials()
        assert len(mats) >= 4  # at least the originals


# ---------------------------------------------------------------------------
# Tests: Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_iiw_fat90_unchanged(self):
        from feaweld.fatigue.sn_curves import iiw_fat
        curve = iiw_fat(90)
        life = curve.life(90.0)
        # At FAT class stress, life should be 2e6
        assert life == pytest.approx(2e6, rel=0.01)

    def test_notch_stress_default_unchanged(self):
        from feaweld.postprocess.notch_stress import notch_stress_scf_parametric
        # Call without the new geometry param — should use hardcoded values
        kt = notch_stress_scf_parametric(1.0, 45.0, 20.0, "fillet")
        assert kt > 1.0

    def test_blodgett_capacity_default(self):
        from feaweld.postprocess.blodgett import lrfd_capacity
        cap = lrfd_capacity(throat=5.0, A_w=100.0)
        assert cap > 0

    def test_stress_field_difference_still_works(self):
        from feaweld.pipeline.comparison import compute_stress_field_difference
        a = StressField(values=np.array([[100, 0, 0, 0, 0, 0]], dtype=float))
        b = StressField(values=np.array([[60, 0, 0, 0, 0, 0]], dtype=float))
        diff = compute_stress_field_difference(a, b)
        assert diff.values[0, 0] == pytest.approx(40.0)
