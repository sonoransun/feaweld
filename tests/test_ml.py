"""Tests for the ML fatigue prediction modules."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    StressField,
    WeldLineDefinition,
)
from feaweld.ml.features import (
    FatigueFeatures,
    build_feature_matrix,
    extract_features,
    standard_feature_names,
)
from feaweld.ml.models import FatiguePredictor, MLModelConfig, MLPrediction
from feaweld.ml.transfer import TransferLearner


# ---------------------------------------------------------------------------
# Helpers: synthetic data generation
# ---------------------------------------------------------------------------


def _make_mock_fea_results() -> FEAResults:
    """Create a minimal mock FEAResults for feature extraction."""
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 3], [1, 2, 4]], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TRI3)

    # Stress field: 5 nodes x 6 components
    stress_vals = np.array(
        [
            [100, 20, 10, 5, 3, 2],
            [150, 30, 15, 8, 4, 3],
            [120, 25, 12, 6, 3, 2],
            [80, 15, 8, 4, 2, 1],
            [200, 40, 20, 10, 5, 4],
        ],
        dtype=np.float64,
    )
    stress = StressField(values=stress_vals, location="nodes")

    return FEAResults(
        mesh=mesh,
        stress=stress,
        metadata={
            "structural_stress_membrane": 130.0,
            "structural_stress_bending": 25.0,
            "weld_toe_angle": 45.0,
            "toe_radius": 1.0,
            "misalignment": 0.2,
        },
    )


def _make_weld_line() -> WeldLineDefinition:
    return WeldLineDefinition(
        name="test_weld",
        node_ids=np.array([0, 1, 2], dtype=np.int64),
        plate_thickness=10.0,
        normal_direction=np.array([0, 0, 1], dtype=np.float64),
    )


def _make_synthetic_features(
    n_samples: int = 100,
    seed: int = 42,
) -> FatigueFeatures:
    """Create a synthetic dataset where target = linear(features) + noise."""
    rng = np.random.default_rng(seed)
    names = standard_feature_names()
    n_feat = len(names)

    X = rng.standard_normal((n_samples, n_feat))
    # Target is a linear function of the first 3 features + noise
    coefs = np.zeros(n_feat)
    coefs[0] = 2.0  # stress_range
    coefs[1] = -0.5  # r_ratio
    coefs[2] = 0.3  # plate_thickness
    y = X @ coefs + 5.0 + rng.normal(0, 0.3, size=n_samples)

    return FatigueFeatures(feature_names=names, values=X, target=y)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Verify feature engineering from FEA results."""

    def test_extract_features_returns_dict(self) -> None:
        results = _make_mock_fea_results()
        weld = _make_weld_line()
        feats = extract_features(results, weld, material_props={"uts": 490, "yield": 355})

        assert isinstance(feats, dict)
        assert "stress_range" in feats
        assert "plate_thickness" in feats
        assert feats["plate_thickness"] == 10.0

    def test_extract_features_stress_values(self) -> None:
        results = _make_mock_fea_results()
        weld = _make_weld_line()
        feats = extract_features(results, weld)

        assert feats["stress_range"] >= 0
        assert feats["scf"] >= 1.0  # SCF >= 1 by definition

    def test_extract_features_material_props(self) -> None:
        results = _make_mock_fea_results()
        weld = _make_weld_line()
        feats = extract_features(
            results, weld,
            material_props={"uts": 490, "yield": 355, "residual_stress": 100},
        )
        assert feats["material_uts"] == 490
        assert feats["material_yield"] == 355
        assert "residual_stress_ratio" in feats
        assert feats["residual_stress_ratio"] == pytest.approx(100 / 355)

    def test_extract_features_metadata(self) -> None:
        results = _make_mock_fea_results()
        weld = _make_weld_line()
        feats = extract_features(results, weld)
        assert feats["weld_toe_angle"] == 45.0
        assert feats["toe_radius"] == 1.0
        assert feats["misalignment"] == 0.2

    def test_standard_feature_names(self) -> None:
        names = standard_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "stress_range" in names


class TestBuildFeatureMatrix:
    """Verify combining feature dicts into a matrix."""

    def test_single_dict(self) -> None:
        fd = {"stress_range": 100.0, "plate_thickness": 10.0}
        ff = build_feature_matrix([fd])
        assert ff.values.shape[0] == 1
        assert ff.target is None

    def test_with_targets(self) -> None:
        fd1 = {"stress_range": 100.0}
        fd2 = {"stress_range": 200.0}
        ff = build_feature_matrix([fd1, fd2], target_lives=[1e6, 1e5])
        assert ff.target is not None
        assert ff.target[0] == pytest.approx(6.0)  # log10(1e6) = 6
        assert ff.target[1] == pytest.approx(5.0)

    def test_missing_features_are_nan(self) -> None:
        fd = {"stress_range": 50.0}
        ff = build_feature_matrix([fd])
        names = ff.feature_names
        idx_r_ratio = names.index("r_ratio")
        assert np.isnan(ff.values[0, idx_r_ratio])

    def test_multiple_dicts_aligned(self) -> None:
        fds = [
            {"stress_range": 100, "r_ratio": 0.1},
            {"stress_range": 200, "plate_thickness": 12},
        ]
        ff = build_feature_matrix(fds)
        assert ff.values.shape == (2, len(standard_feature_names()))


# ---------------------------------------------------------------------------
# FatiguePredictor
# ---------------------------------------------------------------------------


class TestFatiguePredictor:
    """Test ML model training and prediction."""

    def test_train_returns_metrics(self) -> None:
        features = _make_synthetic_features(n_samples=80)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=20, max_depth=4, cv_folds=3
        )
        predictor = FatiguePredictor(config)
        metrics = predictor.train(features)

        assert "rmse" in metrics
        assert "r2" in metrics
        assert "cv_scores" in metrics
        assert metrics["r2"] > 0  # should fit reasonably on training data

    def test_predict_returns_valid_mlprediction(self) -> None:
        features = _make_synthetic_features(n_samples=80)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=20, max_depth=4, cv_folds=3
        )
        predictor = FatiguePredictor(config)
        predictor.train(features)

        sample_feats = {name: 0.5 for name in standard_feature_names()}
        pred = predictor.predict(sample_feats)

        assert isinstance(pred, MLPrediction)
        assert pred.predicted_life > 0
        assert pred.confidence_interval[0] < pred.confidence_interval[1]
        assert len(pred.feature_importances) == len(standard_feature_names())

    def test_predict_with_array(self) -> None:
        features = _make_synthetic_features(n_samples=80)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=20, max_depth=4, cv_folds=3
        )
        predictor = FatiguePredictor(config)
        predictor.train(features)

        x = np.zeros(len(standard_feature_names()))
        pred = predictor.predict(x)
        assert isinstance(pred, MLPrediction)

    def test_feature_importance(self) -> None:
        features = _make_synthetic_features(n_samples=80)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=20, max_depth=4, cv_folds=3
        )
        predictor = FatiguePredictor(config)
        predictor.train(features)

        imp = predictor.feature_importance()
        assert isinstance(imp, dict)
        assert "stress_range" in imp
        # stress_range should be among the most important (coef=2.0 in synthetic data)
        assert imp["stress_range"] > 0

    def test_predict_before_train_raises(self) -> None:
        predictor = FatiguePredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict({"stress_range": 100})

    def test_train_without_target_raises(self) -> None:
        ff = FatigueFeatures(
            feature_names=standard_feature_names(),
            values=np.zeros((10, len(standard_feature_names()))),
            target=None,
        )
        predictor = FatiguePredictor()
        with pytest.raises(ValueError, match="target is None"):
            predictor.train(ff)

    def test_save_load_roundtrip(self) -> None:
        features = _make_synthetic_features(n_samples=80)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=10, max_depth=3, cv_folds=2
        )
        predictor = FatiguePredictor(config)
        predictor.train(features)

        sample_feats = {name: 0.5 for name in standard_feature_names()}
        pred_before = predictor.predict(sample_feats)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        try:
            predictor.save(path)

            loaded = FatiguePredictor()
            loaded.load(path)
            pred_after = loaded.predict(sample_feats)

            assert pred_before.log_predicted_life == pytest.approx(
                pred_after.log_predicted_life
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TransferLearner
# ---------------------------------------------------------------------------


class TestTransferLearner:
    """Test domain adaptation / transfer learning."""

    def _trained_base(self) -> FatiguePredictor:
        features = _make_synthetic_features(n_samples=100, seed=10)
        config = MLModelConfig(
            model_type="random_forest", n_estimators=20, max_depth=4, cv_folds=3
        )
        predictor = FatiguePredictor(config)
        predictor.train(features)
        return predictor

    def test_fine_tune_returns_metrics(self) -> None:
        base = self._trained_base()
        tl = TransferLearner(base)

        # New domain data with a shifted target
        new_feats = _make_synthetic_features(n_samples=50, seed=99)
        new_feats.target = new_feats.target + 1.0  # shift
        metrics = tl.fine_tune(new_feats, n_correction_trees=10)

        assert "rmse_base" in metrics
        assert "rmse_corrected" in metrics
        assert "r2_corrected" in metrics
        # Correction should reduce error
        assert metrics["rmse_corrected"] <= metrics["rmse_base"] + 0.01

    def test_transfer_predict(self) -> None:
        base = self._trained_base()
        tl = TransferLearner(base)

        new_feats = _make_synthetic_features(n_samples=50, seed=99)
        new_feats.target = new_feats.target + 1.0
        tl.fine_tune(new_feats, n_correction_trees=10)

        sample = {name: 0.5 for name in standard_feature_names()}
        pred = tl.predict(sample)
        assert isinstance(pred, MLPrediction)
        assert pred.predicted_life > 0

    def test_predict_before_tune_raises(self) -> None:
        base = self._trained_base()
        tl = TransferLearner(base)
        with pytest.raises(RuntimeError, match="not fine-tuned"):
            tl.predict({"stress_range": 100})

    def test_untrained_base_raises(self) -> None:
        untrained = FatiguePredictor()
        with pytest.raises(RuntimeError, match="must be trained"):
            TransferLearner(untrained)

    def test_domain_adaptation_score(self) -> None:
        base = self._trained_base()
        tl = TransferLearner(base)

        source = _make_synthetic_features(n_samples=50, seed=1)
        target = _make_synthetic_features(n_samples=50, seed=2)

        score = tl.domain_adaptation_score(source, target)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_domain_adaptation_same_data_low_score(self) -> None:
        """MMD between identical distributions should be near zero."""
        base = self._trained_base()
        tl = TransferLearner(base)

        data = _make_synthetic_features(n_samples=80, seed=42)
        score = tl.domain_adaptation_score(data, data)
        # Should be very small (ideally 0, but numerical noise exists)
        assert score < 0.05
