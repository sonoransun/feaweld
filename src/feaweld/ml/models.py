"""ML-based fatigue life prediction models.

Supports Random Forest, XGBoost, and ensemble predictors trained on
physics-informed features extracted from FEA results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from feaweld.ml.features import FatigueFeatures, standard_feature_names


# ---------------------------------------------------------------------------
# Configuration and result data classes
# ---------------------------------------------------------------------------


@dataclass
class MLModelConfig:
    """Configuration for a fatigue life ML model."""

    model_type: Literal["random_forest", "xgboost", "ensemble"] = "random_forest"
    n_estimators: int = 500
    max_depth: int = 8
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class MLPrediction:
    """A single fatigue life prediction with uncertainty."""

    predicted_life: float            # 10^(log_predicted_life)
    log_predicted_life: float        # log10(N)
    confidence_interval: tuple[float, float]  # (lower, upper) on N
    feature_importances: dict[str, float]


# ---------------------------------------------------------------------------
# Fatigue predictor
# ---------------------------------------------------------------------------


class FatiguePredictor:
    """Train and apply ML models for fatigue life prediction.

    Parameters
    ----------
    config : MLModelConfig
        Model hyper-parameters and training settings.
    """

    def __init__(self, config: MLModelConfig | None = None) -> None:
        self.config = config or MLModelConfig()
        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------ train

    def train(self, features: FatigueFeatures) -> dict[str, float | list[float]]:
        """Train the model and return cross-validation metrics.

        Parameters
        ----------
        features : FatigueFeatures
            Training data with non-None ``target``.

        Returns
        -------
        dict
            ``{"rmse": float, "r2": float, "cv_scores": list[float]}``
        """

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, r2_score

        if features.target is None:
            raise ValueError("Cannot train: features.target is None")

        X = features.values.copy()
        y = features.target.copy()

        # Replace NaN with column median
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if not np.isnan(median_val) else 0.0

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._feature_names = list(features.feature_names)

        model = self._build_model()

        # Cross-validation
        n_folds = min(self.config.cv_folds, len(y))
        if n_folds < 2:
            n_folds = 2
        cv_scores = cross_val_score(
            model, X_scaled, y, cv=n_folds, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)

        # Full fit
        model.fit(X_scaled, y)
        self._model = model
        self._is_trained = True

        y_pred = model.predict(X_scaled)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2 = float(r2_score(y, y_pred))

        return {
            "rmse": rmse,
            "r2": r2,
            "cv_scores": cv_rmse.tolist(),
        }

    # --------------------------------------------------------------- predict

    def predict(
        self,
        features: dict[str, float] | NDArray[np.float64],
    ) -> MLPrediction:
        """Predict fatigue life for one sample.

        Parameters
        ----------
        features : dict or NDArray
            Either a feature dict (keys matching feature names) or a 1-D array
            of length ``n_features``.

        Returns
        -------
        MLPrediction
        """

        if not self._is_trained or self._model is None or self._scaler is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        x = self._prepare_single(features)
        x_scaled = self._scaler.transform(x.reshape(1, -1))

        log_pred = float(self._model.predict(x_scaled)[0])

        # Confidence interval from tree variance
        lower_log, upper_log = self._compute_ci(x_scaled)
        lower_life = 10.0 ** lower_log
        upper_life = 10.0 ** upper_log

        return MLPrediction(
            predicted_life=10.0 ** log_pred,
            log_predicted_life=log_pred,
            confidence_interval=(lower_life, upper_life),
            feature_importances=self.feature_importance(),
        )

    # -------------------------------------------------- feature_importance

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances from the trained model."""

        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained.")

        importances = self._get_raw_importances()
        return {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

    # ----------------------------------------------------------------- save

    def save(self, path: str) -> None:
        """Persist model, scaler, and metadata to disk with joblib."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")

        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "config": self.config,
            "is_trained": self._is_trained,
        }
        joblib.dump(payload, path)

    # ----------------------------------------------------------------- load

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")

        payload = joblib.load(path)
        self._model = payload["model"]
        self._scaler = payload["scaler"]
        self._feature_names = payload["feature_names"]
        self.config = payload["config"]
        self._is_trained = payload["is_trained"]

    # ================================================================ private

    def _build_model(self):
        """Instantiate the underlying estimator(s)."""
        cfg = self.config

        if cfg.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=cfg.random_state,
            )
        elif cfg.model_type == "xgboost":
            return self._build_xgboost()
        elif cfg.model_type == "ensemble":
            return self._build_ensemble()
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

    def _build_xgboost(self):
        """Build an XGBoost regressor, raising ImportError if unavailable."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. "
                "Install with: pip install xgboost"
            )

        cfg = self.config
        return XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            verbosity=0,
        )

    def _build_ensemble(self):
        """Build a voting ensemble of RF + XGBoost."""
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor

        cfg = self.config
        rf = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
        )

        estimators = [("rf", rf)]

        try:
            xgb = self._build_xgboost()
            estimators.append(("xgb", xgb))
        except ImportError:
            # Fall back to RF-only if XGBoost is missing
            pass

        if len(estimators) == 1:
            return rf

        return VotingRegressor(estimators=estimators)

    def _prepare_single(self, features: dict[str, float] | NDArray) -> NDArray:
        """Convert a dict or array to a properly ordered 1-D feature vector."""
        if isinstance(features, dict):
            x = np.full(len(self._feature_names), np.nan, dtype=np.float64)
            for i, name in enumerate(self._feature_names):
                if name in features:
                    x[i] = features[name]
            # Fill remaining NaN with 0 (will be scaled)
            x = np.nan_to_num(x, nan=0.0)
            return x
        else:
            arr = np.asarray(features, dtype=np.float64).ravel()
            return np.nan_to_num(arr, nan=0.0)

    def _compute_ci(self, x_scaled: NDArray, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval bounds on log10(N)."""
        from scipy import stats as sp_stats

        model = self._model
        log_pred = float(model.predict(x_scaled)[0])

        # For tree-based models, use individual tree predictions
        tree_preds = self._get_tree_predictions(x_scaled)

        if tree_preds is not None and len(tree_preds) > 1:
            std = float(np.std(tree_preds, ddof=1))
            z = sp_stats.norm.ppf(1 - alpha / 2)
            return log_pred - z * std, log_pred + z * std
        else:
            # Fallback: +/- 10% of predicted log life
            margin = abs(log_pred) * 0.10 + 0.1
            return log_pred - margin, log_pred + margin

    def _get_tree_predictions(self, x_scaled: NDArray) -> NDArray | None:
        """Get per-tree predictions for variance estimation."""
        model = self._model

        # VotingRegressor wrapping RF (+ optional XGB): check first because
        # VotingRegressor also has estimators_ but they are sub-models, not trees.
        try:
            from sklearn.ensemble import VotingRegressor
            if isinstance(model, VotingRegressor):
                all_preds = []
                for est in model.estimators_:
                    if hasattr(est, "estimators_"):
                        for t in est.estimators_:
                            all_preds.append(t.predict(x_scaled)[0])
                if all_preds:
                    return np.array(all_preds)
                return None
        except ImportError:
            pass

        # RandomForestRegressor (or any estimator with tree estimators_)
        if hasattr(model, "estimators_"):
            preds = np.array([t.predict(x_scaled)[0] for t in model.estimators_])
            return preds

        return None

    def _get_raw_importances(self) -> NDArray:
        """Extract raw feature importance array from the underlying model."""
        model = self._model

        # VotingRegressor: average importances from sub-models (check first
        # because VotingRegressor does NOT have feature_importances_ directly)
        try:
            from sklearn.ensemble import VotingRegressor
            if isinstance(model, VotingRegressor):
                imps = []
                for est in model.estimators_:
                    if hasattr(est, "feature_importances_"):
                        imps.append(np.asarray(est.feature_importances_))
                if imps:
                    return np.mean(imps, axis=0)
                return np.zeros(len(self._feature_names))
        except ImportError:
            pass

        # Direct feature_importances_ (RF, XGBoost)
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_)

        # Fallback
        return np.zeros(len(self._feature_names))
