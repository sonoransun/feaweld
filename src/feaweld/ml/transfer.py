"""Transfer learning across weld joint types.

Enables a base fatigue predictor trained on one joint type (e.g. fillet welds)
to be adapted to a new joint type (e.g. butt welds) with limited data by
learning a residual correction model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from feaweld.ml.features import FatigueFeatures, standard_feature_names
from feaweld.ml.models import FatiguePredictor, MLModelConfig, MLPrediction


class TransferLearner:
    """Fine-tune a base fatigue predictor for a new domain.

    Parameters
    ----------
    base_predictor : FatiguePredictor
        A pre-trained predictor (source domain).
    """

    def __init__(self, base_predictor: FatiguePredictor) -> None:
        if not base_predictor._is_trained:
            raise RuntimeError("base_predictor must be trained before transfer.")
        self.base = base_predictor
        self._correction_model = None
        self._correction_scaler = None
        self._is_tuned = False

    def fine_tune(
        self,
        new_features: FatigueFeatures,
        n_correction_trees: int = 50,
    ) -> dict[str, float]:
        """Train a correction model on residuals.

        Steps
        -----
        1. Obtain base-model predictions on the new (target) data.
        2. Compute residuals = target - base_prediction.
        3. Train a lightweight Random Forest on the residuals.
        4. Combined prediction = base + correction.

        Parameters
        ----------
        new_features : FatigueFeatures
            Target-domain data with ``target`` set.
        n_correction_trees : int
            Number of trees in the correction RF.

        Returns
        -------
        dict
            Training metrics: ``{"rmse_base", "rmse_corrected", "r2_corrected"}``.
        """

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score

        if new_features.target is None:
            raise ValueError("new_features.target is required for fine-tuning.")

        X = new_features.values.copy()
        y = new_features.target.copy()

        # Impute NaN with column median
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                median_val = np.nanmedian(X[:, col])
                X[mask, col] = median_val if not np.isnan(median_val) else 0.0

        # Step 1: base predictions on new data
        base_preds = np.empty(len(y))
        for i in range(len(y)):
            x_row = X[i]
            x_scaled = self.base._scaler.transform(x_row.reshape(1, -1))
            base_preds[i] = float(self.base._model.predict(x_scaled)[0])

        rmse_base = float(np.sqrt(mean_squared_error(y, base_preds)))

        # Step 2: residuals
        residuals = y - base_preds

        # Step 3: train correction model
        self._correction_scaler = StandardScaler()
        X_scaled = self._correction_scaler.fit_transform(X)

        self._correction_model = RandomForestRegressor(
            n_estimators=n_correction_trees,
            max_depth=min(4, self.base.config.max_depth),
            random_state=self.base.config.random_state,
        )
        self._correction_model.fit(X_scaled, residuals)
        self._is_tuned = True

        # Step 4: evaluate combined
        correction_preds = self._correction_model.predict(X_scaled)
        combined_preds = base_preds + correction_preds

        rmse_corrected = float(np.sqrt(mean_squared_error(y, combined_preds)))
        r2_corrected = float(r2_score(y, combined_preds))

        return {
            "rmse_base": rmse_base,
            "rmse_corrected": rmse_corrected,
            "r2_corrected": r2_corrected,
        }

    def predict(self, features: dict[str, float]) -> MLPrediction:
        """Predict fatigue life using base + correction.

        Parameters
        ----------
        features : dict[str, float]
            Feature dict for one sample.

        Returns
        -------
        MLPrediction
        """

        if not self._is_tuned:
            raise RuntimeError(
                "TransferLearner is not fine-tuned. Call fine_tune() first."
            )

        # Base prediction
        base_pred = self.base.predict(features)
        log_base = base_pred.log_predicted_life

        # Correction
        x = self.base._prepare_single(features)
        x_scaled = self._correction_scaler.transform(x.reshape(1, -1))
        correction = float(self._correction_model.predict(x_scaled)[0])

        log_combined = log_base + correction

        # CI from base prediction, widened slightly
        base_lo, base_hi = base_pred.confidence_interval
        log_lo = np.log10(base_lo) + correction
        log_hi = np.log10(base_hi) + correction

        return MLPrediction(
            predicted_life=10.0 ** log_combined,
            log_predicted_life=log_combined,
            confidence_interval=(10.0 ** log_lo, 10.0 ** log_hi),
            feature_importances=base_pred.feature_importances,
        )

    def domain_adaptation_score(
        self,
        source_features: FatigueFeatures,
        target_features: FatigueFeatures,
    ) -> float:
        """Assess transferability via distribution distance.

        Computes Maximum Mean Discrepancy (MMD) with a Gaussian kernel
        between source and target feature distributions.  Lower values
        indicate better transferability.

        Parameters
        ----------
        source_features, target_features : FatigueFeatures
            Feature matrices from source and target domains.

        Returns
        -------
        float
            MMD^2 estimate (non-negative).
        """

        X_s = source_features.values.copy()
        X_t = target_features.values.copy()

        # Impute NaN
        X_s = np.nan_to_num(X_s, nan=0.0)
        X_t = np.nan_to_num(X_t, nan=0.0)

        # Standardise jointly
        combined = np.vstack([X_s, X_t])
        mu = np.mean(combined, axis=0)
        sigma = np.std(combined, axis=0) + 1e-10
        X_s = (X_s - mu) / sigma
        X_t = (X_t - mu) / sigma

        # Gaussian kernel bandwidth: median heuristic
        from scipy.spatial.distance import cdist

        all_dists = cdist(X_s, X_t, metric="sqeuclidean")
        bandwidth = float(np.median(all_dists)) + 1e-10

        def _rbf_kernel(A: NDArray, B: NDArray) -> NDArray:
            sq_dist = cdist(A, B, metric="sqeuclidean")
            return np.exp(-sq_dist / (2.0 * bandwidth))

        K_ss = _rbf_kernel(X_s, X_s)
        K_tt = _rbf_kernel(X_t, X_t)
        K_st = _rbf_kernel(X_s, X_t)

        n_s = X_s.shape[0]
        n_t = X_t.shape[0]

        mmd_sq = (
            np.sum(K_ss) / (n_s * n_s)
            - 2.0 * np.sum(K_st) / (n_s * n_t)
            + np.sum(K_tt) / (n_t * n_t)
        )

        return float(max(0.0, mmd_sq))
