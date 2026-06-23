"""
ML Prediction Module — fraud risk scoring.

This is the canonical ML inference class. FraudService (the async orchestrator)
delegates all CPU-bound work here, so predict.py is no longer dead code.

Architecture:
  FraudService (async, Redis, OTP) → FraudPredictor.predict() → FraudPrediction
"""
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class FraudPrediction:
    """Fully typed result from a single inference run."""
    risk_score:      float          # 0.0 → safe, 1.0 → highly suspicious
    action:          str            # "Approved" | "Declined" | "Awaiting Verification"
    is_fraudulent:   bool           # True when action != "Approved"
    iso_anomaly:     int            # -1 = anomaly, 1 = normal (Isolation Forest)
    xgb_probability: float          # Raw XGBoost fraud probability
    explanation:     dict[str, float] = field(default_factory=dict)  # SHAP attributions


class FraudPredictor:
    """
    Wraps Isolation Forest + XGBoost + SHAP into a single predict() call.
    Models are loaded once at startup and cached as class attributes.
    FraudService.load_models() calls FraudPredictor.load() — no duplicate loading.
    """
    _iso_forest      = None
    _xgb_model       = None
    _scaler          = None
    _feature_columns = None
    _shap_explainer  = None

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> None:
        """Load all .pkl artifacts from disk. Safe to call multiple times (idempotent)."""
        if cls._iso_forest is not None:
            return  # Already loaded

        cls._scaler          = joblib.load(MODELS_DIR / "scaler.pkl")
        cls._iso_forest      = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        cls._xgb_model       = joblib.load(MODELS_DIR / "xgboost_classifier.pkl")
        cls._feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

        # Build SHAP TreeExplainer once — expensive to create, cheap to reuse
        cls._shap_explainer  = shap.TreeExplainer(cls._xgb_model)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._iso_forest is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    @classmethod
    def predict(cls, features_df: pd.DataFrame) -> FraudPrediction:
        """
        Score a single transaction. Accepts a DataFrame whose columns match
        the training feature set (cls._feature_columns).

        Steps:
          1. Scale features with StandardScaler
          2. Isolation Forest: -1 = anomaly flag
          3. XGBoost: fraud probability
          4. Ensemble: XGB (primary) + IF anomaly bump
          5. SHAP: per-feature attributions
          6. Threshold logic → action

        Returns a FraudPrediction dataclass.
        """
        if not cls.is_loaded():
            raise RuntimeError("FraudPredictor.load() must be called before predict().")

        # 1. Scale
        X_scaled = cls._scaler.transform(features_df)

        # 2. Isolation Forest  (-1 = anomaly)
        iso_result = int(cls._iso_forest.predict(X_scaled)[0])
        is_anomaly = iso_result == -1

        # 3. XGBoost fraud probability
        xgb_proba = float(cls._xgb_model.predict_proba(X_scaled)[0][1])

        # 4. Ensemble  (XGB weighted 80%, IF anomaly adds +0.20 bump)
        iso_bump   = 0.20 if is_anomaly else 0.0
        risk_score = round(min(0.80 * xgb_proba + iso_bump, 0.99), 4)

        # 5. Threshold decision
        if risk_score > 0.85:
            action = "Awaiting Verification"
        elif risk_score > 0.60:
            action = "Declined"
        else:
            action = "Approved"

        # 6. SHAP attributions
        shap_values = cls._shap_explainer.shap_values(X_scaled)
        # shap_values is a list for binary classifiers in older shap versions
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        explanation = {
            col: round(float(val), 5)
            for col, val in zip(cls._feature_columns, values)
        }

        return FraudPrediction(
            risk_score=risk_score,
            action=action,
            is_fraudulent=(action != "Approved"),
            iso_anomaly=iso_result,
            xgb_probability=round(xgb_proba, 4),
            explanation=explanation,
        )
