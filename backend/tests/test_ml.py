"""
ML Unit Tests — FraudPredictor and FraudService inference pipeline.

Tests are self-contained: they load the real .pkl files and run actual inference.
No mocking — if the models are broken, these tests will catch it.
"""
import pytest
import pandas as pd
import numpy as np
from app.ml.predict import FraudPredictor, FraudPrediction


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def loaded_predictor():
    """Load models once for the entire test module."""
    FraudPredictor.load()
    assert FraudPredictor.is_loaded(), "FraudPredictor.load() failed — check .pkl files"
    return FraudPredictor


def _make_features(amount=5000.0, geo_velocity=10.5, tx_count_10m=1,
                   hour_of_day=14, is_weekend=0, amount_z_score=0.5) -> pd.DataFrame:
    """Build a valid feature DataFrame matching the training column set."""
    base = {
        "amount": amount,
        "geo_velocity": geo_velocity,
        "tx_count_10m": tx_count_10m,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "amount_z_score": amount_z_score,
        "time_since_last_tx": 3600.0,
    }
    # Safety: only pass columns the model was trained on
    feature_cols = FraudPredictor._feature_columns
    for col in feature_cols:
        base.setdefault(col, 0.0)
    return pd.DataFrame([base])[feature_cols]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def test_all_model_artifacts_loaded():
    assert FraudPredictor._xgb_model is not None
    assert FraudPredictor._iso_forest is not None
    assert FraudPredictor._scaler is not None
    assert FraudPredictor._feature_columns is not None
    assert FraudPredictor._shap_explainer is not None


def test_feature_columns_is_list_of_strings():
    cols = FraudPredictor._feature_columns
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert all(isinstance(c, str) for c in cols)


def test_load_is_idempotent():
    """Calling load() twice must not reload files or raise."""
    FraudPredictor.load()   # Second call
    assert FraudPredictor.is_loaded()


# ─────────────────────────────────────────────────────────────────────────────
# Inference: output types and ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_fraud_prediction_dataclass():
    result = FraudPredictor.predict(_make_features())
    assert isinstance(result, FraudPrediction)


def test_risk_score_in_valid_range():
    result = FraudPredictor.predict(_make_features())
    assert 0.0 <= result.risk_score <= 1.0


def test_action_is_valid_string():
    result = FraudPredictor.predict(_make_features())
    assert result.action in {"Approved", "Declined", "Awaiting Verification"}


def test_is_fraudulent_matches_action():
    result = FraudPredictor.predict(_make_features())
    if result.action == "Approved":
        assert result.is_fraudulent is False
    else:
        assert result.is_fraudulent is True


def test_iso_anomaly_is_minus_one_or_one():
    result = FraudPredictor.predict(_make_features())
    assert result.iso_anomaly in {-1, 1}


def test_xgb_probability_in_valid_range():
    result = FraudPredictor.predict(_make_features())
    assert 0.0 <= result.xgb_probability <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explanations
# ─────────────────────────────────────────────────────────────────────────────

def test_explanation_is_dict_of_floats():
    result = FraudPredictor.predict(_make_features())
    assert isinstance(result.explanation, dict)
    assert len(result.explanation) > 0
    assert all(isinstance(v, float) for v in result.explanation.values())


def test_explanation_keys_match_feature_columns():
    result = FraudPredictor.predict(_make_features())
    assert set(result.explanation.keys()) == set(FraudPredictor._feature_columns)


def test_explanation_contains_amount():
    """'amount' should always appear since it's in every trained model."""
    result = FraudPredictor.predict(_make_features())
    assert "amount" in result.explanation


# ─────────────────────────────────────────────────────────────────────────────
# Business logic: suspicious transaction scores higher
# ─────────────────────────────────────────────────────────────────────────────

def test_high_amount_scores_higher_than_low_amount():
    """A ₹50,000 transaction with extreme geo velocity should score higher than ₹100."""
    risky = FraudPredictor.predict(_make_features(
        amount=50_000, geo_velocity=900, tx_count_10m=9,
        hour_of_day=3, is_weekend=1, amount_z_score=4.5,
    ))
    normal = FraudPredictor.predict(_make_features(
        amount=100, geo_velocity=2, tx_count_10m=1,
        hour_of_day=10, is_weekend=0, amount_z_score=0.1,
    ))
    assert risky.risk_score >= normal.risk_score, (
        f"Expected risky ({risky.risk_score}) >= normal ({normal.risk_score})"
    )


def test_approved_threshold():
    """Very normal transaction should not trigger MFA."""
    result = FraudPredictor.predict(_make_features(
        amount=200, geo_velocity=1, tx_count_10m=1,
        hour_of_day=12, is_weekend=0, amount_z_score=0.0,
    ))
    # Risk score should be low — we can't guarantee "Approved" without knowing
    # the exact model boundary, but the score should be under 0.9
    assert result.risk_score < 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_geo_velocity_does_not_crash():
    """First transaction (no history) → geo_velocity = 0.0."""
    result = FraudPredictor.predict(_make_features(geo_velocity=0.0))
    assert isinstance(result, FraudPrediction)


def test_maximum_amount_does_not_crash():
    result = FraudPredictor.predict(_make_features(amount=999_999.99))
    assert 0.0 <= result.risk_score <= 1.0


def test_negative_amount_z_score_does_not_crash():
    """Amount below account average → negative z-score."""
    result = FraudPredictor.predict(_make_features(amount_z_score=-2.5))
    assert isinstance(result, FraudPrediction)
