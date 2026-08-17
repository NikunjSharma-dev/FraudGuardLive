"""
Live model metrics — computed from the committed model artifacts.

Nothing here is hardcoded: the numbers on the Model Performance page come from
actually loading `backend/app/ml/models/*.pkl` and scoring them against the
dataset shipped in `ml_pipeline/data/engineered_data.csv`.

Caveat surfaced in the UI: the artifacts were fit on that same CSV and no
held-out split was preserved alongside them, so the evaluation split below is
reconstructed (stratified 20%, seed 42) and cannot be guaranteed unseen. The
scores are therefore optimistic — `Metrics.in_sample` carries that flag so the
page can label it honestly rather than passing them off as generalization.

Everything degrades gracefully: if the ML libraries or artifacts are missing
(e.g. the slim Docker image that ships only streamlit_app/), `compute()`
returns None and the page renders an explicit "unavailable" state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Candidate repo roots — streamlit_app/ sits next to backend/ in the repo, but
# the streamlit-only Docker image copies just this folder.
_HERE = Path(__file__).resolve().parent
_ROOTS = [_HERE.parent, _HERE, Path.cwd()]

_MODEL_REL = Path("backend/app/ml/models")
_DATA_REL = Path("ml_pipeline/data/engineered_data.csv")


@dataclass
class Metrics:
    """Everything the Model Performance page renders."""
    roc_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: float

    cm_xgb: list[list[int]]
    cm_iso: list[list[int]]
    cm_ens: list[list[int]]

    iso_precision: float
    iso_recall: float
    ens_precision: float
    ens_recall: float
    ens_roc_auc: float

    roc_fpr: Any
    roc_tpr: Any
    pr_precision: Any
    pr_recall: Any

    importances: pd.DataFrame
    scores_legit: Any
    scores_fraud: Any

    n_eval: int
    n_fraud: int
    features: list[str] = field(default_factory=list)
    in_sample: bool = True
    source: str = ""


def _find(rel: Path) -> Path | None:
    for root in _ROOTS:
        p = root / rel
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def _load_artifacts():
    """Load the pickled models once per server process."""
    import joblib  # imported lazily so a missing dep degrades instead of crashing

    mdir = _find(_MODEL_REL)
    if mdir is None:
        return None
    try:
        return {
            "features": joblib.load(mdir / "feature_columns.pkl"),
            "scaler": joblib.load(mdir / "scaler.pkl"),
            "xgb": joblib.load(mdir / "xgboost_classifier.pkl"),
            "iso": joblib.load(mdir / "isolation_forest.pkl"),
            "dir": str(mdir),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def compute() -> Metrics | None:
    """Score the saved ensemble. Returns None if artifacts/deps are unavailable."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_curve, precision_recall_curve,
        )
    except Exception:
        return None

    art = _load_artifacts()
    if art is None:
        return None

    data_path = _find(_DATA_REL)
    if data_path is None:
        return None

    try:
        df = pd.read_csv(data_path)
        features = list(art["features"])
        if not set(features).issubset(df.columns) or "Class" not in df.columns:
            return None

        X = df[features]
        y = df["Class"].values
        X_scaled = art["scaler"].transform(X)

        # Reconstruct an evaluation split using train.py's convention.
        _, X_eval, _, y_eval = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        xgb, iso = art["xgb"], art["iso"]
        proba = xgb.predict_proba(X_eval)[:, 1]
        pred = xgb.predict(X_eval)

        cm = confusion_matrix(y_eval, pred)
        tn, fp, fn, tp = cm.ravel()

        iso_flag = (iso.predict(X_eval) == -1).astype(int)
        cm_iso = confusion_matrix(y_eval, iso_flag)

        # Mirror predict.py's blend: 0.75 * xgb probability + 0.25 * iso anomaly
        ens_score = 0.75 * proba + 0.25 * iso_flag
        ens_pred = (ens_score >= 0.65).astype(int)
        cm_ens = confusion_matrix(y_eval, ens_pred)

        roc_f, roc_t, _ = roc_curve(y_eval, proba)
        pr_p, pr_r, _ = precision_recall_curve(y_eval, proba)

        imp = pd.DataFrame({
            "Feature": features,
            "Importance": np.asarray(xgb.feature_importances_, dtype=float),
        }).sort_values("Importance", ascending=True)

        return Metrics(
            roc_auc=float(roc_auc_score(y_eval, proba)),
            precision=float(precision_score(y_eval, pred, zero_division=0)),
            recall=float(recall_score(y_eval, pred, zero_division=0)),
            f1=float(f1_score(y_eval, pred, zero_division=0)),
            accuracy=float((tp + tn) / len(y_eval)),
            fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
            cm_xgb=cm.tolist(),
            cm_iso=cm_iso.tolist(),
            cm_ens=cm_ens.tolist(),
            iso_precision=float(precision_score(y_eval, iso_flag, zero_division=0)),
            iso_recall=float(recall_score(y_eval, iso_flag, zero_division=0)),
            ens_precision=float(precision_score(y_eval, ens_pred, zero_division=0)),
            ens_recall=float(recall_score(y_eval, ens_pred, zero_division=0)),
            ens_roc_auc=float(roc_auc_score(y_eval, ens_score)),
            roc_fpr=roc_f,
            roc_tpr=roc_t,
            pr_precision=pr_p,
            pr_recall=pr_r,
            importances=imp,
            scores_legit=proba[y_eval == 0],
            scores_fraud=proba[y_eval == 1],
            n_eval=int(len(y_eval)),
            n_fraud=int(y_eval.sum()),
            features=features,
            in_sample=True,
            source=data_path.name,
        )
    except Exception:
        return None
