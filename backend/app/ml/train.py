"""
ML Training Pipeline — Isolation Forest + XGBoost Classifier

Usage:
    python app/ml/train.py

Downloads the Kaggle credit card dataset (or uses synthetic data if unavailable)
and saves trained model artifacts to app/ml/models/.
"""
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "data" / "creditcard.csv"


# ──────────────────────────────────────────────────────────────
# STEP 1: Load or generate data
# ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load real dataset or generate synthetic data for demo."""
    if DATA_PATH.exists():
        print(f"📂 Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df

    print("Dataset not found. Generating synthetic data...")
    np.random.seed(42)
    n = 10000
    fraud_ratio = 0.017

    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        "amount": np.random.exponential(100, n_legit),
        "geo_velocity": np.random.exponential(5, n_legit),
        "tx_count_10m": np.random.poisson(1.5, n_legit),
        "hour_of_day": np.random.randint(6, 22, n_legit),
        "is_weekend": np.random.binomial(1, 0.28, n_legit),
        "amount_z_score": np.random.normal(0, 1, n_legit),
        "Class": 0,
    })

    # Fraudulent transactions — higher amounts, unusual hours, high velocity
    fraud = pd.DataFrame({
        "amount": np.random.exponential(800, n_fraud),
        "geo_velocity": np.random.exponential(200, n_fraud),
        "tx_count_10m": np.random.poisson(6, n_fraud),
        "hour_of_day": np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraud),
        "is_weekend": np.random.binomial(1, 0.5, n_fraud),
        "amount_z_score": np.random.normal(3, 1.5, n_fraud),
        "Class": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df


# ──────────────────────────────────────────────────────────────
# STEP 2: Feature engineering
# ──────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "amount", "geo_velocity", "tx_count_10m",
    "hour_of_day", "is_weekend", "amount_z_score"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive behavioral features from raw transaction data."""
    if "Amount" in df.columns:
        df = df.rename(columns={"Amount": "amount", "Class": "Class"})

    # If using raw Kaggle data, add synthetic behavioral features
    if "geo_velocity" not in df.columns:
        np.random.seed(42)
        n = len(df)
        df["geo_velocity"] = np.where(
            df["Class"] == 1,
            np.random.exponential(150, n),
            np.random.exponential(5, n)
        )
        df["tx_count_10m"] = np.where(
            df["Class"] == 1,
            np.random.poisson(5, n),
            np.random.poisson(1.2, n)
        )
        df["hour_of_day"] = (df.get("Time", pd.Series(np.zeros(n))) / 3600 % 24).astype(int)
        df["is_weekend"] = np.random.binomial(1, 0.28, n)
        df["amount_z_score"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-8)

    return df


# ──────────────────────────────────────────────────────────────
# STEP 3: Train Isolation Forest (unsupervised)
# ──────────────────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    print("\n🌲 Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.017,  # ~1.7% fraud rate
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso


# ──────────────────────────────────────────────────────────────
# STEP 4: Train XGBoost Classifier (supervised)
# ──────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test) -> XGBClassifier:
    print("\n⚡ Training XGBoost Classifier...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred_proba = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, xgb.predict(X_test), target_names=["Legit", "Fraud"]))

    return xgb


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Fraud Detection — ML Training Pipeline")
    print("=" * 60)

    # Load and engineer features
    df = load_data()
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS].values
    y = df["Class"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train models
    iso_forest = train_isolation_forest(X_train)
    xgb_model  = train_xgboost(X_train, y_train, X_test, y_test)

    # Save artifacts
    joblib.dump(iso_forest,       MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(xgb_model,        MODELS_DIR / "xgboost_classifier.pkl")
    joblib.dump(scaler,           MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLUMNS,  MODELS_DIR / "feature_columns.pkl")

    print(f"Models saved to {MODELS_DIR}")
    print("   • isolation_forest.pkl")
    print("   • xgboost_classifier.pkl")
    print("   • scaler.pkl")
    print("   • feature_columns.pkl")


if __name__ == "__main__":
    main()