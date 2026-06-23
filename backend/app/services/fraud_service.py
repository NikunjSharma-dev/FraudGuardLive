"""
FraudService — async orchestrator for ML inference, Redis feature store, and OTP.

FraudService owns all async I/O (Redis reads/writes, geo-velocity, OTP).
CPU-bound inference is delegated to FraudPredictor via run_in_executor.

  FraudService  →  FraudPredictor.predict()
                     StandardScaler → IsolationForest → XGBoost → SHAP
                     returns FraudPrediction dataclass
"""
import os
import asyncio
import secrets
import json
import logging
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

import pandas as pd
import redis.asyncio as aioredis

from app.ml.predict import FraudPredictor, FraudPrediction

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Redis client (singleton, lazy-init)
# ─────────────────────────────────────────────────────────────────────────────
_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return (or create) the module-level async Redis client."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError(
                "REDIS_URL environment variable is not set. "
                "Set it in your .env file before starting the server."
            )
        _redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Haversine distance (km)
# ─────────────────────────────────────────────────────────────────────────────
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(a, 0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# FraudService
# ─────────────────────────────────────────────────────────────────────────────
class FraudService:
    """
    Single public interface used by the transactions API.
    All ML work is delegated to FraudPredictor; this class owns async I/O only.
    """

    _models_loaded: bool = False

    # ── Model Loading ─────────────────────────────────────────────────────────

    @classmethod
    def load_models(cls) -> None:
        """
        Load ML artifacts via FraudPredictor (single source of truth).
        Safe to call multiple times.
        """
        if cls._models_loaded:
            return
        FraudPredictor.load()
        cls._models_loaded = FraudPredictor.is_loaded()
        logger.info("ML models loaded.")

    # ── Redis: Behavioral Context ─────────────────────────────────────────────

    @classmethod
    async def _read_account_context(cls, account_id: str) -> dict:
        """Fetch stored behavioral context for an account from Redis."""
        r = get_redis()
        raw = await r.get(f"ctx:{account_id}")
        return json.loads(raw) if raw else {}

    @classmethod
    async def _write_account_context(
        cls,
        account_id: str,
        amount: float,
        lat: float,
        lon: float,
        prev_ctx: dict,
    ) -> None:
        """
        Update behavioral context after a transaction is processed.
        Uses exponential moving average (α=0.15) for amount stats.
        7-day TTL — inactive accounts auto-expire.
        """
        r = get_redis()
        prev_avg = prev_ctx.get("amount_avg", amount)
        prev_std = prev_ctx.get("amount_std", 0.0)

        alpha   = 0.15
        new_avg = alpha * amount + (1 - alpha) * prev_avg
        new_std = max(alpha * abs(amount - prev_avg) + (1 - alpha) * prev_std, 1.0)

        ctx = {
            "last_lat":   lat,
            "last_lon":   lon,
            "last_tx_ts": datetime.now().timestamp(),
            "amount_avg": round(new_avg, 4),
            "amount_std": round(new_std, 4),
        }
        await r.setex(f"ctx:{account_id}", 604_800, json.dumps(ctx))

    @classmethod
    async def _get_tx_count_10m(cls, account_id: str) -> int:
        """
        Increment and return the transaction count for the last 10 minutes.
        Uses Redis INCR + per-key TTL so the window resets automatically.
        """
        r   = get_redis()
        key = f"txcount10m:{account_id}"
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, 600)   # Set TTL only on first increment
        return int(count)

    # ── Feature Engineering ───────────────────────────────────────────────────

    @classmethod
    async def _build_features(
        cls,
        account_id: str,
        amount: float,
        lat: float,
        lon: float,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Build the feature vector from live Redis context.
        Returns (feature_df, prev_ctx) — prev_ctx is needed to update Redis afterward.

        Feature set:
          amount            — transaction value
          geo_velocity      — km/h between last known location and current
          tx_count_10m      — transaction frequency in last 10 min (Redis INCR)
          hour_of_day       — 0-23 (captures unusual-hour fraud patterns)
          is_weekend        — 0/1 (Saturday=5, Sunday=6)
          amount_z_score    — std deviations from this account's EMA average
          time_since_last_tx— seconds since last transaction (velocity signal)

        Safety net: any column required by the pre-trained model but absent from
        the computed dict is filled with 0.0 — prevents runtime crashes if the
        model was trained with extra features.
        """
        ctx          = await cls._read_account_context(account_id)
        tx_count_10m = await cls._get_tx_count_10m(account_id)
        now          = datetime.now()

        # ── Geo velocity ──────────────────────────────────────────────────────
        geo_velocity       = 0.0
        time_since_last_tx = 86_400.0   # Default 24 h for first-time accounts

        if ctx.get("last_lat") is not None and ctx.get("last_tx_ts") is not None:
            dist_km            = _haversine_km(ctx["last_lat"], ctx["last_lon"], lat, lon)
            elapsed_sec        = max(now.timestamp() - ctx["last_tx_ts"], 1.0)
            elapsed_hr         = elapsed_sec / 3600.0
            geo_velocity       = min(dist_km / elapsed_hr, 2000.0)   # Capped at ~Mach speed
            time_since_last_tx = elapsed_sec

        # ── Amount z-score ────────────────────────────────────────────────────
        acct_avg       = ctx.get("amount_avg", amount)
        acct_std       = max(ctx.get("amount_std", 1.0), 1.0)
        amount_z_score = (amount - acct_avg) / acct_std

        features = {
            "amount":             amount,
            "geo_velocity":       round(geo_velocity, 4),
            "tx_count_10m":       tx_count_10m,
            "hour_of_day":        now.hour,
            "is_weekend":         int(now.weekday() >= 5),
            "amount_z_score":     round(amount_z_score, 4),
            "time_since_last_tx": round(time_since_last_tx, 4),
        }

        # Safety net: fill any unknown training columns with 0.0
        feature_cols = FraudPredictor._feature_columns or list(features.keys())
        for col in feature_cols:
            features.setdefault(col, 0.0)

        df = pd.DataFrame([features])[feature_cols]
        return df, ctx

    # ── ML Inference (sync, runs in executor) ─────────────────────────────────

    @classmethod
    def _run_inference(cls, df: pd.DataFrame) -> FraudPrediction:
        """
        Synchronous wrapper around FraudPredictor.predict().
        Called via run_in_executor so it never blocks the event loop.
        """
        return FraudPredictor.predict(df)

    # ── Public: Evaluate Transaction ──────────────────────────────────────────

    @classmethod
    async def evaluate_transaction(
        cls,
        account_id: str,
        amount: float,
        lat: float,
        lon: float,
    ) -> tuple[float, str, dict]:
        """
        Full async pipeline:
          1. Read account context from Redis
          2. Build feature DataFrame
          3. Run CPU-bound ML inference in a thread pool
          4. Fire-and-forget Redis context update

        Returns: (risk_score, action, explanation)
        """
        if not cls._models_loaded:
            logger.warning("Models not loaded — defaulting to Approved.")
            return 0.0, "Approved", {}

        feature_df, prev_ctx = await cls._build_features(account_id, amount, lat, lon)

        loop    = asyncio.get_event_loop()
        result: FraudPrediction = await loop.run_in_executor(
            None, cls._run_inference, feature_df
        )

        # Non-blocking Redis update (failures here don't affect the response)
        asyncio.create_task(
            cls._write_account_context(account_id, amount, lat, lon, prev_ctx)
        )

        return result.risk_score, result.action, result.explanation

    # ── OTP Management ────────────────────────────────────────────────────────

    @classmethod
    async def generate_otp(cls, transaction_id: str) -> str:
        """
        Generate a 6-digit OTP using secrets.
        Stored in Redis with a 5-minute TTL.
        In production: deliver via Twilio SMS.
        """
        otp = str(secrets.randbelow(900_000) + 100_000)   # Always 6 digits
        await get_redis().setex(f"otp:{transaction_id}", 300, otp)
        logger.info(f"🔑 OTP issued for transaction {transaction_id}")
        return otp

    @classmethod
    async def verify_otp(cls, transaction_id: str, submitted: str) -> bool:
        """
        Validate submitted OTP against Redis.
        Single-use: key is deleted immediately on success.
        Returns False for wrong code or expired token.
        """
        r      = get_redis()
        stored = await r.get(f"otp:{transaction_id}")
        if not stored:
            return False
        if stored == submitted:
            await r.delete(f"otp:{transaction_id}")
            return True
        return False
