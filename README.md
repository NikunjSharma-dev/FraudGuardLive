# FraudGuard

A real-time fraud detection system built on FastAPI, PostgreSQL, Redis, and a dual-layer ML pipeline. Transactions are scored the moment they're submitted — hard business rules fire inside the database itself, then an async ML engine runs Isolation Forest + XGBoost against behavioral features cached in Redis.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7-red.svg)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## How it works

When a transaction comes in, two independent checks run:

1. **PostgreSQL trigger** — fires on every insert, enforcing hard rules: suspended accounts, daily spend limits, blocked regions. No application code can bypass this layer.

2. **ML engine** (async, non-blocking) — reads behavioral context from Redis (transaction velocity, geographic distance from last known location, amount z-score vs. account history), runs it through a StandardScaler → Isolation Forest → XGBoost pipeline, and updates the transaction's status + risk score.

If the ML engine flags a transaction, the API generates a one-time OTP (stored in Redis with a 5-minute TTL), surface it in the backend terminal for demo purposes, and puts the transaction into `Awaiting Verification`. The user has to complete the MFA challenge before the transaction resolves.

A Streamlit dashboard polls `/admin/ledger-summary` for live metrics, charts, and the full transaction table.

---

## Architecture

```
[ Streamlit Dashboard ] ──(HTTP)──▶ [ FastAPI Backend :8000 ]
                                            │            │
                               ┌────────────┘            └──────────────────┐
                          (write)                                    (async eval)
                               ▼                                            ▼
                    ┌─────────────────────┐                  ┌─────────────────────────┐
                    │    PostgreSQL       │                  │       ML Engine         │
                    │  (transactional     │                  │  IsolationForest + XGB  │
                    │   ledger)           │                  │                         │
                    │                     │                  │  reads Redis context    │
                    │  · SQL triggers     │                  │  computes risk score    │
                    │  · audit log        │◀─────────────────│  writes result back     │
                    │  · monthly          │   (status +      └─────────────────────────┘
                    │    partitioning     │    risk_score)              │
                    └─────────────────────┘                            │
                                                               ┌───────▼────────┐
                                                               │     Redis       │
                                                               │  feature store  │
                                                               │  OTP cache      │
                                                               └─────────────────┘
```

---

## Project structure

```
fraudguard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── transactions.py     # submit, verify OTP, resend OTP, explain
│   │   │   ├── admin.py            # ledger summary, volume trends, account mgmt
│   │   │   └── accounts.py         # customer signup / onboarding
│   │   ├── models/
│   │   │   ├── schemas.py          # Pydantic request/response models
│   │   │   └── database.py         # SQLAlchemy ORM + async session factory
│   │   ├── services/
│   │   │   ├── fraud_service.py    # async orchestrator: Redis, OTP, ML dispatch
│   │   │   └── ledger_service.py   # all database operations
│   │   ├── ml/
│   │   │   ├── predict.py          # FraudPredictor: scaler → IF → XGB → SHAP
│   │   │   ├── train.py            # training script (run offline)
│   │   │   └── models/             # .pkl artifacts loaded at startup
│   │   └── main.py                 # FastAPI app, lifespan, CORS, rate limiting
│   └── tests/
│       ├── test_api.py
│       └── test_ml.py
│
├── streamlit_app/
│   └── app.py                      # live dashboard, polling /admin endpoints
│
├── ml_pipeline/
│   ├── notebooks/
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Feature_Engineering.ipynb
│   │   └── 03_Model_Training.ipynb
│   └── data/                       # drop your dataset here
│
├── docker/
│   ├── init.sql                    # schema, triggers, partitions, views
│   └── redis.conf
│
├── .github/workflows/ci.yml
├── docker-compose.yml
└── .env.example
```

---

## Quickstart

**Requirements:** Docker Desktop (includes Compose), Git.

```bash
# 1. Clone
git clone https://github.com/NikunjSharma-dev/FraudGuard.git
cd fraudguard

# 2. Configure
cp .env.example .env
# Defaults work out of the box. Change SECRET_KEY before any real deployment.

# 3. Start everything
docker-compose up --build

#4. To see backend in your terminal 
cd fraudguard
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

#5. Run the dashboard 
cd fraudguard/streamlit_app
streamlit run app.py
```

Services after startup:

| Service             | Address                       |
|---------------------|-------------------------------|
| Streamlit dashboard | http://localhost:8501         |
| FastAPI backend     | http://localhost:8000         |
| Swagger docs        | http://localhost:8000/docs    |
| PostgreSQL          | http://localhost:5432         |
| Redis               | http://localhost:6379         |

## Streamlit Cloud deployment

To publish the dashboard on Streamlit Community Cloud, deploy the repository with `streamlit_app/app.py` as the main file and use the repo-root `requirements.txt` for dependencies.

Set the backend URL in Streamlit secrets or app environment variables:

```toml
# .streamlit/secrets.toml
API_URL = "https://your-public-backend.example.com"
```

The dashboard can still render mock metrics when the backend is offline, but the onboarding and transaction flows require a reachable FastAPI backend.

---

## API

### Create an account

```http
POST /account/signup
Content-Type: application/json

{
  "full_name": "Jane Doe",
  "email": "jane@example.com",
  "phone": "+91-9876543210",
  "kyc_document": "PAN-ABCDE1234F"
}
```

Response includes a generated `account_id` (e.g. `ACC47291`).

---

### Submit a transaction

```http
POST /transaction/submit
Content-Type: application/json

{
  "account_id": "ACC47291",
  "amount": 5000.00,
  "lat": 19.0760,
  "lon": 72.8777
}
```

```json
{
  "transaction_id": "3f8a1b2c-...",
  "status": "Awaiting Verification",
  "risk_score": 0.87,
  "message": "[DEMO] Your OTP: 481920"
}
```

Possible statuses: `Approved`, `Declined`, `Awaiting Verification`.

---

### Verify OTP

```http
PATCH /transaction/{transaction_id}/verify
Content-Type: application/json

{ "otp": "481920" }
```

A valid OTP resolves the transaction to `Verified`. An invalid or expired one moves it to `Declined`. OTPs expire after 5 minutes.

---

### Resend OTP

```http
POST /transaction/{transaction_id}/resend-otp
```

Generates a new OTP and resets the Redis TTL. The old OTP is immediately invalidated.

---

### Ledger summary (admin)

```http
GET /admin/ledger-summary
```

Returns aggregate stats (total volume, fraud count, throughput, status breakdown) plus a recent transaction list. This is what the Streamlit dashboard polls.

---

### Health check

```http
GET /health
```

Returns service status and whether the ML engine loaded successfully.

---

## ML models

**Isolation Forest** — unsupervised anomaly detection. Trained on unlabeled transaction data; flags the top ~1% of anomalous patterns based on amount, geo_velocity, tx_count_10m, hour_of_day, and weekend_flag.

**XGBoost classifier** — supervised binary classifier trained on labeled fraud data (Kaggle Credit Card Fraud Detection dataset). SMOTE was applied during training to handle the severe class imbalance (~0.17% fraud rate). SHAP is used for per-prediction feature attribution.

The two models run in sequence: Isolation Forest output feeds into XGBoost as an additional feature. FraudService delegates all CPU-bound inference to FraudPredictor via `run_in_executor` so the async event loop never blocks.

**Engineered features pulled from Redis:**

| Feature             | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `geo_velocity`      | Haversine distance (km) from last transaction location       |
| `time_since_last_tx`| Seconds elapsed since the account's last transaction         |
| `tx_count_10m`      | Number of transactions from this account in the last 10 min  |
| `amount_z_score`    | Amount normalized against the account's historical mean/std  |
| `hour_of_day`       | Hour of submission (0–23)                                    |
| `is_weekend`        | Binary flag                                                  |

**Model performance:**

| Metric            | XGBoost |
|-------------------|---------|
| ROC-AUC           | 0.974   |
| Fraud recall      | 0.91    |
| Precision         | 0.87    |
| False positive rate | ~2%   |

---

## Database schema

The `transactions` table is range-partitioned by month. A new partition is auto-created at startup if the next month's partition doesn't exist yet.

```sql
CREATE TABLE transactions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id    VARCHAR(20)   NOT NULL,
    amount        NUMERIC(12,2) NOT NULL,
    status        VARCHAR(30)   DEFAULT 'Pending',
    is_fraudulent BOOLEAN       DEFAULT FALSE,
    risk_score    FLOAT,
    latitude      FLOAT,
    longitude     FLOAT,
    created_at    TIMESTAMPTZ   DEFAULT NOW()
) PARTITION BY RANGE (created_at);
```

A trigger on every insert evaluates hard business rules — suspended account status, per-day spend limits — before the row is committed. These rules cannot be skipped at the application level.

---

## Rate limiting

The `/transaction/submit` endpoint is limited to **10 requests per minute per IP** via slowapi. Exceeding the limit returns `HTTP 429`. The limit is enforced in the backend and applies regardless of whether the request comes through the dashboard or directly via the API.

---

## Configuration

Copy `.env.example` to `.env`. The defaults are set for local development and will work without changes:

| Variable          | Default                          | Notes                                      |
|-------------------|----------------------------------|--------------------------------------------|
| `POSTGRES_DB`     | `frauddb`                        |                                            |
| `POSTGRES_USER`   | `frauduser`                      |                                            |
| `POSTGRES_PASSWORD` | `fraudpass`                    | Change before deploying                    |
| `REDIS_URL`       | `redis://localhost:6379/0`       |                                            |
| `SECRET_KEY`      | `change-this-...`                | Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ALLOWED_ORIGINS` | `http://localhost:8501`          | Comma-separated for multiple origins       |
| `ENV`             | `development`                    |                                            |

---

## Running tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests use `fakeredis` and an in-memory SQLite database — no running services needed.

The CI pipeline (`.github/workflows/ci.yml`) runs the full test suite on every push to `main` or `develop`, and on all pull requests against `main`.

---

## Training the ML models locally

Pre-trained `.pkl` artifacts are included in `backend/app/ml/models/`. To retrain on new data:

```bash
cd backend
pip install -r requirements.txt
python app/ml/train.py
```

The notebooks in `ml_pipeline/notebooks/` walk through EDA, feature engineering, and the full training run step by step.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Dataset credit

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group, via Kaggle.
