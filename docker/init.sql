-- ============================================================
-- Fraud Detection System — Database Initialization
-- PostgreSQL 16
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- ACCOUNTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS accounts (
    account_id   VARCHAR(20) PRIMARY KEY,
    owner_name   VARCHAR(100) NOT NULL,
    status       VARCHAR(20) DEFAULT 'Active'  -- Active | Suspended | Closed
                 CHECK (status IN ('Active', 'Suspended', 'Closed')),
    daily_limit  NUMERIC(12, 2) DEFAULT 500000.00,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Seed a few test accounts
INSERT INTO accounts (account_id, owner_name, status, daily_limit) VALUES
    ('ACC10294', 'Nik Kumar',      'Active',    500000.00),
    ('ACC20381', 'Priya Singh',    'Active',    250000.00),
    ('ACC30019', 'Raj Verma',      'Suspended', 100000.00),
    ('ACC40772', 'Anita Joshi',    'Active',    1000000.00)
ON CONFLICT DO NOTHING;

-- ============================================================
-- TRANSACTIONS TABLE (partitioned by month)
-- ============================================================
CREATE TABLE IF NOT EXISTS transactions (
    id            UUID DEFAULT gen_random_uuid(),
    account_id    VARCHAR(20) NOT NULL REFERENCES accounts(account_id),
    amount        NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
    status        VARCHAR(30) DEFAULT 'Pending'
                  CHECK (status IN ('Pending', 'Approved', 'Declined', 'Awaiting Verification', 'Verified')),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    risk_score    FLOAT DEFAULT 0.0,
    latitude      FLOAT,
    longitude     FLOAT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions (Updated for current 2026 dates)
CREATE TABLE IF NOT EXISTS transactions_2026_05 PARTITION OF transactions
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

CREATE TABLE IF NOT EXISTS transactions_2026_06 PARTITION OF transactions
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE TABLE IF NOT EXISTS transactions_2026_07 PARTITION OF transactions
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

CREATE TABLE IF NOT EXISTS transactions_2026_08 PARTITION OF transactions
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');

-- Default catch-all partition
CREATE TABLE IF NOT EXISTS transactions_default PARTITION OF transactions DEFAULT;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_txn_account_id  ON transactions (account_id);
CREATE INDEX IF NOT EXISTS idx_txn_status      ON transactions (status);
CREATE INDEX IF NOT EXISTS idx_txn_created_at  ON transactions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_fraudulent  ON transactions (is_fraudulent) WHERE is_fraudulent = TRUE;

-- ============================================================
-- AUDIT LOG TABLE
-- ============================================================
-- Note: transaction_id is a logical reference. A hard FK to a 
-- composite PK partitioned table can cause lock contention at scale.
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    transaction_id UUID NOT NULL,
    event_type   VARCHAR(50) NOT NULL,  -- e.g. TRIGGER_DECLINE, ML_FLAG, MFA_VERIFIED
    old_status   VARCHAR(30),
    new_status   VARCHAR(30),
    notes        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_txn_id ON audit_log (transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log (created_at DESC);

-- ============================================================
-- TRIGGER FUNCTION: Hard business-rule enforcement
-- Runs BEFORE INSERT — no application-level bypass possible
-- ============================================================
CREATE OR REPLACE FUNCTION verify_transaction_limits()
RETURNS TRIGGER AS $$
DECLARE
    acct_status   VARCHAR(20);
    acct_limit    NUMERIC(12, 2);
    daily_total   NUMERIC(12, 2);
BEGIN
    -- Check 1: Is the account suspended or closed?
    -- Added FOR UPDATE to lock the account row to prevent race conditions during concurrent transactions
    SELECT status, daily_limit
    INTO acct_status, acct_limit
    FROM accounts
    WHERE account_id = NEW.account_id
    FOR UPDATE;

    IF acct_status IN ('Suspended', 'Closed') THEN
        NEW.status := 'Declined';
        INSERT INTO audit_log (transaction_id, event_type, old_status, new_status, notes)
        VALUES (NEW.id, 'TRIGGER_DECLINE', 'Pending', 'Declined',
                'Account status: ' || acct_status);
        RETURN NEW;
    END IF;

    -- Check 2: Exceeds single-transaction hard cap?
    IF NEW.amount > acct_limit THEN
        NEW.status := 'Declined';
        INSERT INTO audit_log (transaction_id, event_type, old_status, new_status, notes)
        VALUES (NEW.id, 'TRIGGER_DECLINE', 'Pending', 'Declined',
                'Amount ' || NEW.amount || ' exceeds limit ' || acct_limit);
        RETURN NEW;
    END IF;

    -- Check 3: Daily rolling spend limit exceeded?
    SELECT COALESCE(SUM(amount), 0)
    INTO daily_total
    FROM transactions
    WHERE account_id = NEW.account_id
      AND status IN ('Approved', 'Awaiting Verification', 'Verified')
      AND created_at >= date_trunc('day', NOW());

    IF (daily_total + NEW.amount) > acct_limit THEN
        NEW.status := 'Declined';
        INSERT INTO audit_log (transaction_id, event_type, old_status, new_status, notes)
        VALUES (NEW.id, 'TRIGGER_DECLINE', 'Pending', 'Declined',
                'Daily limit exceeded. Spent today: ' || daily_total);
        RETURN NEW;
    END IF;

    -- All checks passed — approve for ML evaluation
    NEW.status := 'Approved';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger
DROP TRIGGER IF EXISTS txn_limit_trigger ON transactions;
CREATE TRIGGER txn_limit_trigger
BEFORE INSERT ON transactions
FOR EACH ROW EXECUTE FUNCTION verify_transaction_limits();

-- ============================================================
-- TRIGGER FUNCTION: Audit status changes
-- ============================================================
CREATE OR REPLACE FUNCTION log_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO audit_log (transaction_id, event_type, old_status, new_status)
        VALUES (NEW.id, 'STATUS_CHANGE', OLD.status, NEW.status);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS txn_status_audit_trigger ON transactions;
CREATE TRIGGER txn_status_audit_trigger
AFTER UPDATE ON transactions
FOR EACH ROW EXECUTE FUNCTION log_status_change();

-- ============================================================
-- VIEW: Admin summary view
-- ============================================================
DROP VIEW IF EXISTS vw_ledger_summary;
CREATE VIEW vw_ledger_summary AS
SELECT
    DATE_TRUNC('hour', created_at)   AS hour,
    COUNT(*)                          AS total_transactions,
    SUM(amount)                       AS total_volume,
    SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END)               AS approved_count,
    SUM(CASE WHEN status = 'Declined' THEN 1 ELSE 0 END)               AS declined_count,
    SUM(CASE WHEN status = 'Awaiting Verification' THEN 1 ELSE 0 END)  AS pending_mfa_count,
    SUM(CASE WHEN is_fraudulent = TRUE THEN 1 ELSE 0 END)              AS fraud_flagged_count,
    ROUND(AVG(risk_score)::numeric, 4)                                 AS avg_risk_score
FROM transactions
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;