"""LedgerService — all PostgreSQL read/write operations for transactions and accounts."""
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.models.database import TransactionORM
from app.models.schemas import TransactionSubmitRequest


class LedgerService:

    # ── Transactions: Write ───────────────────────────────────────────────────

    @staticmethod
    async def create_transaction(
        db: AsyncSession, payload: TransactionSubmitRequest
    ) -> TransactionORM:
        """
        Insert a new transaction row.
        The PostgreSQL BEFORE INSERT trigger evaluates hard business rules
        and sets status = 'Approved' or 'Declined' before commit.
        """
        new_txn = TransactionORM(
            account_id=payload.account_id,
            amount=payload.amount,
            latitude=payload.lat,
            longitude=payload.lon,
            status="Pending",
            risk_score=0.0,
        )
        db.add(new_txn)
        await db.commit()
        await db.refresh(new_txn)
        return new_txn

    @staticmethod
    async def update_transaction_status(
        db: AsyncSession,
        txn_id: UUID,
        status: str,
        is_fraudulent: bool = False,
        risk_score: float = 0.0,
    ) -> None:
        """Update status, fraud flag, and risk score after ML evaluation or OTP."""
        result = await db.execute(
            select(TransactionORM).where(TransactionORM.id == txn_id)
        )
        txn = result.scalars().first()
        if txn:
            txn.status        = status
            txn.is_fraudulent = is_fraudulent
            txn.risk_score    = risk_score
            await db.commit()

    # ── Transactions: Read ────────────────────────────────────────────────────

    @staticmethod
    async def get_ledger_summary(db: AsyncSession) -> dict:
        """
        Returns aggregate stats: total volume, fraud count,
        throughput, and status breakdown. Used by /admin/ledger-summary.
        """
        row = (await db.execute(text("""
            SELECT
                COALESCE(SUM(amount), 0)                                            AS total_volume,
                COUNT(CASE WHEN is_fraudulent = TRUE THEN 1 END)                   AS fraud_count,
                COALESCE(
                    SUM(CASE WHEN created_at >= NOW() - INTERVAL '1 minute' THEN 1 ELSE 0 END)
                    / 60.0, 0
                )                                                                   AS throughput,
                COUNT(CASE WHEN status = 'Approved'             THEN 1 END)        AS approved,
                COUNT(CASE WHEN status = 'Declined'             THEN 1 END)        AS declined,
                COUNT(CASE WHEN status = 'Awaiting Verification' THEN 1 END)       AS awaiting
            FROM transactions
        """))).fetchone()

        if not row:
            return {
                "total_volume": 0.0, "fraud_count": 0, "throughput": 0.0,
                "status_breakdown": {"Approved": 0, "Declined": 0, "Awaiting Verification": 0},
            }

        d = row._mapping
        return {
            "total_volume": float(d["total_volume"]),
            "fraud_count":  int(d["fraud_count"]),
            "throughput":   round(float(d["throughput"]), 2),
            "status_breakdown": {
                "Approved":              int(d["approved"]),
                "Declined":              int(d["declined"]),
                "Awaiting Verification": int(d["awaiting"]),
            },
        }

    @staticmethod
    async def get_volume_trend(db: AsyncSession) -> list[dict]:
        """Hourly transaction volume for the last 24 hours — replaces fake x**2 chart."""
        rows = (await db.execute(text("""
            SELECT
                TO_CHAR(DATE_TRUNC('hour', created_at), 'YYYY-MM-DD HH24:00:00') AS hour,
                COALESCE(SUM(amount), 0)                                           AS volume,
                COUNT(*)                                                            AS tx_count,
                COALESCE(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END), 0)       AS fraud_count
            FROM transactions
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', created_at)
            ORDER BY DATE_TRUNC('hour', created_at) ASC
        """))).fetchall()
        return [
            {
                "hour":        r.hour,
                "volume":      float(r.volume),
                "tx_count":    int(r.tx_count),
                "fraud_count": int(r.fraud_count),
            }
            for r in rows
        ]

    @staticmethod
    async def get_recent_transactions(db: AsyncSession, limit: int = 50) -> list[dict]:
        """Paginated list of the most recent transactions for the admin ledger table."""
        rows = (await db.execute(text("""
            SELECT id, account_id, amount, status, is_fraudulent, risk_score, created_at
            FROM transactions
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit})).fetchall()
        return [
            {
                "id":           str(r.id),
                "account_id":   r.account_id,
                "amount":       float(r.amount),
                "status":       r.status,
                "is_fraudulent": r.is_fraudulent,
                "risk_score":   float(r.risk_score) if r.risk_score else 0.0,
                "created_at":   r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    # ── Accounts ──────────────────────────────────────────────────────────────

    @staticmethod
    async def create_account(
        db: AsyncSession,
        account_id: str,
        full_name: str,
        email: str,
        phone: str,
        kyc_document: str,
    ) -> None:
        """Provision a new customer account. Idempotent via ON CONFLICT DO NOTHING."""
        await db.execute(text("""
            ALTER TABLE accounts
            ADD COLUMN IF NOT EXISTS full_name    VARCHAR(100),
            ADD COLUMN IF NOT EXISTS email        VARCHAR(100),
            ADD COLUMN IF NOT EXISTS phone        VARCHAR(20),
            ADD COLUMN IF NOT EXISTS kyc_document VARCHAR(50)
        """))
        await db.commit()

        await db.execute(text("""
            INSERT INTO accounts (account_id, owner_name, full_name, email, phone, kyc_document, status)
            VALUES (:account_id, :full_name, :full_name, :email, :phone, :kyc_document, 'Active')
            ON CONFLICT (account_id) DO NOTHING
        """), {
            "account_id":   account_id,
            "full_name":    full_name,
            "email":        email,
            "phone":        phone,
            "kyc_document": kyc_document,
        })
        await db.commit()

    @staticmethod
    async def get_all_accounts(db: AsyncSession, search: str | None = None) -> list[dict]:
        """
        Fetch all accounts with optional search.
        Uses parameterized query to prevent SQL injection.
        """
        await db.execute(text("""
            ALTER TABLE accounts
            ADD COLUMN IF NOT EXISTS full_name    VARCHAR(100),
            ADD COLUMN IF NOT EXISTS email        VARCHAR(100),
            ADD COLUMN IF NOT EXISTS phone        VARCHAR(20),
            ADD COLUMN IF NOT EXISTS kyc_document VARCHAR(50)
        """))
        await db.commit()

        if search:
            rows = (await db.execute(text("""
                SELECT account_id, full_name, email, phone, kyc_document,
                       COALESCE(status, 'Active') AS status
                FROM accounts
                WHERE account_id ILIKE :search OR full_name ILIKE :search
                ORDER BY account_id DESC
            """), {"search": f"%{search}%"})).fetchall()
        else:
            rows = (await db.execute(text("""
                SELECT account_id, full_name, email, phone, kyc_document,
                       COALESCE(status, 'Active') AS status
                FROM accounts
                ORDER BY account_id DESC
            """))).fetchall()

        return [dict(r._mapping) for r in rows]

    @staticmethod
    async def update_account_status(
        db: AsyncSession, account_id: str, new_status: str
    ) -> bool:
        """Block or unblock an account. Returns True if found."""
        result = await db.execute(
            text("UPDATE accounts SET status = :status WHERE account_id = :acc_id RETURNING account_id"),
            {"status": new_status, "acc_id": account_id},
        )
        await db.commit()
        return result.fetchone() is not None
