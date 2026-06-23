"""Admin API — ledger stats, transaction history, volume trends, account management."""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from app.models.database import get_db, TransactionORM
from app.models.schemas import LedgerSummaryResponse, TransactionDetail
from app.services.ledger_service import LedgerService

from fastapi import APIRouter



logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


def _empty_ledger_summary() -> LedgerSummaryResponse:
    return LedgerSummaryResponse(
        total_volume=0.0,
        fraud_count=0,
        throughput=0.0,
        transactions=[],
        status_breakdown={"Approved": 0, "Declined": 0, "Awaiting Verification": 0},
    )



# ─────────────────────────────────────────────────────────────────────────────
# Inline schema (used only in this router)
# ─────────────────────────────────────────────────────────────────────────────
class AccountStatusUpdate(BaseModel):
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/ledger-summary
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/ledger-summary", response_model=LedgerSummaryResponse)
async def get_ledger_summary(db: AsyncSession = Depends(get_db)):
    """
Aggregated ledger stats — total volume, fraud count, throughput (TPS),
status breakdown, and the 20 most recent transactions.

Polled by the Streamlit dashboard every few seconds.
"""
    try:
        query = text("""
            SELECT
                SUM(total_volume)        AS total_volume,
                SUM(fraud_flagged_count) AS fraud_count,
                SUM(approved_count)      AS approved,
                SUM(declined_count)      AS declined,
                SUM(pending_mfa_count)   AS pending
            FROM vw_ledger_summary;
        """)
        row = (await db.execute(query)).fetchone()

        if not row or row[0] is None:
            return _empty_ledger_summary()

        # Real TPS from LedgerService
        summary = await LedgerService.get_ledger_summary(db)
        recent_txns = await LedgerService.get_recent_transactions(db, limit=20)

        return LedgerSummaryResponse(
            total_volume=float(row.total_volume or 0),
            fraud_count=int(row.fraud_count or 0),
            throughput=summary["throughput"],
            transactions=recent_txns,
            status_breakdown={
                "Approved":              int(row.approved or 0),
                "Declined":              int(row.declined or 0),
                "Awaiting Verification": int(row.pending or 0),
            },
        )
    except Exception as exc:
        logger.warning("Ledger summary unavailable, returning empty response: %s", exc)
        return _empty_ledger_summary()


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/transactions
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/transactions", response_model=List[TransactionDetail])
async def get_recent_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
Paginated transaction list, newest first.

Use the `limit` query parameter (1–500, default 50) to control page size.
"""
    try:
        rows = (
            await db.execute(
                select(TransactionORM).order_by(TransactionORM.created_at.desc()).limit(limit)
            )
        ).scalars().all()
        return [TransactionDetail.model_validate(r) for r in rows]
    except Exception as exc:
        logger.warning("Recent transactions unavailable, returning empty list: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/volume-trend
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/volume-trend")
async def get_volume_trend(db: AsyncSession = Depends(get_db)):
    """
Hourly transaction volume and fraud count for the last 24 hours.

Returns a list of 24 objects: `{ hour, total, fraud_count }`.
Used by the volume trend chart on the dashboard.
"""
    try:
        return await LedgerService.get_volume_trend(db)
    except Exception as exc:
        logger.warning("Volume trend unavailable, returning empty list: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/audit-log
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
Recent status-change audit events from the `audit_log` table.

Each event records: `transaction_id`, `event_type`, `old_status`,
`new_status`, `notes`, and `created_at`.

Use `limit` (1–200, default 50) to control how many rows are returned.
"""
    try:
        rows = (await db.execute(text(f"""
            SELECT id, transaction_id, event_type, old_status, new_status, notes, created_at
            FROM audit_log
            ORDER BY created_at DESC
            LIMIT {limit}
        """))).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as exc:
        logger.warning("Audit log unavailable, returning empty list: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GET /admin/accounts
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/accounts")
async def get_all_accounts(
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
List all customer accounts.

Pass `?search=ACC12345` or `?search=nikunj` to filter by account ID or name.
"""
    try:
        return await LedgerService.get_all_accounts(db, search=search)
    except Exception as exc:
        logger.warning("Accounts unavailable, returning empty list: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /admin/accounts/{account_id}/status
# ─────────────────────────────────────────────────────────────────────────────
@router.patch("/accounts/{account_id}/status")
async def update_account_status(
    account_id: str,
    payload: AccountStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
Update an account's status.

Valid values for `status`: `Active`, `Suspended`, `Blocked`.
Suspended accounts are rejected by the PostgreSQL trigger on the next transaction attempt.
"""
    try:
        found = await LedgerService.update_account_status(db, account_id, payload.status)
        if not found:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")
        return {"message": f"Account {account_id} is now {payload.status}."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Account status update unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Account status updates are unavailable right now.")

@router.get("/some-endpoint")
async def some_function():
    return {"message": "Success"}
