"""Async PostgreSQL setup — SQLAlchemy 2.0 + asyncpg."""
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Boolean, DateTime, Text, Numeric, BigInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Copy .env.example to .env and set it before starting the server."
    )

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class TransactionORM(Base):
    __tablename__ = "transactions"

    id:            Mapped[uuid.UUID]       = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id:    Mapped[str]             = mapped_column(String(20), nullable=False)
    amount:        Mapped[float]           = mapped_column(Numeric(12, 2), nullable=False)
    status:        Mapped[str]             = mapped_column(String(30), default="Pending")
    is_fraudulent: Mapped[bool]            = mapped_column(Boolean, default=False)
    risk_score:    Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    latitude:      Mapped[Optional[float]] = mapped_column(Float)
    longitude:     Mapped[Optional[float]] = mapped_column(Float)
    created_at:    Mapped[datetime]        = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )


class AuditLogORM(Base):
    __tablename__ = "audit_log"

    id:             Mapped[int]            = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type:     Mapped[str]            = mapped_column(String(50), nullable=False)
    old_status:     Mapped[Optional[str]]  = mapped_column(String(30))
    new_status:     Mapped[Optional[str]]  = mapped_column(String(30))
    notes:          Mapped[Optional[str]]  = mapped_column(Text)
    created_at:     Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# DB Utilities
# ─────────────────────────────────────────────────────────────────────────────

async def get_db():
    """Yield an async session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Verify DB connectivity at startup."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    print("PostgreSQL connection established.")


async def create_next_month_partition():
    """
    Auto-create the next calendar month's transaction partition if it doesn't exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    Prevents transactions from falling into the slow catch-all 'transactions_default'.
    """
    async with engine.connect() as conn:
        await conn.execute(text("""
            DO $$
            DECLARE
                next_month      DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
                partition_name  TEXT := 'transactions_' || TO_CHAR(next_month, 'YYYY_MM');
                start_date      TEXT := TO_CHAR(next_month, 'YYYY-MM-DD');
                end_date        TEXT := TO_CHAR(next_month + INTERVAL '1 month', 'YYYY-MM-DD');
            BEGIN
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS %I PARTITION OF transactions '
                    'FOR VALUES FROM (%L) TO (%L)',
                    partition_name, start_date, end_date
                );
            END $$;
        """))
        await conn.commit()
