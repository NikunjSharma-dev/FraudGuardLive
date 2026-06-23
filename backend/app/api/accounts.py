"""Customer onboarding and account provisioning."""
import random
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services.ledger_service import LedgerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/account", tags=["Customer Onboarding"])

class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: str
    phone: str
    kyc_document: str

@router.post("/signup")
async def create_account(payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    """
    Provision a new customer account.

    Generates a unique `ACC#####` account ID and inserts the customer record
    into the `accounts` table. The returned `account_id` is required for
    all subsequent `/transaction/submit` calls.
    """
    try:
        # 1. Generate unique Account ID
        unique_number = random.randint(10000, 99999)
        new_account_id = f"ACC{unique_number}"

        # 2. Delegate the actual database writing (ALTER and INSERT) to the Service Layer
        await LedgerService.create_account(
            db=db,
            account_id=new_account_id,
            full_name=payload.full_name,
            email=payload.email,
            phone=payload.phone,
            kyc_document=payload.kyc_document
        )

        # 3. Return identical success payload
        return {
            "status": "Success",
            "account_id": new_account_id,
            "message": f"Welcome {payload.full_name}! Your account has been provisioned.",
        }
        
    except Exception as e:
        # Note: Rollback is handled inside the Service Layer or DB dependency if needed
        logger.error(f"Account creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to provision account: {str(e)}")