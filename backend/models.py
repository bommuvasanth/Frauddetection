from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TransactionBase(BaseModel):
    email: str
    amount: float
    quantity: int
    customer_age: int
    account_age: int
    transaction_hour: int

class TransactionCreate(TransactionBase):
    prediction: str # 'Fraud' or 'Legit'

class TransactionInDB(TransactionCreate):
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True # updated from orm_mode = True for Pydantic v2 