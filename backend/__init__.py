"""Backend module for fraud detection system"""

from .fraud_detection import detect_fraud, detect_fraud_advanced, predict_fraud, map_fields
from .database import connect_to_mongo, close_mongo_connection, get_database
from .crud import insert_transaction, get_analytics, get_user_analytics
from .models import TransactionBase, TransactionCreate, TransactionInDB
from .config import settings

__all__ = [
    'detect_fraud',
    'detect_fraud_advanced', 
    'predict_fraud',
    'map_fields',
    'connect_to_mongo',
    'close_mongo_connection',
    'get_database',
    'insert_transaction',
    'get_analytics',
    'get_user_analytics',
    'TransactionBase',
    'TransactionCreate', 
    'TransactionInDB',
    'settings'
]