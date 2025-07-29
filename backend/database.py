"""
MongoDB Database Connection Module - Local MongoDB Community Edition Only
This module handles connection to a local MongoDB instance running on localhost:27017
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .config import settings
import logging
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import IndexModel, ASCENDING, DESCENDING
from typing import Dict, Any, List
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database connection variables
_db_client: AsyncIOMotorClient = None
db: AsyncIOMotorDatabase = None
_connection_status = "disconnected"

async def connect_to_mongo():
    """
    Connect to local MongoDB Community Edition
    
    This function establishes a connection to MongoDB running locally on port 27017.
    It replaces the previous multi-connection approach (Atlas + fallback) with a 
    single, direct connection to local MongoDB.
    """
    global _db_client, db, _connection_status 
    
    # Local MongoDB connection configuration
    local_mongodb_url = "mongodb://localhost:27017/"
    database_name = settings.MONGODB_DB_NAME or "fraud_detection_db"
    
    try:
        logger.info("[INFO] Connecting to local MongoDB Community Edition...")
        logger.info(f"[INFO] Connection URL: {local_mongodb_url}")
        logger.info(f"[INFO] Database name: {database_name}")
        
        # Create connection to local MongoDB
        _db_client = AsyncIOMotorClient(
            local_mongodb_url,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=10000,          # 10 second socket timeout
            maxPoolSize=10,                 # Maximum 10 connections in pool
            minPoolSize=1                   # Minimum 1 connection in pool
        )
        
        # Test the connection by pinging the server
        await asyncio.wait_for(_db_client.admin.command('ping'), timeout=10)
        logger.info("[SUCCESS] MongoDB ping successful")
        
        # Connect to the specific database
        db = _db_client[database_name]
        
        # Test database access by listing collections
        collections = await db.list_collection_names()
        logger.info(f"[SUCCESS] Database '{database_name}' accessible")
        logger.info(f"[INFO] Existing collections: {collections}")
        
        # Create indexes for optimal query performance
        await create_indexes()
        
        # Update connection status
        _connection_status = "connected"
        logger.info("[SUCCESS] Local MongoDB connection established successfully!")
        
    except ConnectionFailure as e:
        error_msg = f"Failed to connect to MongoDB at {local_mongodb_url}"
        logger.error(f"[ERROR] {error_msg}: {str(e)}")
        logger.error("[ERROR] Make sure MongoDB Community Edition is running on localhost:27017")
        logger.error("[ERROR] You can start MongoDB with: 'mongod' or 'net start MongoDB'")
        _connection_status = "failed"
        raise ConnectionError(f"{error_msg}. Is MongoDB running?")
        
    except ServerSelectionTimeoutError as e:
        error_msg = "MongoDB server selection timeout"
        logger.error(f"[ERROR] {error_msg}: {str(e)}")
        logger.error("[ERROR] MongoDB server is not responding. Check if MongoDB service is running.")
        _connection_status = "failed"
        raise ConnectionError(f"{error_msg}. Check MongoDB service status.")
        
    except Exception as e:
        error_msg = f"Unexpected error connecting to MongoDB"
        logger.error(f"[ERROR] {error_msg}: {str(e)}")
        _connection_status = "failed"
        raise ConnectionError(f"{error_msg}: {str(e)}")

async def create_indexes():
    """
    Create database indexes for optimal query performance
    
    This function creates indexes on the transactions collection to improve
    query performance for common operations like filtering by email, prediction, etc.
    """
    try:
        collection = db["transactions"]
        
        # Define indexes for common query patterns
        indexes = [
            IndexModel([("email", ASCENDING)], name="email_idx"),
            IndexModel([("prediction", ASCENDING)], name="prediction_idx"),
            IndexModel([("timestamp", DESCENDING)], name="timestamp_desc_idx"),
            IndexModel([("amount", DESCENDING)], name="amount_desc_idx"),
            IndexModel([("email", ASCENDING), ("timestamp", DESCENDING)], name="email_timestamp_idx"),
            IndexModel([("prediction", ASCENDING), ("timestamp", DESCENDING)], name="prediction_timestamp_idx")
        ]
        # Create indexes
        await collection.create_indexes(indexes)
        logger.info("[SUCCESS] Database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"[WARNING] Failed to create indexes: {e}")
        # Don't raise exception as indexes are optional for functionality

async def close_mongo_connection():
    """
    Close MongoDB connection gracefully
    
    This function properly closes the MongoDB connection when the application shuts down.
    """
    global _db_client, _connection_status
    try:
        if _db_client:
            _db_client.close()
            logger.info("[SUCCESS] MongoDB connection closed successfully")
            _connection_status = "disconnected"
    except Exception as e:
        logger.error(f"[ERROR] Error closing MongoDB connection: {e}")

async def get_database():
    """
    Get database instance with connection check
    
    Returns the database instance if connected, raises exception if not connected.
    """
    if _connection_status != "connected":
        raise Exception("Database not connected. Call connect_to_mongo() first.")
    if db is None:
        raise Exception("Database instance is None. Connection may have failed.")
    return db

def get_connection_status() -> str:
    """
    Get current connection status
    
    Returns: "connected", "disconnected", or "failed"
    """
    return _connection_status

def is_connected() -> bool:
    """
    Check if MongoDB is connected
    
    Returns True if successfully connected to local MongoDB, False otherwise.
    """
    return _connection_status == "connected"

# Note: Removed fallback mode functions as we now require MongoDB to be running
# If you need fallback functionality, consider implementing a separate fallback service