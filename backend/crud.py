from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any, Optional
from datetime import datetime
from pymongo.errors import DuplicateKeyError
import logging

logger = logging.getLogger(__name__)

async def insert_transaction(db_client: AsyncIOMotorDatabase, transaction: Dict[str, Any]) -> Optional[str]:
    """Insert transaction with enhanced error handling"""
    
    # Normal MongoDB operation
    try:
        if db_client is None:
            raise Exception("Database client is None - MongoDB connection required")
        
        # Ensure timestamp is set
        if "timestamp" not in transaction:
            transaction["timestamp"] = datetime.utcnow()
        
        collection = db_client["transactions"]
        
        result = await collection.insert_one(transaction)
        logger.info(f"[SUCCESS] Transaction inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except DuplicateKeyError as e:
        logger.error(f"[ERROR] Duplicate transaction: {e}")
        raise Exception("Transaction already exists")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error inserting transaction: {e}")
        raise Exception(f"Unexpected error: {e}")

async def get_analytics(db_client: AsyncIOMotorDatabase) -> Dict[str, Any]:
    """Get comprehensive analytics with enhanced error handling"""
    
    # Normal MongoDB operation
    try:
        if db_client is None:
            raise Exception("Database client is None - MongoDB connection required")
        
        collection = db_client["transactions"]
        
        # Use aggregation pipeline for better performance
        analytics_pipeline = [
            {
                "$facet": {
                    "overview": [
                        {
                            "$group": {
                                "_id": None,
                                "total_transactions": {"$sum": 1},
                                "total_amount": {"$sum": "$amount"},
                                "average_amount": {"$avg": "$amount"},
                                "max_amount": {"$max": "$amount"},
                                "min_amount": {"$min": "$amount"}
                            }
                        }
                    ],
                    "by_prediction": [
                        {
                            "$group": {
                                "_id": "$prediction",
                                "count": {"$sum": 1},
                                "total_amount": {"$sum": "$amount"},
                                "avg_amount": {"$avg": "$amount"}
                            }
                        }
                    ],
                    "hourly_distribution": [
                        {
                            "$group": {
                                "_id": "$transaction_hour",
                                "count": {"$sum": 1}
                            }
                        },
                        {"$sort": {"_id": 1}}
                    ],
                    "recent_transactions": [
                        {
                            "$match": {
                                "timestamp": {
                                    "$gte": datetime.utcnow() - datetime.timedelta(hours=24)
                                }
                            }
                        },
                        {"$sort": {"timestamp": -1}},
                        {"$limit": 10}
                    ]
                }
            }
        ]
        
        result = await collection.aggregate(analytics_pipeline).to_list(None)
        
        if not result:
            return {
                "total_transactions": 0,
                "fraud_count": 0,
                "legit_count": 0,
                "fraud_rate_percentage": 0,
                "total_amount": 0,
                "average_amount": 0,
                "max_amount": 0,
                "min_amount": 0,
                "recent_transactions": [],
                "hourly_distribution": [],
                "prediction_breakdown": {
                    "fraud": {"count": 0, "total_amount": 0, "avg_amount": 0},
                    "legit": {"count": 0, "total_amount": 0, "avg_amount": 0}
                }
            }
        
        data = result[0]
        
        # Process overview data
        overview = data["overview"][0] if data["overview"] else {}
        total_transactions = overview.get("total_transactions", 0)
        
        # Process prediction data
        prediction_data = {item["_id"]: item for item in data["by_prediction"]}
        fraud_count = prediction_data.get("Fraud", {}).get("count", 0)
        legit_count = prediction_data.get("Legit", {}).get("count", 0)
        
        # Process recent transactions
        recent_transactions = []
        for doc in data["recent_transactions"]:
            doc["_id"] = str(doc["_id"])
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            recent_transactions.append(doc)
        
        # Calculate fraud rate
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        
        analytics_result = {
            "total_transactions": total_transactions,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_rate_percentage": round(fraud_rate, 2),
            "total_amount": round(overview.get("total_amount", 0), 2),
            "average_amount": round(overview.get("average_amount", 0), 2),
            "max_amount": round(overview.get("max_amount", 0), 2),
            "min_amount": round(overview.get("min_amount", 0), 2),
            "recent_transactions": recent_transactions,
            "hourly_distribution": data["hourly_distribution"],
            "prediction_breakdown": {
                "fraud": {
                    "count": fraud_count,
                    "total_amount": round(prediction_data.get("Fraud", {}).get("total_amount", 0), 2),
                    "avg_amount": round(prediction_data.get("Fraud", {}).get("avg_amount", 0), 2)
                },
                "legit": {
                    "count": legit_count,
                    "total_amount": round(prediction_data.get("Legit", {}).get("total_amount", 0), 2),
                    "avg_amount": round(prediction_data.get("Legit", {}).get("avg_amount", 0), 2)
                }
            }
        }
        
        logger.info("[SUCCESS] Analytics retrieved successfully")
        return analytics_result
        
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error retrieving analytics: {e}")
        raise Exception(f"Unexpected error: {e}")

async def get_user_analytics(db_client: AsyncIOMotorDatabase, email: str) -> Dict[str, Any]:
    """Get analytics for a specific user"""
    try:
        if db_client is None:
            raise Exception("Database client is None - MongoDB connection required")
        
        collection = db_client["transactions"]
        
        user_pipeline = [
            {"$match": {"email": email}},
            {
                "$group": {
                    "_id": None,
                    "total_transactions": {"$sum": 1},
                    "total_amount": {"$sum": "$amount"},
                    "average_amount": {"$avg": "$amount"},
                    "fraud_count": {
                        "$sum": {"$cond": [{"$eq": ["$prediction", "Fraud"]}, 1, 0]}
                    },
                    "legit_count": {
                        "$sum": {"$cond": [{"$eq": ["$prediction", "Legit"]}, 1, 0]}
                    }
                }
            }
        ]
        
        result = await collection.aggregate(user_pipeline).to_list(None)
        
        if not result:
            return {
                "email": email,
                "total_transactions": 0,
                "total_amount": 0,
                "average_amount": 0,
                "fraud_count": 0,
                "legit_count": 0,
                "fraud_rate_percentage": 0
            }
        
        data = result[0]
        fraud_rate = (data["fraud_count"] / data["total_transactions"] * 100) if data["total_transactions"] > 0 else 0
        
        return {
            "email": email,
            "total_transactions": data["total_transactions"],
            "total_amount": round(data["total_amount"], 2),
            "average_amount": round(data["average_amount"], 2),
            "fraud_count": data["fraud_count"],
            "legit_count": data["legit_count"],
            "fraud_rate_percentage": round(fraud_rate, 2)
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Error retrieving user analytics: {e}")
        raise Exception(f"Failed to retrieve user analytics: {e}")