"""
Fraud Detection Module
Contains the core fraud detection logic and prediction functions.
"""

import numpy as np
import joblib
import os
from typing import Dict, Any

# Load model and scaler
try:
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    print("[SUCCESS] Fraud detection models loaded successfully")
except FileNotFoundError as e:
    print(f"[ERROR] Model files not found: {e}")
    print("[INFO] Please run fraud_advanced_models.py first to generate models")
    model = None
    scaler = None

# Field aliases for mapping different input formats
FIELD_ALIASES = {
    "TransactionAmount": ["TransactionAmount", "txn_amt", "amount", "transaction_amount"],
    "Quantity": ["Quantity", "qty", "quantity"],
    "CustomerAge": ["CustomerAge", "cust_age", "customer_age", "age"],
    "PaymentMethod": ["PaymentMethod", "method", "payment_method", "pay_method"],
    "AccountAge": ["Account_Age_Days", "account_age", "accountagedays"]
}

REQUIRED_SCHEMA = ["TransactionAmount", "Quantity", "CustomerAge", "PaymentMethod", "AccountAge"]

DEFAULTS = {
    "TransactionAmount": 0,
    "Quantity": 1,
    "CustomerAge": 0,
    "PaymentMethod": "unknown",
    "AccountAge": 0
}

def map_fields(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map input fields to the required schema using aliases.
    
    Args:
        input_data: Dictionary containing input transaction data
        
    Returns:
        Dictionary with mapped fields according to REQUIRED_SCHEMA
    """
    processed = {}
    for key in REQUIRED_SCHEMA:
        found = False
        for alias in FIELD_ALIASES[key]:
            if alias in input_data:
                processed[key] = input_data[alias]
                found = True
                break
        if not found:
            processed[key] = DEFAULTS[key]
    return processed

def predict_fraud(processed_data: Dict[str, Any]) -> bool:
    """
    Predict if a transaction is fraudulent using the trained model.
    
    Args:
        processed_data: Dictionary containing processed transaction features
        
    Returns:
        Boolean indicating if transaction is fraudulent (True) or legitimate (False)
    """
    if model is None:
        # Fallback prediction logic when model is not available
        print("[WARNING] Model not available, using fallback logic")
        # Simple rule-based fallback
        amount = processed_data.get("TransactionAmount", 0)
        account_age = processed_data.get("AccountAge", 0)
        
        # Basic heuristic: high amount + new account = potential fraud
        if amount > 1000 and account_age < 30:
            return True
        if amount > 5000:
            return True
        return False
    
    try:
        # Ensure the order matches REQUIRED_SCHEMA
        features = [
            processed_data["TransactionAmount"],
            processed_data["Quantity"],
            processed_data["CustomerAge"],
            processed_data["PaymentMethod"] if isinstance(processed_data["PaymentMethod"], (int, float)) else 0,
            processed_data["AccountAge"]
        ]
        
        X = np.array([features])
        pred = model.predict(X)[0]
        return bool(pred)
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        # Fallback to simple heuristic
        amount = processed_data.get("TransactionAmount", 0)
        return amount > 2000  # Simple threshold-based fallback

def detect_fraud(email: str, transaction_amount: float, account_age_days: int) -> str:
    """
    Main fraud detection function - simplified wrapper for compatibility.
    
    Args:
        email: Customer email address
        transaction_amount: Transaction amount in currency units
        account_age_days: Age of customer account in days
        
    Returns:
        String indicating "Fraud" or "Legit"
    """
    # Map inputs to the expected format
    processed_data = {
        "TransactionAmount": transaction_amount,
        "Quantity": 1,  # Default quantity
        "CustomerAge": 0,  # Default customer age (not provided in this interface)
        "PaymentMethod": 0,  # Default payment method as numeric
        "AccountAge": account_age_days
    }
    
    # Get prediction
    is_fraudulent = predict_fraud(processed_data)
    
    return "Fraud" if is_fraudulent else "Legit"

def detect_fraud_advanced(
    email: str,
    transaction_amount: float,
    quantity: int = 1,
    customer_age: int = 0,
    account_age_days: int = 0,
    transaction_hour: int = 12,
    payment_method: str = "unknown"
) -> Dict[str, Any]:
    """
    Advanced fraud detection with more detailed input and output.
    
    Args:
        email: Customer email address
        transaction_amount: Transaction amount
        quantity: Number of items
        customer_age: Customer age
        account_age_days: Account age in days
        transaction_hour: Hour of transaction (0-23)
        payment_method: Payment method used
        
    Returns:
        Dictionary containing prediction result and confidence metrics
    """
    processed_data = {
        "TransactionAmount": transaction_amount,
        "Quantity": quantity,
        "CustomerAge": customer_age,
        "PaymentMethod": 0 if payment_method == "unknown" else hash(payment_method) % 10,
        "AccountAge": account_age_days
    }
    
    is_fraudulent = predict_fraud(processed_data)
    
    # Calculate risk score (simplified)
    risk_factors = []
    risk_score = 0
    
    if transaction_amount > 1000:
        risk_factors.append("High transaction amount")
        risk_score += 0.3
    
    if account_age_days < 30:
        risk_factors.append("New account")
        risk_score += 0.2
    
    if transaction_hour < 6 or transaction_hour > 22:
        risk_factors.append("Unusual transaction time")
        risk_score += 0.1
    
    if quantity > 10:
        risk_factors.append("High quantity")
        risk_score += 0.1
    
    return {
        "email": email,
        "prediction": "Fraud" if is_fraudulent else "Legit",
        "is_fraudulent": is_fraudulent,
        "risk_score": min(risk_score, 1.0),
        "risk_factors": risk_factors,
        "transaction_amount": transaction_amount,
        "account_age_days": account_age_days
    }
