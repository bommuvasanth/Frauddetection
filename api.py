from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from backend.database import connect_to_mongo, close_mongo_connection # Removed: , db
import backend.database as database # Import the module to access its attributes
from backend.config import settings
from backend.models import TransactionCreate, TransactionInDB #, TransactionInput as OriginalTransactionInput
import numpy as np
import joblib
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import sys
from email.mime.multipart import MIMEMultipart
import datetime
import logging
from backend.crud import insert_transaction, get_analytics, get_user_analytics
from backend.fraud_detection import detect_fraud, predict_fraud, map_fields
from pydantic import BaseModel

print(">>> api.py loaded")
import sys
sys.stdout.flush()

load_dotenv()

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    try:
        print(f"[INFO] Starting Fraud Detection API...")
        print(f"[INFO] Attempting MongoDB connection...")
        print(f"[INFO] Database name: {settings.MONGODB_DB_NAME}")

        await connect_to_mongo()
        
        # Check connection status
        if database.is_connected():
            print("[SUCCESS] MongoDB connection established successfully!")
        elif database.is_fallback_mode():
            print("[WARNING] Running in fallback mode - using in-memory storage")
            print("[INFO] Application functionality will be limited but operational")
        else:
            print("[ERROR] Database connection status unknown")
            
    except Exception as e:
        print(f"[ERROR] Startup error: {e}")
        print("[INFO] Application will continue in fallback mode")

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()
    print("MongoDB connection closed for FastAPI app.")

# Field aliases and defaults are now defined in backend.fraud_detection module

# map_fields function is now imported from backend.fraud_detection

# Model loading and predict_fraud function are now in backend.fraud_detection

# detect_fraud function is now imported from backend.fraud_detection

# Define input schema - This model will match the incoming JSON payload from the client.
class TransactionInput(BaseModel):
    email: str
    Transaction_Amount: float # Matches incoming payload
    Quantity: int             # Matches incoming payload
    Customer_Age: int         # Matches incoming payload
    Account_Age_Days: int     # Matches incoming payload
    Transaction_Hour: int     # Matches incoming payload

# Model loading is now handled in backend.fraud_detection module

async def send_email_report(to_email, amount, quantity, customer_age, account_age, trans_hour, prediction):
    """
    Send fraud detection email report using Gmail SMTP
    """
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Validate email credentials
    if not smtp_user or not smtp_password:
        print("[ERROR] Email credentials not found in environment variables")
        return False, "Email credentials not configured"

    subject = f"🛡️ Fraud Detection Alert - {'⚠️ FRAUD DETECTED' if prediction.lower() == 'fraud' else '✅ Transaction Verified'}"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Fetch analytics from MongoDB
        analytics = await get_analytics(database.db)
        print(f"[INFO] Fetched analytics for email report")
    except Exception as e:
        print(f"[WARNING] Could not fetch analytics: {e}")
        analytics = {}

    analytics_html = ""                                 
    if analytics and analytics.get('total_transactions', 0) > 0:
        recent_transactions_html = "".join(
            f"<tr><td>{row.get('timestamp', 'N/A')}</td><td>${row.get('amount', 'N/A')}</td><td>{row.get('prediction', 'N/A')}</td></tr>"
            for row in analytics.get('recent_transactions', [])[:5]  # Show only last 5 transactions
        )

        analytics_html = f"""
        <h3>📊 Account Analytics Summary</h3>
        <ul>
          <li>Total transactions: {analytics['total_transactions']}</li>
          <li>Fraudulent: {analytics['fraud_count']} ({analytics.get('fraud_rate_percentage', 0):.1f}%)</li>
          <li>Legitimate: {analytics['legit_count']}</li>
          <li>Average Amount: ${analytics['average_amount']:.2f}</li>
        </ul>
        <h4>Recent Transactions:</h4>
        <table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>
        <tr style='background-color: #f0f0f0;'><th>Date</th><th>Amount</th><th>Status</th></tr>
        {recent_transactions_html}
        </table>
        """
    else:
        analytics_html = "<p><i>No previous transaction data available.</i></p>"

    # Enhanced HTML email template
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: {'#ffebee' if prediction.lower() == 'fraud' else '#e8f5e8'}; padding: 15px; border-radius: 5px; }}
            .transaction-details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .fraud-alert {{ color: #d32f2f; font-weight: bold; font-size: 18px; }}
            .safe-alert {{ color: #388e3c; font-weight: bold; font-size: 18px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🛡️ Fraud Detection Report</h2>
            <p class="{'fraud-alert' if prediction.lower() == 'fraud' else 'safe-alert'}">
                {'⚠️ POTENTIAL FRAUD DETECTED!' if prediction.lower() == 'fraud' else '✅ Transaction Verified as Legitimate'}
            </p>
        </div>
        
        <div class="transaction-details">
            <h3>Transaction Details</h3>
            <table>
                <tr><th>Email Address</th><td>{to_email}</td></tr>
                <tr><th>Transaction Amount</th><td>${amount:.2f}</td></tr>
                <tr><th>Quantity</th><td>{quantity}</td></tr>
                <tr><th>Customer Age</th><td>{customer_age} years</td></tr>
                <tr><th>Account Age</th><td>{account_age} days</td></tr>
                <tr><th>Transaction Hour</th><td>{trans_hour}:00</td></tr>
                <tr><th>AI Prediction</th><td><b>{prediction}</b></td></tr>
                <tr><th>Report Generated</th><td>{now}</td></tr>
            </table>
        </div>
        
        {analytics_html}
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;">
            <p><small>This is an automated fraud detection report. If you believe this assessment is incorrect, please contact our support team.</small></p>
            <p><small>Generated by AI Fraud Detection System v2.0</small></p>
        </div>
    </body>
    </html>
    """

    # Create email message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Fraud Detection System <{smtp_user}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        print(f"[INFO] Attempting to send email to {to_email}")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        
        print(f"[SUCCESS] Email sent successfully to {to_email}")
        return True, None
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Gmail authentication failed. Check your app password: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg
    except smtplib.SMTPException as e:
        error_msg = f"SMTP error occurred: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error sending email: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg

@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()
    logging.info(f"Original input: {input_data}")
    processed_data = map_fields(input_data)
    logging.info(f"Processed data: {processed_data}")
    result = predict_fraud(processed_data)
    return JSONResponse({
        "fraudulent": bool(result),
        "processed_data": processed_data
    })

@app.post("/submit_transaction")
async def submit_transaction(request: TransactionInput):
    print(">>> /submit_transaction endpoint called")
    sys.stdout.flush()
    try:
        prediction_value = detect_fraud(
            request.email,
            request.Transaction_Amount,
            request.Account_Age_Days
        )
        print(f"Model Prediction: {prediction_value}") # Debugging: print prediction

        transaction_data = TransactionInDB(
            email=request.email,
            amount=request.Transaction_Amount,
            quantity=request.Quantity,
            customer_age=request.Customer_Age,
            account_age=request.Account_Age_Days,
            transaction_hour=request.Transaction_Hour,
            prediction=prediction_value
        )

        await insert_transaction(database.db, transaction_data.model_dump())
        print("Transaction inserted into MongoDB.")
        sys.stdout.flush()

        # Send email report
        email_success, email_error = await send_email_report(
            request.email,
            request.Transaction_Amount,
            request.Quantity,
            request.Customer_Age,
            request.Account_Age_Days,
            request.Transaction_Hour,
            prediction_value
        )
        print("Email sent:", email_success, email_error)
        sys.stdout.flush()
        return {
            "email": request.email,
            "Transaction_Amount": request.Transaction_Amount,
            "Account_Age_Days": request.Account_Age_Days,
            "prediction": prediction_value,
            "email_sent": email_success,
            "email_error": email_error if not email_success else None
        }
    except Exception as e:
        import traceback
        print("Error in submit_transaction:", str(e))
        traceback.print_exc() # Print full traceback to console
        sys.stdout.flush()
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"}) # Return error to user

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health_check():
    # Check if models are loaded in the backend module
    try:
        from backend.fraud_detection import model, scaler
        models_loaded = model is not None
    except (ImportError, AttributeError):
        models_loaded = False
    
    db_status = database.get_connection_status()
    db_connected = database.is_connected()
    
    status = "healthy" if models_loaded and db_connected else "unhealthy"
    
    return {
        "status": status,
        "models_loaded": models_loaded,
        "database_status": db_status,
        "database_connected": db_connected,
        "mongodb_type": "local",
        "message": "API is operational" if status == "healthy" else "Some components may not be working"
    }

@app.get("/analytics")
async def get_all_analytics():
    """Fetches overall fraud detection analytics from MongoDB."""
    analytics_data = await get_analytics(database.db) # Access via module
    return JSONResponse(analytics_data)

@app.get("/analytics/{email}")
async def get_user_analytics_endpoint(email: str):
    """Fetches analytics for a specific user from MongoDB."""
    try:
        user_analytics = await get_user_analytics(database.db, email)
        return JSONResponse(user_analytics)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error retrieving user analytics: {str(e)}"})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print("UNHANDLED EXCEPTION:", exc)
    traceback.print_exc()
    return PlainTextResponse(str(exc), status_code=500)

# To run: uvicorn api:app --reload 