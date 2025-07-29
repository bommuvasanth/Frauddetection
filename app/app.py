import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import requests
from backend import detect_fraud
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import altair as alt
import time

# Load environment variables from .env file
load_dotenv()

# --- Real-Time Fraud Detection with Kafka and WebSocket ---
# (All code in this section and related to Kafka/WebSocket is removed)


def load_analytics_data(csv_path="transactions.csv"):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('models/model.joblib')

# Load and prepare data
@st.cache_data
def load_data():
    # Removed dependency on large training dataset
    # df = pd.read_csv('data/Fraudulent_E-Commerce_Transaction_Data.csv', nrows=10000)
    # Create sample data for demo purposes
    import numpy as np
    np.random.seed(42)
    sample_size = 1000
    
    sample_data = {
        'Transaction Amount': np.random.uniform(10, 1000, sample_size),
        'Quantity': np.random.randint(1, 10, sample_size),
        'Customer Age': np.random.randint(18, 80, sample_size),
        'Account Age Days': np.random.randint(1, 2000, sample_size),
        'Transaction Hour': np.random.randint(0, 24, sample_size),
        'Is Fraudulent': np.random.choice([0, 1], sample_size, p=[0.9, 0.1])
    }
    df = pd.DataFrame(sample_data)
    
    numeric_features = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
    X = df[numeric_features]
    y = df['Is Fraudulent']
    return X, y, df

# Load model and data
model = load_model()
X, y, df = load_data()

# Debug: Show DataFrame info
if st.checkbox("🔍 Debug: Show DataFrame Info"):
    st.write("DataFrame shape:", df.shape)
    st.write("DataFrame columns:", df.columns.tolist())
    st.write("DataFrame head:")
    st.dataframe(df.astype(str))
    st.info("📝 Note: This is sample data for demonstration. The large training dataset has been removed to reduce project size.")

# Calculate model performance
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division='warn')
recall = recall_score(y, y_pred, zero_division='warn')
f1 = f1_score(y, y_pred, zero_division='warn')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection in E-Commerce Transactions")
st.header("Enter Transaction Details Manually")

if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

def gemini_analyze_transaction(email, transaction_amount, account_age, quantity, transaction_hour, customer_age, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    message = (
        f"Analyze this transaction:\n"
        f"Email: {email}\n"
        f"Transaction Amount: {transaction_amount}\n"
        f"Account Age: {account_age} days\n"
        f"Quantity: {quantity}\n"
        f"Transaction Hour: {transaction_hour}\n"
        f"Customer Age: {customer_age}"
    )
    payload = {
        "contents": [{"parts": [{"text": message}]}],
        "generationConfig": {
            "maxOutputTokens": 100,
            "temperature": 0.7,
            "topP": 0.8
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            comment = result["candidates"][0]["content"]["parts"][0]["text"]
            return comment
        else:
            return f"Error: {response.status_code} - {response.reason}"
    except Exception as e:
        return f"Error: {e}"

# Backend API key configuration
# The API key should be set as an environment variable: GEMINI_API_KEY
# Or you can set it here for testing (not recommended for production)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get from environment variable

if not GEMINI_API_KEY:
    st.warning("⚠️ No Gemini API key found. Using fallback fraud detection.")
    st.info("To enable Gemini AI analysis, set the GEMINI_API_KEY environment variable.")
else:
    st.success("✅ Gemini API key configured. AI-powered fraud detection enabled.")

# Fraud Detection File Uploader
st.header("🔍 Batch Fraud Detection")
uploaded_file = st.file_uploader("Upload Excel file with transactions", type=["xlsx"], 
                                help="File should contain: Email Address, Transaction Amount, Account Age")

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        # Convert to string to avoid Arrow serialization issues
        display_df = df.head().astype(str)
        st.dataframe(display_df.astype(str), use_container_width=True)
        
        # Check if required columns exist
        required_columns = ['Email Address', 'Transaction Amount', 'Account Age']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Please ensure your Excel file contains: Email Address, Transaction Amount, Account Age")
        else:
            st.success(f"✅ Found {len(df)} transactions to analyze")
            
            # Use the configured API key
            if not GEMINI_API_KEY:
                st.warning("⚠️ No API key configured. Using fallback fraud detection.")
            else:
                st.info("🔑 Using configured Gemini API key for AI-powered analysis.")
                
                if st.button("🚀 Detect Fraud for All Transactions"):
                    with st.spinner("🔍 Analyzing transactions with Gemini AI..."):
                        # Add progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize prediction column
                        df['Prediction'] = ''
                        
                        # Process each transaction
                        for idx, row in df.iterrows():
                            try:
                                # Extract transaction details
                                email = str(row['Email Address'])
                                amount = float(row['Transaction Amount'])
                                account_age = int(row['Account Age'])
                                
                                # Call the secure backend function
                                prediction = detect_fraud(email, amount, account_age, GEMINI_API_KEY)
                                df.at[idx, 'Prediction'] = prediction
                                
                                # Update progress
                                progress = (idx + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"Processed {idx + 1}/{len(df)} transactions...")
                                
                            except Exception as e:
                                st.error(f"Error processing row {idx + 1}: {str(e)}")
                                df.at[idx, 'Prediction'] = 'Error'
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    # Show results
                    st.success("✅ Fraud detection completed!")
                    
                    # Display results with fraud highlighted
                    st.write("📋 Transaction Results:")
                    fraud_df = df[df['Prediction'] == 'Fraud']
                    legit_df = df[df['Prediction'] == 'Legit']
                    
                    if len(fraud_df) > 0:
                        st.warning(f"🚨 Found {len(fraud_df)} potentially fraudulent transactions:")
                        # Convert to string to avoid Arrow serialization issues
                        display_fraud = fraud_df.astype(str)
                        st.dataframe(display_fraud.astype(str), use_container_width=True)
                    
                    if len(legit_df) > 0:
                        st.success(f"✅ {len(legit_df)} legitimate transactions:")
                        # Convert to string to avoid Arrow serialization issues
                        display_legit = legit_df.head().astype(str)
                        st.dataframe(display_legit.astype(str), use_container_width=True)
                    
                    # Generate summary by email address
                    st.header("📊 Summary by Email Address")
                    
                    # Group by email and calculate statistics
                    summary_data = []
                    for email in df['Email Address'].unique():
                        email_df = df[df['Email Address'] == email]
                        total_transactions = len(email_df)
                        fraud_count = len(email_df[email_df['Prediction'] == 'Fraud'])
                        total_amount = email_df['Transaction Amount'].sum()
                        fraud_loss = email_df[email_df['Prediction'] == 'Fraud']['Transaction Amount'].sum()
                        
                        summary_data.append({
                            'Email Address': email,
                            'Total_Transactions': total_transactions,
                            'Fraud_Count': fraud_count,
                            'Total_Amount_₹': f"₹{total_amount:,.2f}",
                            'Fraud_Loss_₹': f"₹{fraud_loss:,.2f}"
                        })
                    
                    summary = pd.DataFrame(summary_data)
                    summary_display = summary.set_index('Email Address')
                    
                    # Convert to string to avoid Arrow serialization issues
                    display_summary = summary_display.astype(str)
                    st.dataframe(display_summary.astype(str), use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='fraud_detection_results.csv',
                        mime='text/csv'
                    )
                    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please ensure your file is a valid Excel (.xlsx) file with the required columns.")

st.divider()

# Manual entry form (always visible)
with st.form("manual_entry"):
    st.caption("Or enter a single transaction manually below:")
    email = st.text_input("Email Address")
    amount = st.number_input("Transaction Amount", min_value=0.0)
    account_age = st.number_input("Account Age (Days)", min_value=0)
    quantity = st.number_input("Quantity", min_value=1)
    trans_hour = st.number_input("Transaction Hour", min_value=0, max_value=23)
    cust_age = st.number_input("Customer Age", min_value=0, max_value=120)
    submitted = st.form_submit_button("Submit Transaction")

if submitted:
    data = {
        "email": email,
        "Transaction_Amount": amount,
        "Quantity": quantity,
        "Customer_Age": cust_age,
        "Account_Age_Days": account_age,
        "Transaction_Hour": trans_hour
    }
    try:
        response = requests.post("http://127.0.0.1:8000/submit_transaction", json=data)
        if response.status_code == 200:
            try:
                result = response.json()
                st.write("Prediction:", result.get("prediction"))
                if result.get("email_sent"):
                    st.success(f"✅ Report sent to {email}!")
                else:
                    st.error(f"❌ Failed to send email: {result.get('email_error')}")
            except Exception as e:
                st.error(f"Error parsing backend response: {e}")
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

def send_transaction_report(email, transaction_details):
    # Configure your SMTP server and credentials
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "your_email@gmail.com"
    smtp_password = "your_app_password"  # Use an app password, not your main password

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = email
    msg["Subject"] = "Transaction Report"
    body = f"Transaction Details:\n{transaction_details}"
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email, msg.as_string())
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# All code and sections related to 'Real-Time Fraud Detection Dashboard' have been removed.


# Define input schema
class TransactionInput(BaseModel):
    Transaction_Amount: float
    Quantity: int
    Customer_Age: int
    Account_Age_Days: int
    Transaction_Hour: int

# Load scaler and model with error handling
scaler = None
model = None
is_autoencoder = False

try:
    if os.path.exists("models/scaler.joblib"):
        scaler = joblib.load("models/scaler.joblib")
    else:
        st.warning("⚠️ Scaler file not found. Please run fraud_advanced_models.py to generate model files.")
        
    # Try to load the best model (joblib or keras)
    if os.path.exists("models/best_model.joblib"):
        
        model = joblib.load("models/best_model.joblib")
    elif os.path.exists("models/best_autoencoder.h5"):
        try:
            from tensorflow import keras
            model = keras.models.load_model("models/best_autoencoder.h5")
            is_autoencoder = True
        except ImportError:
            st.warning("⚠️ TensorFlow not installed. Install with: pip install tensorflow")
    else:
        st.warning("⚠️ Model file not found. Please run fraud_advanced_models.py to generate model files.")
        
except Exception as e:
    st.error(f"❌ Error loading model files: {str(e)}")
    st.info("Please ensure model files exist in the models/ directory.")

# The following FastAPI endpoint is no longer used as WebSocket/Kafka is removed
# @app.post("/predict")
# def predict(input: TransactionInput):
#     if scaler is None or model is None:
#         return {"error": "Models not loaded. Please run fraud_advanced_models.py first"}
    
#     try:
#         features = np.array([
#             [
#                 input.Transaction_Amount,
#                 input.Quantity,
#                 input.Customer_Age,
#                 input.Account_Age_Days,
#                 input.Transaction_Hour
#             ]
#         ])
#         features_scaled = scaler.transform(features)

#         if is_autoencoder:
#             recon = model.predict(features_scaled)
#             mse = np.mean(np.square(features_scaled - recon), axis=1)
#             threshold = 0.5  # Example threshold
#             is_fraud = int(mse[0] > threshold)
#             confidence = float(mse[0])
#             return {"fraud": is_fraud, "confidence": confidence, "model": "autoencoder"}
#         else:
#             pred = model.predict(features_scaled)[0]
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba(features_scaled)[0][1]
#             else:
#                 proba = None
#             return {"fraud": int(pred), "probability": proba, "model": "sklearn"}
#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}

# To run: uvicorn api:app --reload

# Data overview
st.header("Dataset Overview")

# Check if we have the original dataset with 'Is Fraudulent' column
if 'Is Fraudulent' in df.columns:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud vs Non-Fraud Distribution")
        fraud_counts = df['Is Fraudulent'].value_counts()
        st.bar_chart(fraud_counts)

    with col2:
        st.subheader("Transaction Amount Statistics")
        fraud_amounts = df[df['Is Fraudulent'] == 1]['Transaction Amount']
        non_fraud_amounts = df[df['Is Fraudulent'] == 0]['Transaction Amount']
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean Amount', 'Count'],
            'Fraudulent': [f"${fraud_amounts.mean():.2f}", len(fraud_amounts)],
            'Non-Fraudulent': [f"${non_fraud_amounts.mean():.2f}", len(non_fraud_amounts)]
        })
        st.dataframe(stats_df.astype(str), use_container_width=True)
else:
    st.info("📊 Dataset overview available for original training data. Uploaded files will show prediction results instead.")
    
    # Show basic statistics for uploaded data
    if 'Transaction Amount' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            # Create histogram data
            import numpy as np
            hist_values, bin_edges = np.histogram(df['Transaction Amount'], bins=20)
            hist_df = pd.DataFrame({
                'Amount Range': [f"${bin_edges[i]:.0f}-${bin_edges[i+1]:.0f}" for i in range(len(bin_edges)-1)],
                'Count': hist_values
            })
            st.bar_chart(hist_df.set_index('Amount Range'))
            
        with col2:
            st.subheader("Basic Statistics")
            stats = df['Transaction Amount'].describe()
            st.dataframe(stats.to_frame().astype(str), use_container_width=True)

# Feature importance if available
if model is not None and hasattr(model, 'feature_importances_'):
    st.header("Feature Importance")
    feature_names = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))
else:
    st.info("🔍 Feature importance available when using trained machine learning models.")

st.divider()

st.header("Generate Report by Email Address")

# Use the most recent DataFrame (from upload or manual entry)
report_df = None
if 'df' in locals():
    report_df = df.copy()
elif 'single_df' in locals():
    report_df = single_df.copy()

if report_df is not None and 'Email Address' in report_df.columns:
    # If 'Fraud' column not present, simulate for demo
    if 'Fraud' not in report_df.columns:
        import numpy as np
        np.random.seed(42)
        report_df['Fraud'] = np.random.choice([0, 1], size=len(report_df), p=[0.95, 0.05])
    # If 'Gemini_Analysis' not present, fill with empty string
    if 'Gemini_Analysis' not in report_df.columns:
        report_df['Gemini_Analysis'] = ""
    # Group and summarize
    summary = report_df.groupby('Email Address').agg(
        num_transactions=('Email Address', 'count'),
        num_frauds=('Fraud', 'sum'),
        gemini_comments=('Gemini_Analysis', lambda x: ' | '.join(x.dropna().astype(str))),
        total_loss=('Transaction Amount', lambda x: x[report_df.loc[x.index, 'Fraud'] == 1].sum())
    ).reset_index()
    st.write("### Report Table", summary)
    csv = summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report as CSV",
        data=csv,
        file_name='fraud_report_by_email.csv',
        mime='text/csv'
    )
else:
    st.info("No data available to generate report. Please upload data or enter a transaction.")

# Patch st.dataframe to always convert to string
_original_dataframe = st.dataframe
def safe_dataframe(df, *args, **kwargs):
    if isinstance(df, pd.DataFrame):
        return _original_dataframe(df.astype(str), *args, **kwargs)
    return _original_dataframe(df, *args, **kwargs)
st.dataframe = safe_dataframe

# Poll the backend every 2 seconds
REFRESH_INTERVAL = 2  # seconds

while True:
    try:
        # response = requests.get("http://localhost:8000/latest-predictions") # Removed: No backend
        # if response.status_code == 200: # Removed: No backend
        #     predictions = response.json() # Removed: No backend
        #     st.write(f"Last {len(predictions)} transactions:") # Removed: No backend
        #     st.table(predictions) # Removed: No backend
        # else: # Removed: No backend
        #     st.warning("Could not fetch predictions.") # Removed: No backend
        pass # Removed: No backend
    except Exception as e:
        st.error(f"Error: {e}")
    time.sleep(REFRESH_INTERVAL)
# Remove or comment out the following line to fix Streamlit error
# st.experimental_rerun()