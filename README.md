🛡️ Fraud Detection System – Real-Time AI-Based Solution

### 🚀 Overview

This project is an intelligent, real-time **Fraud Detection System** designed to protect e-commerce platforms and financial institutions from fraudulent transactions using AI, risk scoring, and automated decision-making.

> 🔍 "Think of it as a smart security guard that works 24/7, instantly blocking suspicious transactions and alerting analysts."

---

### 🎯 Key Features

* ⚡ Real-time fraud detection in under 0.1 seconds
* 🧠 AI-powered risk scoring (0–100%)
* 📊 Streamlit dashboard for monitoring transactions
* 📩 Email alerts for high-risk transactions
* 🧾 MongoDB-based persistent storage
* 📈 Self-learning capability from feedback

---

### 🔄 How It Works

1. **Data Collection**
   Captures details like email, amount, account age, time of transaction, etc.

2. **Intelligent Analysis**
   Machine learning model checks for:

   * Suspicious emails
   * Unusual transaction amounts or times
   * New accounts making expensive purchases

3. **Risk Scoring**

   * 0–30% → ✅ Approve
   * 30–70% → ⚠️ Review
   * 70–100% → ⛔ Block

4. **Instant Decision**
   Backend responds within milliseconds and logs data into MongoDB.

5. **Learning & Improvement**
   Continuously learns from past outcomes and feedback.

---

### 🧱 Tech Stack

| Component     | Tech Used                              |
| ------------- | -------------------------------------- |
| Frontend      | Streamlit (Python)                     |
| Backend       | FastAPI                                |
| AI Model      | Logistic Regression (via scikit-learn) |
| Database      | MongoDB (Atlas/local fallback)         |
| Alerts        | SMTP (Email integration)               |
| Hosting Ready | Cloud/Container-ready                  |

---

### 📁 Project Structure

```
fraud-detection-system/
│
├── backend/
│   ├── api.py                 # FastAPI endpoints
│   ├── fraud_detection.py     # Fraud logic and AI prediction
│   ├── model.joblib           # Trained ML model
│   └── scaler.joblib          # Scaler used for input normalization
│
├── frontend/
│   └── app.py                 # Streamlit frontend interface
│
├── data/
│   └── example_transactions.csv  # Sample input
│
├── utils/
│   └── email_alerts.py        # Email notification module
│
├── requirements.txt
└── README.md
```

---

### 🛠️ How to Run Locally

#### 1️⃣ Clone the Repo

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

#### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Start Backend Server

```bash
cd backend
uvicorn api:app --reload
```

Runs at: [http://localhost:8000](http://localhost:8000)

#### 4️⃣ Start Frontend Dashboard

```bash
cd frontend
streamlit run app.py
```

Runs at: [http://localhost:8501](http://localhost:8501)

---

### 📊 Sample Screenshot


<img width="1908" height="912" alt="Screenshot 2025-07-29 234141" src="https://github.com/user-attachments/assets/9dfe0828-771d-46d3-ba1f-80b2439de3ad" />
<img width="1869" height="899" alt="Screenshot 2025-07-29 234302" src="https://github.com/user-attachments/assets/9eb881ec-054b-494d-919b-a003fc204858" />
<img width="1613" height="803" alt="Screenshot 2025-07-29 234547" src="https://github.com/user-attachments/assets/90d2e5aa-7220-49e5-95f3-5fa379cdafc9" />



### ✅ Use Cases

* 🛒 **E-commerce platforms** – Protect against fake accounts, bot transactions
* 💳 **Banks/Fintech** – Detect identity theft and stolen cards
* 🛍️ **Digital marketplaces** – Reduce fraud from suspicious buyers

---

### 💡 Future Improvements

* Integrate advanced ML models like XGBoost or Neural Networks
* Deploy to cloud (AWS, GCP, or Azure)
* Add user authentication for analysts
* Add feedback loop from human analysts to retrain model

### 📜 License

This project is open-source under the **MIT License**.

---
