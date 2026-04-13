# Credit Card Fraud Detection System

## Overview

This project implements a full-stack machine learning system for detecting fraudulent financial transactions using the PaySim dataset. The system combines a global batch-trained model with user-specific online learning models to simulate real-world fraud detection scenarios.

The application provides:

* Real-time fraud prediction via a FastAPI backend
* Interactive user interface using Streamlit
* Continuous learning through user-specific models
* Persistent user state management

---

## Features

* Global fraud detection model trained on imbalanced transaction data
* User-specific adaptive models using online learning
* Transaction simulation with balance tracking
* Fraud prevention mechanism (blocks suspicious transactions)
* Persistent storage of user models and account state
* Configurable system using YAML configuration
* Containerized deployment using Docker

---

## Project Structure

```
Credit_Card_Fraud/
│
├── app/
│   ├── app.py              # FastAPI backend
│   ├── frontend.py         # Streamlit UI
│   └── schema.py
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── predict.py
│   ├── utils.py
│
├── models/
│   ├── model_v1.pkl        # Global trained model
│   └── users/              # User-specific models
│
├── data/
│   └── user_state.json
│
├── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## System Architecture

1. Input transaction is received via API or UI
2. Data is preprocessed and transformed into features
3. Global model generates fraud probability
4. User-specific model contributes additional prediction
5. Final decision is made based on combined output
6. Online learning updates user model periodically
7. User state (balance, step) is persisted

---

## Models

### 1. Global Model

The global model is trained on the PaySim dataset to capture general fraud patterns.

* Algorithm: XGBoost (primary) or Logistic Regression (baseline)
* Handles class imbalance using techniques such as:

  * Class weighting
  * Scale adjustment for minority class
* Learns general transaction patterns such as:

  * Transaction amount behavior
  * Transaction types (TRANSFER, PAYMENT, DEBIT)
  * Balance-based anomalies
* Outputs:

  * Fraud probability (`predict_proba`)
  * Binary classification using a tuned threshold

---

### 2. User-Specific Model

Each user is assigned an individual model that adapts over time.

* Algorithm: SGDClassifier (logistic loss)
* Initialized per user and stored in `models/users/`
* Training strategy:

  * First 10 transactions use only global model
  * After that, user model contributes to prediction
  * Model is updated every 3 transactions using buffered data
* Purpose:

  * Capture personalized behavior patterns
  * Improve detection of unusual activity specific to a user

---

### 3. Hybrid Decision Strategy

The final prediction combines both models:

* If either global or user model predicts fraud → transaction is flagged
* This improves recall and ensures suspicious activity is not missed
* Probability threshold tuning is used to balance precision and recall

---

### 4. Online Learning Mechanism

* Transactions are buffered per user
* Once buffer size is reached:

  * Model is updated using `partial_fit`
* Supports:

  * Continuous adaptation
  * Real-time learning from new patterns

---

## Running the Application (Without Docker)

### 1. Clone the Repository

```
git clone <repository-url>
cd Credit_Card_Fraud
```

### 2. Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Start Backend (FastAPI)

```
uvicorn app.app:app --reload
```

API Documentation:
http://127.0.0.1:8000/docs

### 5. Start Frontend (Streamlit)

```
streamlit run app/frontend.py
```

Frontend UI:
http://127.0.0.1:8501

---

## Running with Docker

### 1. Build Docker Image

```
docker build -t fraud-detection-app .
```

### 2. Run Container

```
docker run -p 8000:8000 -p 8501:8501 fraud-detection-app
```

### Access Services

* FastAPI Backend: http://localhost:8000/docs
* Streamlit Frontend: http://localhost:8501

---

## Configuration

System parameters are defined in `config.yaml`, including:

* Model thresholds
* Online learning settings
* Fraud detection rules
* File paths
* API configuration

This enables easy tuning without modifying core logic.

---

## Dataset

The system is based on the PaySim simulated financial transaction dataset.

A sample dataset is included: 

---

## Key Design Decisions

* Combination of batch learning and online learning
* Handling extreme class imbalance in fraud detection
* Use of probabilistic thresholds instead of fixed predictions
* Persistent user modeling for personalization
* Separation of backend, ML pipeline, and frontend

---

## Future Improvements

* Integration with a database (PostgreSQL or MongoDB)
* Real-time streaming using Kafka
* Advanced anomaly detection techniques
* Model monitoring and drift detection
* Cloud deployment (AWS, GCP, Azure)

---

## Conclusion

This project demonstrates a scalable and realistic fraud detection system by combining machine learning, adaptive modeling, and deployment practices. It reflects real-world systems where both global trends and individual user behavior are essential for accurate detection.
