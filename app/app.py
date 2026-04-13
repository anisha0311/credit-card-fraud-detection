from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import os

from src.predict import predict_transaction

app = FastAPI(title="Fraud Detection API")

# ==============================
# FILE PATH
# ==============================
STATE_FILE = "data/user_state.json"

# ==============================
# LOAD / SAVE FUNCTIONS
# ==============================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ==============================
# LOAD STATE (IMPORTANT)
# ==============================
user_state = load_state()


# ==============================
# INPUT SCHEMAS
# ==============================
class InitUser(BaseModel):
    user_id: str
    balance: Optional[float] = None


class Transaction(BaseModel):
    user_id: str
    type: str
    amount: float
    nameDest: Optional[str] = "M1"
    oldbalanceDest: Optional[float] = 0


# ==============================
# ROOT
# ==============================
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running 🚀"}


# ==============================
# INIT USER (SMART FLOW)
# ==============================
@app.post("/init_user")
def init_user(data: InitUser):

    # -----------------------------
    # EXISTING USER
    # -----------------------------
    if data.user_id in user_state:
        return {
            "status": "existing_user",
            "message": f"Welcome back {data.user_id}",
            "state": user_state[data.user_id]
        }

    # -----------------------------
    # NEW USER → NEED BALANCE
    # -----------------------------
    if data.balance is None:
        return {
            "status": "new_user",
            "message": "User not found. Please provide initial balance."
        }

    # -----------------------------
    # CREATE USER
    # -----------------------------
    user_state[data.user_id] = {
        "balance": data.balance,
        "step": 1
    }

    save_state(user_state)

    return {
        "status": "created",
        "message": f"User {data.user_id} created",
        "state": user_state[data.user_id]
    }


# ==============================
# PREDICT TRANSACTION
# ==============================
@app.post("/predict")
def predict(txn: Transaction):

    # -----------------------------
    # CHECK USER
    # -----------------------------
    if txn.user_id not in user_state:
        return {"error": "User not initialized"}

    state = user_state[txn.user_id]

    # -----------------------------
    # BUILD FULL TRANSACTION
    # -----------------------------
    full_txn = {
        "step": state["step"],
        "type": txn.type,
        "amount": txn.amount,
        "nameOrig": txn.user_id,
        "nameDest": txn.nameDest,
        "oldbalanceOrg": state["balance"],
        "oldbalanceDest": txn.oldbalanceDest
    }

    # -----------------------------
    # CALL MODEL
    # -----------------------------
    result = predict_transaction(full_txn)

    is_fraud = result["final_prediction"] == 1

    # -----------------------------
    # UPDATE BALANCE (BLOCK FRAUD)
    # -----------------------------
    if not is_fraud:
        if txn.type in ["PAYMENT", "DEBIT", "TRANSFER"]:
            if txn.amount <= state["balance"]:
                state["balance"] -= txn.amount
            else:
                state["balance"] = 0  # safety
    else:
        print(f"🚨 Fraud detected for {txn.user_id}: {txn.amount}")

    # -----------------------------
    # UPDATE STEP
    # -----------------------------
    state["step"] += 1

    # -----------------------------
    # SAVE STATE (CRITICAL)
    # -----------------------------
    save_state(user_state)

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "transaction": full_txn,
        "result": result,
        "fraud_detected": is_fraud,
        "updated_balance": state["balance"]
    }