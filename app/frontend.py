import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection System")

# -----------------------------
# SESSION STATE
# -----------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "awaiting_balance" not in st.session_state:
    st.session_state.awaiting_balance = False

if "temp_user" not in st.session_state:
    st.session_state.temp_user = None


# =============================
# STEP 1: ENTER USERNAME
# =============================
if st.session_state.user_id is None and not st.session_state.awaiting_balance:

    user_id = st.text_input("Enter User ID")

    if st.button("Continue"):

        response = requests.post(
            f"{BASE_URL}/init_user",
            json={"user_id": user_id}
        )

        data = response.json()

        if data["status"] == "existing_user":
            st.session_state.user_id = user_id
            st.success(data["message"])
            st.rerun()

        elif data["status"] == "new_user":
            st.session_state.awaiting_balance = True
            st.session_state.temp_user = user_id
            st.rerun()


# =============================
# STEP 2: ASK BALANCE (ONLY NEW USER)
# =============================
elif st.session_state.awaiting_balance:

    st.warning(f"New user: {st.session_state.temp_user}")

    balance = st.number_input("Enter Initial Balance", min_value=0.0)

    if st.button("Create User"):

        response = requests.post(
            f"{BASE_URL}/init_user",
            json={
                "user_id": st.session_state.temp_user,
                "balance": balance
            }
        )

        data = response.json()

        st.session_state.user_id = st.session_state.temp_user
        st.session_state.awaiting_balance = False

        st.success(data["message"])
        st.rerun()


# =============================
# STEP 3: TRANSACTION UI
# =============================
elif st.session_state.user_id:

    st.success(f"Active User: {st.session_state.user_id}")

    txn_type = st.selectbox("Transaction Type", ["PAYMENT", "DEBIT", "TRANSFER"])
    amount = st.number_input("Amount", min_value=0.0)
    nameDest = st.text_input("Destination ID", value="M1")

    if st.button("Make Transaction"):

        txn = {
            "user_id": st.session_state.user_id,
            "type": txn_type,
            "amount": amount,
            "nameDest": nameDest
        }

        response = requests.post(f"{BASE_URL}/predict", json=txn)
        result = response.json()

        # -----------------------------
        # FRAUD ALERT
        # -----------------------------
        if result.get("fraud_detected"):
            st.error("🚨 FRAUD DETECTED!")
            st.warning("Transaction blocked. Balance not deducted.")
        else:
            st.success("✅ Transaction Approved")

        st.json(result)

    # -----------------------------
    # CHANGE USER
    # -----------------------------
    if st.button("🔄 Change User"):
        st.session_state.user_id = None
        st.session_state.awaiting_balance = False
        st.session_state.temp_user = None
        st.rerun()