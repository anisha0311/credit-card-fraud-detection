import logging
import joblib
import pandas as pd
import numpy as np

from src.preprocess import preprocess_data
from src.features import feature_pipeline_inference

from src.utils import (
    get_user_model,
    update_user_buffer,
    is_buffer_ready,
    get_and_clear_buffer,
    update_transaction_count,
    save_user_model,
    final_decision
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_model = joblib.load("models/model_v1.pkl")


def predict_transaction(input_data: dict, ground_truth_label: int = None):

    try:
        df = pd.DataFrame([input_data])
        user_id = input_data.get("nameOrig", "unknown_user")

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        df = preprocess_data(df)

        # Transaction count
        txn_count = update_transaction_count(user_id)
        df["txn_frequency"] = txn_count

        # -----------------------------
        # FEATURE ENGINEERING
        # -----------------------------
        X = feature_pipeline_inference(df)

        # -----------------------------
        # GLOBAL MODEL (FIXED)
        # -----------------------------
        global_prob = global_model.predict_proba(X)[0][1]

        # 🔥 Use LOWER threshold (IMPORTANT)
        threshold = 0.1
        global_pred = int(global_prob > threshold)

        # -----------------------------
        # STRONG SIGNAL: OVERDRAFT
        # -----------------------------
        old_balance = input_data.get("oldbalanceOrg", 0)
        amount = input_data.get("amount", 0)

        if old_balance > 0 and amount > old_balance:
            logger.info("Overdraft detected → boosting fraud signal")
            global_pred = 1
            global_prob = max(global_prob, 0.95)

        # -----------------------------
        # USER MODEL
        # -----------------------------
        user_model = get_user_model(user_id)

        if txn_count <= 10:
            user_pred = global_pred
        else:
            try:
                user_pred = int(user_model.predict(X)[0])
            except Exception:
                user_pred = global_pred

        # -----------------------------
        # FINAL DECISION
        # -----------------------------
        final_pred = final_decision(global_pred, user_pred)

        # ==========================================
        # ONLINE LEARNING (FIXED LABEL LOGIC)
        # ==========================================
        if ground_truth_label is not None:
            buffer_label = ground_truth_label

        else:
            # 🔥 Better anomaly detection
            ratio = amount / (old_balance + 1)

            if ratio > 2:   # previously 10 → too high ❌
                buffer_label = 1
            else:
                buffer_label = global_pred

        update_user_buffer(user_id, X, buffer_label)

        # -----------------------------
        # TRAIN USER MODEL
        # -----------------------------
        if is_buffer_ready(user_id):

            data = get_and_clear_buffer(user_id)

            X_batch = np.vstack([d[0] for d in data])
            y_batch = np.array([d[1] for d in data])

            user_model.partial_fit(X_batch, y_batch, classes=[0, 1])

            logger.info(
                f"Updated user model for {user_id}: trained on {len(y_batch)} transactions ({y_batch.sum()} fraud)"
            )

            save_user_model(user_id)

        # -----------------------------
        # RETURN
        # -----------------------------
        return {
            "user_id": user_id,
            "transaction_count": txn_count,
            "global_prediction": global_pred,
            "user_prediction": user_pred,
            "final_prediction": final_pred,
            "fraud_probability": float(global_prob)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise