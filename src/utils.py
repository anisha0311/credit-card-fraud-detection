import logging
import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure models/users directory exists
os.makedirs("models/users", exist_ok=True)
USERS_MODEL_DIR = "models/users"


# -----------------------------
# Global Storage (In-Memory)
# -----------------------------

# Stores user-specific SGD models
user_models = {}

# Stores recent transactions for each user
user_buffers = {}

# Tracks number of transactions per user
user_transaction_count = {}


# -----------------------------
# Get / Initialize User Model
# -----------------------------
def get_user_model(user_id):
    """
    Returns existing SGD model from memory or loads from disk.
    Creates new one if doesn't exist or if loading fails.
    """
    # Check if already in memory
    if user_id not in user_models:
        model_path = os.path.join(USERS_MODEL_DIR, f"{user_id}.pkl")
        
        # Try loading from disk
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading user model for {user_id} from disk")
                user_models[user_id] = joblib.load(model_path)
            except Exception as e:
                # Handle version mismatch or corruption
                logger.warning(f"Failed to load model for {user_id}: {e}. Creating new model.")
                user_models[user_id] = SGDClassifier(loss="log_loss")
        else:
            # Create new model
            logger.info(f"Initializing new user model for {user_id}")
            user_models[user_id] = SGDClassifier(loss="log_loss")

    return user_models[user_id]


def save_user_model(user_id):
    """
    Saves user model to disk.
    """
    if user_id in user_models:
        model_path = os.path.join(USERS_MODEL_DIR, f"{user_id}.pkl")
        joblib.dump(user_models[user_id], model_path)
        logger.info(f"Saved user model for {user_id} to {model_path}")
    else:
        logger.warning(f"No model found for user {user_id} to save")


# -----------------------------
# Update User Buffer
# -----------------------------
def update_user_buffer(user_id, X, y):
    """
    Stores transaction data for batch training.
    """
    if user_id not in user_buffers:
        user_buffers[user_id] = []

    user_buffers[user_id].append((X, y))

    logger.info(f"Buffer updated for {user_id} (size={len(user_buffers[user_id])})")

    return user_buffers[user_id]


# -----------------------------
# Check Buffer Readiness
# -----------------------------
def is_buffer_ready(user_id, buffer_size=1):
    """
    Checks if enough transactions are collected for training.
    """
    return len(user_buffers.get(user_id, [])) >= buffer_size


# -----------------------------
# Get and Clear Buffer
# -----------------------------
def get_and_clear_buffer(user_id):
    """
    Returns buffered data and clears buffer.
    """
    data = user_buffers.get(user_id, [])
    user_buffers[user_id] = []

    logger.info(f"Buffer cleared for {user_id}")

    return data


# -----------------------------
# Update Transaction Count
# -----------------------------
def update_transaction_count(user_id):
    """
    Increments and returns transaction count for user.
    """
    user_transaction_count[user_id] = user_transaction_count.get(user_id, 0) + 1

    logger.info(f"{user_id} transaction count: {user_transaction_count[user_id]}")

    return user_transaction_count[user_id]


# -----------------------------
# Final Decision Logic
# -----------------------------
def final_decision(global_pred, user_pred):
    """
    Combines global model and user model predictions.

    Rule:
    If ANY model predicts fraud → FRAUD
    """
    if global_pred == 1 or user_pred == 1:
        return 1
    return 0


# -----------------------------
# OPTIONAL DEBUG FUNCTION
# -----------------------------
# ⚠️ Not used in pipeline/predict
def print_user_stats():
    """
    Debug helper to inspect user models and buffers.
    """
    print("\n--- USER STATS ---")
    print("Users:", list(user_models.keys()))
    print("Buffer sizes:", {k: len(v) for k, v in user_buffers.items()})
    print("Transaction counts:", user_transaction_count)