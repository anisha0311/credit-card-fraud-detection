import numpy as np
import pandas as pd
import logging
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Feature Engineering
# -----------------------------
def create_transaction_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert transaction type into credit/debit.
    """
    df = df.copy()

    df["transaction_direction"] = df["type"].apply(
        lambda x: "credit" if x == "CASH-IN" else "debit"
    )

    return df


def create_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve raw transaction type categories.
    """
    df = df.copy()

    df["transaction_type"] = df["type"].astype(str)

    return df


def create_balance_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-to-balance ratio features for anomaly detection.
    Large transactions relative to account balance are more suspicious.
    Fills missing balance columns (for syntax: set them = amount. This ensures worst-case scenario).
    """
    df = df.copy()

    # Fill missing balance columns (for inference on synthetic transactions)
    if "oldbalanceOrg" not in df.columns:
        df["oldbalanceOrg"] = df["amount"]
    if "oldbalanceDest" not in df.columns:
        df["oldbalanceDest"] = df["amount"]

    # Fill NaN values
    df["oldbalanceOrg"] = df["oldbalanceOrg"].fillna(df["amount"])
    df["oldbalanceDest"] = df["oldbalanceDest"].fillna(df["amount"])

    # Prevent division by zero
    df["amount_to_oldbalance_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_oldbalance_dest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)

    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a log-scaled amount feature to handle heavy-tailed transaction values.
    """
    df = df.copy()

    df["amount_log"] = np.log1p(df["amount"])

    return df


def create_frequency_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction frequency per user.
    Handles new users by assigning frequency = 1.
    If txn_frequency is already filled (inference state), preserve it.
    """
    df = df.copy()

    if "txn_frequency" in df.columns and df["txn_frequency"].notnull().all():
        df["txn_frequency"] = df["txn_frequency"].fillna(1)
        return df

    if "nameOrig" in df.columns:
        df["txn_frequency"] = df.groupby("nameOrig")["nameOrig"].transform("count")
    else:
        df["txn_frequency"] = 1  # new user fallback

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    """
    df = df.copy()

    df["txn_hour"] = df["step"] % 24
    df["is_night"] = (df["txn_hour"] < 6).astype(int)
    df["is_weekend"] = ((df["step"] // 24) % 7 >= 5).astype(int)

    return df


def create_account_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive origin/destination account type from name codes.
    """
    df = df.copy()

    if "nameOrig" in df.columns:
        df["origin_type"] = df["nameOrig"].str[0]
    if "nameDest" in df.columns:
        df["dest_type"] = df["nameDest"].str[0]

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """
    df = create_transaction_direction(df)
    df = create_transaction_type(df)
    df = create_balance_ratio_features(df)
    df = create_amount_features(df)
    df = create_frequency_feature(df)
    df = create_account_type_features(df)
    df = create_time_features(df)

    # Drop unused raw columns
    df = df.drop(columns=["type", "nameOrig", "nameDest"], errors="ignore")

    return df

def create_features(df):

    # Existing
    df["amount_log"] = np.log1p(df["amount"])

    # FIX 1: Safe ratio
    df["balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # FIX 2: Log ratio (VERY IMPORTANT)
    df["log_ratio"] = np.log1p(df["balance_ratio"])

    # FIX 3: Overdraft signal (CRITICAL FEATURE)
    df["is_overdraft"] = (df["amount"] > df["oldbalanceOrg"]).astype(int)

    # Transaction type encoding
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_payment"] = (df["type"] == "PAYMENT").astype(int)
    df["is_debit"] = (df["type"] == "DEBIT").astype(int)

    return df
# -----------------------------
# Build Feature Pipeline
# -----------------------------
def build_feature_pipeline():
    """
    Builds encoding + scaling pipeline.
    """

    categorical_cols = ["transaction_direction", "transaction_type", "origin_type", "dest_type"]

    numeric_cols = [
        "step",
        "amount",
        "amount_log",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_oldbalance_orig_ratio",
        "amount_to_oldbalance_dest_ratio",
        "txn_frequency",
        "txn_hour",
        "is_night",
        "is_weekend"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    return preprocessor


# -----------------------------
# Train Feature Pipeline
# -----------------------------
def feature_pipeline_train(df: pd.DataFrame, save_path="models/feature_pipeline.pkl"):
    """
    Train feature pipeline and transform data.

    ✅ Used in train.py
    """
    try:
        logger.info("Starting feature engineering (train)...")

        df = df.copy()

        df = apply_feature_engineering(df)

        y = df["isFraud"]
        X = df.drop(columns=["isFraud"])

        pipeline = build_feature_pipeline()

        X_transformed = pipeline.fit_transform(X)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, save_path)

        logger.info(f"Feature pipeline saved at {save_path}")

        return X_transformed, y, pipeline

    except Exception as e:
        logger.error(f"Error in feature training: {e}")
        raise


# -----------------------------
# Inference Feature Pipeline
# -----------------------------
def feature_pipeline_inference(df: pd.DataFrame, pipeline_path="models/feature_pipeline.pkl"):
    """
    Apply feature engineering for new data.

    ✅ Used in pipeline.py and predict.py
    """
    try:
        logger.info("Starting feature engineering (inference)...")

        df = df.copy()

        df = apply_feature_engineering(df)

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError("Feature pipeline not found. Train first.")

        pipeline = joblib.load(pipeline_path)

        X_transformed = pipeline.transform(df)

        return X_transformed

    except Exception as e:
        logger.error(f"Error in feature inference: {e}")
        raise


# -----------------------------
# OPTIONAL TEST BLOCK
# -----------------------------
# ⚠️ NOT used in pipeline.py
# Only for standalone testing

if __name__ == "__main__":
    try:
        from src.data_loader import load_and_validate_data
        from src.preprocess import preprocess_data

        df = load_and_validate_data()

        # Step 1: preprocess
        df_clean = preprocess_data(df)

        # Step 2: train features
        X, y, pipeline = feature_pipeline_train(df_clean)

        print("Train Shape:", X.shape)

        # Step 3: simulate NEW USER
        sample = df_clean.drop(columns=["isFraud"]).head(5).copy()
        sample = sample.drop(columns=["nameOrig"], errors="ignore")

        X_inf = feature_pipeline_inference(sample)

        print("Inference Shape:", X_inf.shape)
        print("Feature Names:", pipeline.get_feature_names_out())

    except Exception as e:
        logger.error(f"Test block failed: {e}")