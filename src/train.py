import logging
import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# ✅ Correct imports for package structure
from src.data_loader import load_and_validate_data
from src.preprocess import preprocess_data
from src.features import feature_pipeline_train


# -----------------------------
# Configure Logging (FILE + CONSOLE)
# -----------------------------
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Avoid duplicate logs
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


# -----------------------------
# Train Models
# -----------------------------
def train_models(X_train, y_train, X_test, y_test):
    results = {}

    # Logistic Regression
    logger.info("Training Logistic Regression...")

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "model": lr_model,
        "metrics": evaluate_model(y_test, lr_pred, lr_prob)
    }

    # XGBoost
    logger.info("Training XGBoost...")

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        objective="binary:logistic",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results["XGBoost"] = {
        "model": xgb_model,
        "metrics": evaluate_model(y_test, xgb_pred, xgb_prob)
    }

    return results


# -----------------------------
# Select Best Model
# -----------------------------
def select_best_model(results):
    """
    Select best model based on ROC-AUC
    """
    best_model = None
    best_score = -1
    best_name = ""

    for name, result in results.items():
        score = result["metrics"]["roc_auc"]

        logger.info(f"{name} ROC-AUC: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = result["model"]
            best_name = name

    logger.info(f"Best model selected: {best_name}")

    return best_model, best_name


# -----------------------------
# Main Training Pipeline
# -----------------------------
def train_pipeline():
    """
    ✅ This is the ONLY function used in pipeline.py
    """
    logger.info("Starting training pipeline...")

    # Step 1: Load Data
    df = load_and_validate_data()

    # Step 2: Preprocess
    df_clean = preprocess_data(df)

    # Step 3: Feature Engineering
    X, y, _ = feature_pipeline_train(df_clean)

    # Step 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 5: Train Models
    results = train_models(X_train, y_train, X_test, y_test)

    # Step 6: Select Best Model
    best_model, best_name = select_best_model(results)

    # -----------------------------
    # Step 7: Save Models
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(results["Logistic Regression"]["model"], "models/model_lr_v1.pkl")
    logger.info("Saved Logistic Regression")

    joblib.dump(results["XGBoost"]["model"], "models/model_xgb_v1.pkl")
    logger.info("Saved XGBoost")

    joblib.dump(best_model, "models/model_v1.pkl")
    logger.info(f"Saved Best Model: {best_name}")

    # Print metrics
    for name, result in results.items():
        print(f"\n{name} Metrics:")
        for metric, value in result["metrics"].items():
            print(f"{metric}: {value:.4f}")


# -----------------------------
# OPTIONAL TEST BLOCK
# -----------------------------
# ⚠️ NOT used in pipeline.py

if __name__ == "__main__":
    try:
        train_pipeline()
    except Exception as e:
        logger.error(f"Training failed: {e}")