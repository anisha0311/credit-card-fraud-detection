import logging
import joblib
import pandas as pd
import os

from src.data_loader import load_and_validate_data
from src.preprocess import preprocess_data
from src.features import (
    feature_pipeline_train,
    feature_pipeline_inference
)

from src.train import train_pipeline


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
# Pipeline Class
# -----------------------------
class FraudDetectionPipeline:

    def __init__(self):
        self.global_model = None

    # -----------------------------
    # Train Full Pipeline
    # -----------------------------
    def train(self):
        logger.info("Running full training pipeline...")
        train_pipeline()
        logger.info("Training completed.")

    # -----------------------------
    # Load Models
    # -----------------------------
    def load_model(self):
        logger.info("Loading trained model...")
        self.global_model = joblib.load("models/model_v1.pkl")

    # -----------------------------
    # Predict (Single Input)
    # -----------------------------
    def predict(self, input_data: dict):
        """
        Predict fraud for a single transaction (global model only)
        Online learning handled separately in predict.py
        """

        if self.global_model is None:
            self.load_model()

        logger.info(f"Received input: {input_data}")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Step 1: Preprocess
        df = preprocess_data(df)

        # Step 2: Feature Engineering
        X = feature_pipeline_inference(df)

        # Step 3: Predict
        pred = self.global_model.predict(X)[0]
        prob = self.global_model.predict_proba(X)[0][1]

        logger.info(f"Prediction: {pred}, Probability: {prob:.4f}")

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    pipeline = FraudDetectionPipeline()

    logger.info("=====================================")
    logger.info( "Starting Fraud Detection Pipeline...")
    logger.info("=====================================")

    # 🔹 Train
    pipeline.train()

    logger.info("=====================================")
    logger.info("Testing Prediction...")
    logger.info("=====================================")

    # 🔹 Example prediction
    sample_input = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000,
        "nameOrig": "C123",
        "nameDest": "C456"
    }

    result = pipeline.predict(sample_input)

    print("\nPrediction Result:", result)