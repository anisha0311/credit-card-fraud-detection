import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Columns to DROP (avoid leakage)
# -----------------------------
DROP_COLUMNS = [
    "newbalanceOrig",
    "newbalanceDest"
]

TARGET_COLUMN = "isFraud"


# -----------------------------
# Handle Missing Values
# -----------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in dataset.
    """
    df = df.copy()

    # Numerical columns → fill with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} with median.")

    # Categorical columns → fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled missing values in {col} with mode.")

    return df


# -----------------------------
# Drop Unnecessary Columns
# -----------------------------
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that may cause data leakage or are not useful.
    """
    df = df.copy()

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    logger.info(f"Dropped columns: {DROP_COLUMNS}")

    return df


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing:
    - Drop leakage columns
    - Handle missing values

    ✅ This is the ONLY function used in pipeline.py
    """
    try:
        logger.info("Starting preprocessing...")

        df = df.copy()

        # Step 1: Drop leakage columns
        df = drop_columns(df)

        # Step 2: Handle missing values
        df = handle_missing_values(df)

        logger.info("Preprocessing completed successfully.")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise


# -----------------------------
# OPTIONAL TEST BLOCK
# -----------------------------
# ⚠️ This block is NOT used in pipeline.py
# It is only for standalone testing

if __name__ == "__main__":
    try:
        from src.data_loader import load_and_validate_data

        df = load_and_validate_data()

        processed_df = preprocess_data(df)

        print("\nProcessed Data Preview:")
        print(processed_df.head())

        print("\nMissing Values After Processing:")
        print(processed_df.isnull().sum())

    except Exception as e:
        logger.error(f"Test block failed: {e}")