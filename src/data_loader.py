import os
import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str = "data/creditcard_paysim.csv") -> pd.DataFrame:
    """
    Load dataset from the given file path.

    Args:
        file_path (str): Path to the dataset CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at path: {file_path}")

        logger.info(f"Loading dataset from {file_path}...")

        df = pd.read_csv(file_path)

        logger.info(f"Dataset loaded successfully with shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Get basic dataset information.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Summary information
    """
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    return info


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate if required columns exist in dataset.

    Args:
        df (pd.DataFrame): Input dataset

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = [
        "step", "type", "amount",
        "nameOrig", "nameDest",
        "isFraud"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info("All required columns are present.")


def load_and_validate_data(file_path: str = "data/creditcard_paysim.csv") -> pd.DataFrame:
    """
    Load dataset and validate schema.

    Args:
        file_path (str): Path to dataset

    Returns:
        pd.DataFrame: Validated dataset
    """
    df = load_data(file_path)
    validate_columns(df)

    logger.info("Data validation completed successfully.")

    return df


if __name__ == "__main__":
    # Test loading
    df = load_and_validate_data()

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Info:")
    info = get_basic_info(df)
    for key, value in info.items():
        print(f"{key}: {value}")