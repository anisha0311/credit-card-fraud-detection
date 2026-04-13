import logging
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ✅ Correct imports
from src.data_loader import load_and_validate_data
from src.preprocess import preprocess_data
from src.features import feature_pipeline_inference


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable scientific notation globally
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False


# -----------------------------
# Load Models
# -----------------------------
def load_models():
    lr_model = joblib.load("models/model_lr_v1.pkl")
    xgb_model = joblib.load("models/model_xgb_v1.pkl")
    return lr_model, xgb_model


# -----------------------------
# Get Feature Names
# -----------------------------
def get_feature_names():
    pipeline = joblib.load("models/feature_pipeline.pkl")
    return pipeline.get_feature_names_out()


# -----------------------------
# Format Axis
# -----------------------------
def format_axis(ax):
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


# -----------------------------
# Main Evaluation
# -----------------------------
def evaluate():
    """
    ✅ This is the ONLY function used externally (pipeline / manual run)
    """
    logger.info("Starting evaluation...")

    os.makedirs("model_results", exist_ok=True)

    # Load data
    df = load_and_validate_data()
    df = preprocess_data(df)

    # Features
    X = feature_pipeline_inference(df.drop(columns=["isFraud"]))
    y = df["isFraud"]

    # Load models
    lr_model, xgb_model = load_models()

    # Predictions
    y_pred_lr = lr_model.predict(X)
    y_prob_lr = lr_model.predict_proba(X)[:, 1]

    y_pred_xgb = xgb_model.predict(X)
    y_prob_xgb = xgb_model.predict_proba(X)[:, 1]

    feature_names = get_feature_names()

    # -----------------------------
    # Create Figure (2x3 layout)
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # LR Confusion Matrix
    cm_lr = confusion_matrix(y, y_pred_lr)
    ConfusionMatrixDisplay(cm_lr).plot(ax=axes[0, 0], colorbar=False, values_format='d')
    axes[0, 0].set_title("LR Confusion Matrix")

    # ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y, y_prob_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y, y_prob_xgb)

    auc_lr = roc_auc_score(y, y_prob_lr)
    auc_xgb = roc_auc_score(y, y_prob_xgb)

    axes[0, 1].plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.4f})")
    axes[0, 1].plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={auc_xgb:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--")

    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend()

    format_axis(axes[0, 1])

    # XGB Confusion Matrix
    cm_xgb = confusion_matrix(y, y_pred_xgb)
    ConfusionMatrixDisplay(cm_xgb).plot(ax=axes[0, 2], colorbar=False, values_format='d')
    axes[0, 2].set_title("XGB Confusion Matrix")

    # LR Feature Importance
    importance_lr = np.abs(lr_model.coef_[0])
    indices_lr = np.argsort(importance_lr)[-10:]

    values = importance_lr[indices_lr]
    names = [feature_names[i] for i in indices_lr]

    axes[1, 0].barh(range(len(indices_lr)), values)
    axes[1, 0].set_yticks([])

    for i, (val, name) in enumerate(zip(values, names)):
        axes[1, 0].text(val * 0.01, i, name, va='center', fontsize=8)

    axes[1, 0].set_title("LR Feature Importance")
    format_axis(axes[1, 0])

    # Empty middle
    axes[1, 1].axis("off")

    # XGB Feature Importance
    importance_xgb = xgb_model.feature_importances_
    indices_xgb = np.argsort(importance_xgb)[-10:]

    values = importance_xgb[indices_xgb]
    names = [feature_names[i] for i in indices_xgb]

    axes[1, 2].barh(range(len(indices_xgb)), values)
    axes[1, 2].set_yticks([])

    for i, (val, name) in enumerate(zip(values, names)):
        axes[1, 2].text(val * 0.01, i, name, va='center', fontsize=8)

    axes[1, 2].set_title("XGB Feature Importance")
    format_axis(axes[1, 2])

    # Layout
    plt.tight_layout()

    # Save
    save_path = "model_results/evaluation_plots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    logger.info(f"Saved plots at {save_path}")

    plt.show()


# -----------------------------
# OPTIONAL TEST BLOCK
# -----------------------------
# ⚠️ NOT used in pipeline.py

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")