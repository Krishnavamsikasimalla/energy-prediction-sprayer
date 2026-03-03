#!/usr/bin/env python3
"""
Validation script — aligned with train.py features and paths.
Run after train.py to get full validation report.
"""
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "data/processed/training_dataset.csv"
MODEL_PATH = "outputs/energy_model.pkl"
PLOT_DIR   = Path("outputs/plots")

TARGET  = "Energy_per_meter"


def validate():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["Distance_m"] > 1].copy()

    if TARGET not in df.columns:
        df[TARGET] = df["Energy_Wh"] / df["Distance_m"]
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[TARGET])

    pipeline = joblib.load(MODEL_PATH)
    feature_names = pipeline.named_steps["imputer"].feature_names_in_

    X = df[list(feature_names)].copy()
    y = df[TARGET]

    # Time-ordered test split (last 15%)
    split = int(len(X) * 0.85)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    preds  = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test.values - preds) / np.where(y_test.values == 0, 1e-9, y_test.values))) * 100

    logger.info("── Validation Metrics ────────────────────────")
    logger.info(f"  Samples : {len(y_test)}")
    logger.info(f"  MAE     : {mae:.5f} Wh/m")
    logger.info(f"  RMSE    : {rmse:.5f} Wh/m")
    logger.info(f"  R²      : {r2:.4f}")
    logger.info(f"  MAPE    : {mape:.2f}%")

    # Per-file breakdown
    if "Source_File" in df.columns:
        test_df = df.iloc[split:].copy()
        test_df["Predicted"] = preds
        test_df["Actual"]    = y_test.values
        logger.info("── Per-File Breakdown ────────────────────────")
        for fname, grp in test_df.groupby("Source_File"):
            f_mae = mean_absolute_error(grp["Actual"], grp["Predicted"])
            logger.info(f"  {fname:<35} MAE={f_mae:.5f}  n={len(grp)}")

    # Time-series plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(y_test.values,  label="Actual",    linewidth=2)
    axes[0].plot(preds,          label="Predicted", linewidth=2, linestyle="--")
    axes[0].set_title("Energy/meter — Actual vs Predicted (Test Set)")
    axes[0].set_ylabel("Wh/m")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_test.values - preds
    axes[1].bar(range(len(residuals)), residuals, alpha=0.6, color="steelblue")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_title("Residuals per Sample")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Error (Wh/m)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / "validation_report.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Validation plot saved → {out}")


if __name__ == "__main__":
    validate()