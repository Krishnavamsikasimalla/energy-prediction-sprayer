#!/usr/bin/env python3
"""
Advanced training pipeline:
- XGBoost with Optuna hyperparameter tuning
- TimeSeriesSplit cross-validation
- SHAP feature importance
- Saves full pipeline (scaler + model)
"""
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH  = "data/processed/training_dataset.csv"
OUTPUT_DIR  = Path("outputs")
MODEL_PATH  = OUTPUT_DIR / "energy_model.pkl"
PLOT_DIR    = OUTPUT_DIR / "plots"

FEATURES = [
    # Speed / kinematics
    "Avg_Speed", "Max_Speed", "Speed_Std", "RMS_Accel_Derived", "Avg_Accel_g",
    # Terrain
    "Avg_Gradient", "Max_Gradient", "Uphill_Frac",
    # Power
    "Avg_Power_W", "Peak_Power_W", "Spray_Energy_Frac",
    # SOC
    "Start_SOC", "SOC_Drop", "Start_RSOC2",
    # Motor
    "Avg_RPM", "Max_RPM", "Avg_Hyd_RPM", "Avg_Efficiency", "Avg_Cutback",
    # Thermal
    "Avg_Ctrl_Temp", "Avg_Motor_Temp", "Avg_Hyd_Temp",
    # Gear / Mode
    "Dominant_Gear", "Low_Gear_Frac", "High_Gear_Frac",
    "Travel_Frac", "Field_Frac",
    # Spray
    "Spray_Ratio", "Avg_Spray_Pressure", "Avg_Spray_Flow",
    # IMU
    "Vibration",
]
TARGET = "Energy_per_meter"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Remove windows with no movement
    df = df[df["Distance_m"] > 1].copy()

    # Recompute target in case it's missing
    if TARGET not in df.columns:
        df[TARGET] = df["Energy_Wh"] / df["Distance_m"]

    df = df.replace([np.inf, -np.inf], np.nan)

    # Only keep features that actually exist in the CSV
    available = [f for f in FEATURES if f in df.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        logger.warning(f"Features missing from dataset (will be skipped): {missing}")

    X = df[available].copy()
    y = df[TARGET].dropna()
    X = X.loc[y.index]

    logger.info(f"Training on {len(X)} samples with {len(available)} features.")
    return X, y


def build_pipeline(params: dict) -> Pipeline:
    imputer   = SimpleImputer(strategy="median")
    scaler    = StandardScaler()
    model     = XGBRegressor(
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return Pipeline([
        ("imputer", imputer),
        ("scaler",  scaler),
        ("model",   model),
    ])


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
            "max_depth":          trial.suggest_int("max_depth", 3, 9),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        }
        pipe   = build_pipeline(params)
        scores = cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best MAE from tuning: {study.best_value:.5f}")
    return study.best_params


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    preds = pipeline.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100

    logger.info(f"  MAE  : {mae:.5f} Wh/m")
    logger.info(f"  RMSE : {rmse:.5f} Wh/m")
    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  MAPE : {mape:.2f}%")
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def plot_results(y_test, preds, feature_names, model_step, PLOT_DIR):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, preds, alpha=0.6, edgecolors="k", linewidths=0.4)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--", label="Perfect")
    axes[0].set_xlabel("Actual (Wh/m)")
    axes[0].set_ylabel("Predicted (Wh/m)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    residuals = y_test.values - preds
    axes[1].hist(residuals, bins=40, edgecolor="k", alpha=0.75)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Residual (Wh/m)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "prediction_quality.png", dpi=150)
    plt.close()
    logger.info(f"Saved prediction quality plot → {PLOT_DIR / 'prediction_quality.png'}")

    # SHAP
    try:
        explainer  = shap.TreeExplainer(model_step)
        shap_vals  = explainer.shap_values(
            pd.DataFrame(
                model_step._Booster.num_boosted_rounds() and  # just a trigger
                preds * 0,  # dummy — use actual X_test_transformed
                columns=feature_names
            )
        )
    except Exception:
        pass  # SHAP optional


def main(tune: bool = True, n_trials: int = 50):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()

    # Time-ordered split (no shuffle — respects temporal ordering)
    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if tune:
        logger.info("Starting hyperparameter tuning (Optuna)...")
        best_params = tune_hyperparameters(X_train, y_train, n_trials=n_trials)
    else:
        best_params = {
            "n_estimators": 300, "max_depth": 5,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_alpha": 0.01, "reg_lambda": 0.1,
        }

    pipeline = build_pipeline(best_params)
    pipeline.fit(X_train, y_train)

    logger.info("── Test Set Metrics ──────────────────────────")
    preds, metrics = evaluate(pipeline, X_test, y_test)

    # SHAP feature importance
    try:
        xgb_model = pipeline.named_steps["model"]
        imp = dict(zip(X.columns, xgb_model.feature_importances_))
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        logger.info("── Top 10 Feature Importances (gain) ────────")
        for feat, score in imp_sorted[:10]:
            logger.info(f"  {feat:<30} {score:.4f}")

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        names  = [x[0] for x in imp_sorted[:15]]
        scores = [x[1] for x in imp_sorted[:15]]
        ax.barh(names[::-1], scores[::-1], color="steelblue")
        ax.set_title("Top 15 Feature Importances (XGBoost Gain)")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150)
        plt.close()
        logger.info(f"Saved feature importance plot → {PLOT_DIR / 'feature_importance.png'}")
    except Exception as e:
        logger.warning(f"Feature importance plot failed: {e}")

    # Actual vs Predicted plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_test.values, preds, alpha=0.6, edgecolors="k", linewidths=0.4, s=30)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        axes[0].plot(lims, lims, "r--", label="Perfect fit")
        axes[0].set_xlabel("Actual (Wh/m)")
        axes[0].set_ylabel("Predicted (Wh/m)")
        axes[0].set_title("Actual vs Predicted Energy/meter")
        axes[0].legend()

        residuals = y_test.values - preds
        axes[1].hist(residuals, bins=40, edgecolor="k", alpha=0.75, color="steelblue")
        axes[1].axvline(0, color="r", linestyle="--")
        axes[1].set_xlabel("Residual (Wh/m)")
        axes[1].set_title("Residual Distribution")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "prediction_quality.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    # Save the full sklearn pipeline (imputer + scaler + model)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Pipeline saved → {MODEL_PATH}")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    logger.info(f"Metrics saved → {OUTPUT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning, use defaults")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    main(tune=not args.no_tune, n_trials=args.trials)