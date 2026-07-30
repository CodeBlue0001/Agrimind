"""
Crop Production Prediction - Training Script
=============================================
Trains a single unified model to predict crop Production for any
State + Crop + Season + Year combination across all Indian states.

Usage:
    python train.py
    python train.py --dataset "path/to/state_wise_crop_yild.csv"

Output:
    models/production_model.pkl       - trained sklearn Pipeline
    models/model_metadata.json        - feature info, metrics, training timestamp
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Paths -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "dataset", "state_wise_crop_yild.csv")
)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


# -- Data loading & cleaning ------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the CSV and apply basic cleaning."""
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Strip trailing whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop rows with missing values in critical columns
    critical_cols = ["State", "Crop", "Season", "Crop_Year", "Production"]
    before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    after = len(df)
    if before != after:
        print(f"   [WARN] Dropped {before - after} rows with missing values in {critical_cols}")

    print(f"   [OK] Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   States: {df['State'].nunique()}  |  Crops: {df['Crop'].nunique()}  |  "
          f"Years: {df['Crop_Year'].min()}-{df['Crop_Year'].max()}")
    return df


# -- Feature engineering ----------------------------------------------------
CATEGORICAL_FEATURES = ["State", "Crop", "Season"]
NUMERICAL_FEATURES = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
TARGET = "Production"

# NOTE: 'Yield' is intentionally excluded -- it is derived from
# Production / Area, so including it would cause data leakage.


def build_pipeline(model) -> Pipeline:
    """Build a preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


# -- Training ---------------------------------------------------------------
def train(df: pd.DataFrame) -> dict:
    """
    Train RandomForest and GradientBoosting regressors, pick the best.
    Returns dict with best pipeline, metrics, and metadata.
    """
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Train/Test split: {len(X_train)} train, {len(X_test)} test")

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        print(f"\n[TRAIN] Training {name}...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        # Cross-validation R2 (3-fold for speed)
        cv_scores = cross_val_score(
            build_pipeline(model.__class__(**model.get_params())),
            X, y, cv=3, scoring="r2", n_jobs=-1
        )
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"] = float(cv_scores.std())

        results[name] = {"pipeline": pipeline, "metrics": metrics}
        print(f"   R2: {metrics['r2']:.4f}  |  RMSE: {metrics['rmse']:.2f}  |  "
              f"MAE: {metrics['mae']:.2f}  |  CV R2: {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")

    # Pick best by R2
    best_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best = results[best_name]
    print(f"\n[BEST] Winner: {best_name}  (R2 = {best['metrics']['r2']:.4f})")

    # Show sample predictions
    y_pred_best = best["pipeline"].predict(X_test)
    sample = pd.DataFrame({
        "State": X_test["State"].values[:10],
        "Crop": X_test["Crop"].values[:10],
        "Year": X_test["Crop_Year"].values[:10],
        "Actual": y_test.values[:10],
        "Predicted": np.round(y_pred_best[:10], 2),
    })
    print(f"\n[SAMPLE] Predictions ({best_name}):")
    print(sample.to_string(index=False))

    return {
        "best_name": best_name,
        "pipeline": best["pipeline"],
        "metrics": best["metrics"],
        "all_results": {k: v["metrics"] for k, v in results.items()},
    }


# -- Save artifacts ---------------------------------------------------------
def save_model(result: dict, df: pd.DataFrame) -> None:
    """Save the trained pipeline and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save pipeline
    model_path = os.path.join(MODEL_DIR, "production_model.pkl")
    joblib.dump(result["pipeline"], model_path)
    print(f"\n[SAVE] Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_type": result["best_name"],
        "target": TARGET,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "metrics": result["metrics"],
        "all_model_metrics": result["all_results"],
        "training_data": {
            "rows": len(df),
            "states": sorted(df["State"].unique().tolist()),
            "crops": sorted(df["Crop"].unique().tolist()),
            "seasons": sorted(df["Season"].unique().tolist()),
            "year_range": [int(df["Crop_Year"].min()), int(df["Crop_Year"].max())],
        },
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVE] Metadata saved to: {meta_path}")


# -- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train crop production prediction model")
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help="Path to the state_wise_crop_yild.csv file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Crop Production Prediction - Model Training")
    print("=" * 60)

    df = load_and_clean(args.dataset)
    result = train(df)
    save_model(result, df)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
