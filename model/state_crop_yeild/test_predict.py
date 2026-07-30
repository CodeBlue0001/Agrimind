"""
Crop Production Prediction - Prediction & Testing Script
=========================================================
Uses the trained unified model to predict crop Production.

Modes:
    1. Single prediction  - provide all inputs via CLI
    2. Future forecast     - predict future years for a State + Crop
    3. Batch prediction    - feed a CSV of inputs

Usage:
    # Single prediction
    python test_predict.py predict --state "West Bengal" --crop "Coconut" --season "Whole Year" \
        --year 2025 --area 520 --rainfall 1852.9 --fertilizer 766879 --pesticide 2497

    # Future forecast (auto-fills area/rainfall/fertilizer/pesticide from historical averages)
    python test_predict.py forecast --state "Assam" --crop "Arecanut" --start-year 2025 --end-year 2030

    # Batch prediction from CSV
    python test_predict.py batch --input inputs.csv --output predictions.csv
"""

import os
import sys
import json
import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Paths -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "production_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
DEFAULT_DATASET = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "dataset", "state_wise_crop_yild.csv")
)


# -- Load model ------------------------------------------------------------
def load_model():
    """Load the trained pipeline and metadata."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)

    metadata = None
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)

    return pipeline, metadata


def print_metadata(metadata: dict) -> None:
    """Print model info."""
    if metadata:
        print(f"[MODEL] Type: {metadata['model_type']}")
        print(f"   R2: {metadata['metrics']['r2']:.4f}  |  "
              f"RMSE: {metadata['metrics']['rmse']:.2f}  |  "
              f"MAE: {metadata['metrics']['mae']:.2f}")
        print(f"   Trained on: {metadata['training_data']['rows']} rows, "
              f"{len(metadata['training_data']['states'])} states, "
              f"{len(metadata['training_data']['crops'])} crops")
        print(f"   Year range: {metadata['training_data']['year_range'][0]}-"
              f"{metadata['training_data']['year_range'][1]}")


# -- Validation helpers ----------------------------------------------------
def validate_inputs(metadata: dict, state: str, crop: str, season: str) -> None:
    """Warn if inputs aren't in the training data."""
    if metadata is None:
        return

    td = metadata["training_data"]
    issues = []
    if state not in td["states"]:
        issues.append(f"State '{state}' not in training data. Available: {td['states']}")
    if crop not in td["crops"]:
        issues.append(f"Crop '{crop}' not in training data. Available: {td['crops']}")
    if season not in td["seasons"]:
        issues.append(f"Season '{season}' not in training data. Available: {td['seasons']}")

    for issue in issues:
        print(f"   [WARN] {issue}")


# -- Single prediction -----------------------------------------------------
def predict_single(pipeline, metadata, state, crop, season, year, area,
                   rainfall, fertilizer, pesticide):
    """Predict production for a single input."""
    print(f"\n[PREDICT] Production for:")
    print(f"   State: {state}  |  Crop: {crop}  |  Season: {season}  |  Year: {year}")
    print(f"   Area: {area}  |  Rainfall: {rainfall}  |  Fertilizer: {fertilizer}  |  Pesticide: {pesticide}")

    validate_inputs(metadata, state, crop, season)

    input_df = pd.DataFrame([{
        "State": state,
        "Crop": crop,
        "Season": season,
        "Crop_Year": year,
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
    }])

    prediction = max(0, pipeline.predict(input_df)[0])
    print(f"\n   => Predicted Production: {prediction:,.2f}")

    if area > 0:
        estimated_yield = prediction / area
        print(f"   => Estimated Yield (Production/Area): {estimated_yield:.4f}")

    return prediction


# -- Future forecast -------------------------------------------------------
def get_historical_defaults(dataset_path: str, state: str, crop: str,
                            season: Optional[str] = None) -> dict:
    """
    Get recent historical averages for Area, Rainfall, Fertilizer, Pesticide
    from the dataset for a given State + Crop. Uses the last 3 years of data
    available as the baseline.
    """
    if not os.path.exists(dataset_path):
        print(f"   [WARN] Dataset not found at {dataset_path}. Cannot auto-fill defaults.")
        return None

    df = pd.read_csv(dataset_path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    mask = (df["State"] == state) & (df["Crop"] == crop)
    if season:
        mask &= (df["Season"] == season)

    subset = df[mask]
    if subset.empty:
        print(f"   [WARN] No historical data found for State='{state}', Crop='{crop}'"
              f"{f', Season={season}' if season else ''}.")
        return None

    # Use last 3 years of data
    recent_years = sorted(subset["Crop_Year"].unique())[-3:]
    recent = subset[subset["Crop_Year"].isin(recent_years)]

    defaults = {
        "Area": float(recent["Area"].mean()),
        "Annual_Rainfall": float(recent["Annual_Rainfall"].mean()),
        "Fertilizer": float(recent["Fertilizer"].mean()),
        "Pesticide": float(recent["Pesticide"].mean()),
        "Season": season or recent["Season"].mode().iloc[0],
        "based_on_years": [int(y) for y in recent_years],
    }

    return defaults


def forecast(pipeline, metadata, state, crop, start_year, end_year,
             season=None, area=None, rainfall=None, fertilizer=None,
             pesticide=None, dataset_path=None):
    """
    Predict production for a range of future years.
    Auto-fills missing inputs from historical averages.
    """
    print(f"\n[FORECAST] Production for: {state} - {crop}")
    print(f"   Years: {start_year} to {end_year}")

    # Get historical defaults for missing values
    ds_path = dataset_path or DEFAULT_DATASET
    defaults = get_historical_defaults(ds_path, state, crop, season)

    if defaults:
        print(f"   [INFO] Historical defaults (based on years {defaults['based_on_years']}):")
        if area is None:
            area = defaults["Area"]
            print(f"      Area: {area:.2f} (auto)")
        if rainfall is None:
            rainfall = defaults["Annual_Rainfall"]
            print(f"      Rainfall: {rainfall:.2f} (auto)")
        if fertilizer is None:
            fertilizer = defaults["Fertilizer"]
            print(f"      Fertilizer: {fertilizer:.2f} (auto)")
        if pesticide is None:
            pesticide = defaults["Pesticide"]
            print(f"      Pesticide: {pesticide:.2f} (auto)")
        if season is None:
            season = defaults["Season"]
            print(f"      Season: {season} (auto)")
    else:
        # If no historical data, require all inputs
        if any(v is None for v in [area, rainfall, fertilizer, pesticide, season]):
            print("[ERROR] No historical data found. Please provide all inputs manually:")
            print("   --area, --rainfall, --fertilizer, --pesticide, --season")
            sys.exit(1)

    validate_inputs(metadata, state, crop, season)

    years = list(range(start_year, end_year + 1))
    rows = []
    for yr in years:
        rows.append({
            "State": state,
            "Crop": crop,
            "Season": season,
            "Crop_Year": yr,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
        })

    input_df = pd.DataFrame(rows)
    predictions = np.maximum(0, pipeline.predict(input_df))

    results = pd.DataFrame({
        "Year": years,
        "Predicted_Production": np.round(predictions, 2),
        "Estimated_Yield": np.round(predictions / area, 4) if area > 0 else 0,
    })

    print(f"\n[RESULTS] Forecast - {state} / {crop} ({season}):\n")
    print(results.to_string(index=False))

    return results


# -- Batch prediction ------------------------------------------------------
def batch_predict(pipeline, metadata, input_csv, output_csv=None):
    """
    Predict production for a batch of inputs from a CSV.
    Expected columns: State, Crop, Season, Crop_Year, Area,
                      Annual_Rainfall, Fertilizer, Pesticide
    """
    print(f"\n[BATCH] Prediction from: {input_csv}")

    if not os.path.exists(input_csv):
        print(f"[ERROR] Input CSV not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    required = ["State", "Crop", "Season", "Crop_Year", "Area",
                "Annual_Rainfall", "Fertilizer", "Pesticide"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"   Required: {required}")
        sys.exit(1)

    predictions = np.maximum(0, pipeline.predict(df[required]))
    df["Predicted_Production"] = np.round(predictions, 2)
    df["Estimated_Yield"] = np.where(
        df["Area"] > 0,
        np.round(df["Predicted_Production"] / df["Area"], 4),
        0
    )

    print(f"   [OK] Predicted {len(df)} rows")
    print(df.to_string(index=False))

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n[SAVE] Results saved to: {output_csv}")

    return df


# -- CLI -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Crop Production Prediction - Predict & Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python test_predict.py predict --state "West Bengal" --crop "Coconut" \\
      --season "Whole Year" --year 2025 --area 520 --rainfall 1852.9 \\
      --fertilizer 766879 --pesticide 2497

  # Future forecast (auto-fills from historical data)
  python test_predict.py forecast --state "Assam" --crop "Arecanut" \\
      --start-year 2025 --end-year 2030

  # Batch from CSV
  python test_predict.py batch --input inputs.csv --output results.csv
        """
    )
    subparsers = parser.add_subparsers(dest="mode", help="Prediction mode")

    # -- predict --
    p_pred = subparsers.add_parser("predict", help="Single prediction")
    p_pred.add_argument("--state", required=True, help="State name")
    p_pred.add_argument("--crop", required=True, help="Crop name")
    p_pred.add_argument("--season", required=True, help="Season (Kharif/Rabi/Whole Year/etc.)")
    p_pred.add_argument("--year", type=int, required=True, help="Crop year")
    p_pred.add_argument("--area", type=float, required=True, help="Area in hectares")
    p_pred.add_argument("--rainfall", type=float, required=True, help="Annual rainfall (mm)")
    p_pred.add_argument("--fertilizer", type=float, required=True, help="Fertilizer usage")
    p_pred.add_argument("--pesticide", type=float, required=True, help="Pesticide usage")

    # -- forecast --
    p_fc = subparsers.add_parser("forecast", help="Forecast future years")
    p_fc.add_argument("--state", required=True, help="State name")
    p_fc.add_argument("--crop", required=True, help="Crop name")
    p_fc.add_argument("--start-year", type=int, required=True, help="Start year for forecast")
    p_fc.add_argument("--end-year", type=int, required=True, help="End year for forecast")
    p_fc.add_argument("--season", default=None, help="Season (auto-detected if omitted)")
    p_fc.add_argument("--area", type=float, default=None, help="Override area (auto if omitted)")
    p_fc.add_argument("--rainfall", type=float, default=None, help="Override rainfall")
    p_fc.add_argument("--fertilizer", type=float, default=None, help="Override fertilizer")
    p_fc.add_argument("--pesticide", type=float, default=None, help="Override pesticide")
    p_fc.add_argument("--dataset", default=None, help="Path to dataset for historical defaults")

    # -- batch --
    p_batch = subparsers.add_parser("batch", help="Batch prediction from CSV")
    p_batch.add_argument("--input", required=True, help="Input CSV path")
    p_batch.add_argument("--output", default=None, help="Output CSV path (optional)")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        print("\n[TIP] No mode specified. Use 'predict', 'forecast', or 'batch'.")
        sys.exit(0)

    # Load model
    print("=" * 60)
    print("  Crop Production Prediction")
    print("=" * 60)

    pipeline, metadata = load_model()
    print(f"[OK] Model loaded from: {MODEL_PATH}")
    print_metadata(metadata)

    # Dispatch
    if args.mode == "predict":
        predict_single(
            pipeline, metadata,
            state=args.state, crop=args.crop, season=args.season,
            year=args.year, area=args.area, rainfall=args.rainfall,
            fertilizer=args.fertilizer, pesticide=args.pesticide,
        )

    elif args.mode == "forecast":
        forecast(
            pipeline, metadata,
            state=args.state, crop=args.crop,
            start_year=args.start_year, end_year=args.end_year,
            season=args.season, area=args.area, rainfall=args.rainfall,
            fertilizer=args.fertilizer, pesticide=args.pesticide,
            dataset_path=args.dataset,
        )

    elif args.mode == "batch":
        batch_predict(pipeline, metadata, input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
