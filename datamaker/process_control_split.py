"""
Process Control Demo - Data Split & Mask Script

This script takes a full raw dataset and produces three demo files:
1) Training data (70%) - known + unknown parameters
2) Prediction input (30%) - known parameters only
3) Ground truth (30%) - known + unknown parameters

Author: Demo / Research Pipeline
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration (DEMO CONTRACT)
# --------------------------------------------------

TIME_COL = "timestamp"

KNOWN_COLS = [
    "inlet_flow_A",
    "inlet_flow_B",
    "inlet_flow_C",
    "inlet_conc_A",
    "inlet_conc_B",
    "inlet_conc_C",
    "outlet_flow_1",
    "outlet_flow_2",
    "outlet_conc_1",
    "outlet_conc_2",
]

UNKNOWN_COLS = [
    "reactor_temperature",
    "reactor_pressure",
    "agitator_speed",
    "reactant_feed_flow",
    "reactant_feed_concentration",
]

TRAIN_RATIO = 0.7
RANDOM_SEED = 42


# --------------------------------------------------
# Core Logic
# --------------------------------------------------

def split_process_control_data(
    raw_csv_path: str,
    output_dir: str = "data/demo",
):
    """
    Split raw dataset into train / prediction / ground-truth files.
    """

    raw_csv_path = Path(raw_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df = pd.read_csv(raw_csv_path)

    # Sanity checks
    required_cols = [TIME_COL] + KNOWN_COLS + UNKNOWN_COLS
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Train / Predict split
    train_df, predict_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # -----------------------------
    # File 1: Training data (70%)
    # -----------------------------
    train_full = train_df.copy()

    train_path = output_dir / "train_full_70.csv"
    train_full.to_csv(train_path, index=False)

    # -----------------------------
    # File 2: Prediction input (30%, masked)
    # -----------------------------
    predict_known = predict_df[[TIME_COL] + KNOWN_COLS].copy()

    predict_known_path = output_dir / "predict_known_30.csv"
    predict_known.to_csv(predict_known_path, index=False)

    # -----------------------------
    # File 3: Ground truth (30%)
    # -----------------------------
    predict_truth = predict_df[[TIME_COL] + KNOWN_COLS + UNKNOWN_COLS].copy()

    predict_truth_path = output_dir / "predict_truth_30.csv"
    predict_truth.to_csv(predict_truth_path, index=False)

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n--- PROCESS CONTROL DEMO DATA GENERATED ---")
    print(f"Raw input file:        {raw_csv_path}")
    print(f"Training data (70%):   {train_path}")
    print(f"Prediction input (30%):{predict_known_path}")
    print(f"Ground truth (30%):    {predict_truth_path}")
    print("------------------------------------------")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_full)}")
    print(f"Predict rows: {len(predict_df)}")


# --------------------------------------------------
# CLI Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    split_process_control_data(
        raw_csv_path="data/raw/process_control_full.csv",
        output_dir="data/demo",
    )
