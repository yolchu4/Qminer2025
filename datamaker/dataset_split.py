"""
Process Control Demo - Data Split & Mask Script (with Calibration)

This script takes a full raw dataset and produces four demo files:
1) Training data (65%)        - known + unknown parameters
2) Calibration data (20%)     - known + unknown parameters
3) Prediction input (15%)     - known parameters only
4) Ground truth (15%)         - known + unknown parameters

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

TRAIN_RATIO = 0.65
CALIB_RATIO = 0.20   # of full data
COMPARE_RATIO = 0.15 # of full data

RANDOM_SEED = 42


# --------------------------------------------------
# Core Logic
# --------------------------------------------------

def split_process_control_data(
    raw_csv_path: str,
    output_dir: str = "data",
):
    """
    Split raw dataset into train / calibration / prediction / ground-truth files.
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

    # --------------------------------------------------
    # Step 1: Train vs Non-train
    # --------------------------------------------------
    train_df, rest_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # --------------------------------------------------
    # Step 2: Calibration vs Compare
    # --------------------------------------------------
    calib_frac_of_rest = CALIB_RATIO / (CALIB_RATIO + COMPARE_RATIO)

    calib_df, compare_df = train_test_split(
        rest_df,
        train_size=calib_frac_of_rest,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # -----------------------------
    # File 1: Training data (65%)
    # -----------------------------
    train_path = output_dir / "train_dataset.csv"
    train_df.to_csv(train_path, index=False)

    # -----------------------------
    # File 2: Calibration data (20%)
    # -----------------------------
    calib_path = output_dir / "calib_dataset.csv"
    calib_df.to_csv(calib_path, index=False)

    # -----------------------------
    # File 3: Prediction input (15%, masked)
    # -----------------------------
    predict_known = compare_df[[TIME_COL] + KNOWN_COLS].copy()
    predict_known_path = output_dir / "predict_dataset.csv"
    predict_known.to_csv(predict_known_path, index=False)

    # -----------------------------
    # File 4: Ground truth (15%)
    # -----------------------------
    compare_truth = compare_df[[TIME_COL] + KNOWN_COLS + UNKNOWN_COLS].copy()
    compare_truth_path = output_dir / "compare_dataset.csv"
    compare_truth.to_csv(compare_truth_path, index=False)

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n--- PROCESS CONTROL DEMO DATA GENERATED ---")
    print(f"Raw input file:          {raw_csv_path}")
    print(f"Training data (65%):     {train_path}")
    print(f"Calibration data (20%):  {calib_path}")
    print(f"Prediction input (15%):  {predict_known_path}")
    print(f"Ground truth (15%):      {compare_truth_path}")
    print("------------------------------------------")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Calibration rows: {len(calib_df)}")
    print(f"Compare/Predict rows: {len(compare_df)}")


# --------------------------------------------------
# CLI Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    split_process_control_data(
        raw_csv_path="data/full_dataset.csv",
        output_dir="data",
    )
