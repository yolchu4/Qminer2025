"""
Process Control Demo - Run Script (with External Calibration)

- Trains model on train_dataset.csv
- Uses calib_dataset.csv for uncertainty calibration (mean-based)
- Runs prediction on predict_dataset.csv
- Evaluates on compare_dataset.csv (NO leakage)

Author: Demo / Research Pipeline
"""

import sys
import json
from pathlib import Path

# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from core.conditional_global_mixture import ConditionalGlobalMixtureModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
OUT_DIR = Path("out/demo_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

USE_CALIBRATION = True
CALIB_ALPHA = 0.90  # target coverage

# --------------------------------------------------
# Calibration helper (mean-based, external)
# --------------------------------------------------

def fit_absolute_calibration(mean_pred, y_true, alpha=0.9):
    """
    mean_pred: [n, n_targets]
    y_true:    [n, n_targets]

    Returns:
        delta: [n_targets] absolute half-width for coverage alpha
    """
    abs_err = np.abs(y_true - mean_pred)
    delta = np.quantile(abs_err, alpha, axis=0)
    return delta

# --------------------------------------------------
# Main Demo
# --------------------------------------------------

def run_demo():

    print(">>> START DEMO")
    print(f"Project root: {Path('.').resolve()}")

    # -----------------------------
    # Load datasets
    # -----------------------------
    print("Loading datasets...")

    train = pd.read_csv(DATA_DIR / "train_dataset.csv")
    calib = pd.read_csv(DATA_DIR / "calib_dataset.csv")
    pred = pd.read_csv(DATA_DIR / "predict_dataset.csv")
    compare = pd.read_csv(DATA_DIR / "compare_dataset.csv")

    print(DATA_DIR / "train_dataset.csv")
    print(DATA_DIR / "calib_dataset.csv")
    print(DATA_DIR / "predict_dataset.csv")
    print(DATA_DIR / "compare_dataset.csv")

    print(f"Train shape:   {train.shape}")
    print(f"Calib shape:   {calib.shape}")
    print(f"Predict shape: {pred.shape}")
    print(f"Compare shape: {compare.shape}")

    # -----------------------------
    # Fit model (TRAIN ONLY)
    # -----------------------------
    print("Fitting model...")

    model = ConditionalGlobalMixtureModel(
        numerical_cols=KNOWN_COLS,
        categorical_cols=[],
        n_regimes=6,
        random_state=42,
    )

    model.fit(
        X=train[KNOWN_COLS],
        U=train[UNKNOWN_COLS].values,
    )

    print("Model fitted.")

    # -----------------------------
    # Calibration (OPTIONAL)
    # -----------------------------
    if USE_CALIBRATION:
        print("Running calibration on calib_dataset...")

        calib_out = model.predict(calib[KNOWN_COLS])
        calib_mean = calib_out["mean"]
        y_calib = calib[UNKNOWN_COLS].values

        calib_delta = fit_absolute_calibration(
            mean_pred=calib_mean,
            y_true=y_calib,
            alpha=CALIB_ALPHA,
        )

        print("Calibration deltas:")
        for name, d in zip(UNKNOWN_COLS, calib_delta):
            print(f"  {name:<30}: {d:.4f}")

    else:
        calib_delta = None

    # -----------------------------
    # Prediction (NO LEAKAGE)
    # -----------------------------
    print("Running prediction...")

    out = model.predict(pred[KNOWN_COLS])
    mean = out["mean"]

    if USE_CALIBRATION:
        q_low = mean - calib_delta
        q_high = mean + calib_delta
    else:
        q_low = out["q_low"]
        q_high = out["q_high"]

    print("Prediction done.")

    # -----------------------------
    # Save outputs
    # -----------------------------
    mean_df = pd.DataFrame(mean, columns=UNKNOWN_COLS)
    ql_df = pd.DataFrame(q_low, columns=UNKNOWN_COLS)
    qh_df = pd.DataFrame(q_high, columns=UNKNOWN_COLS)
    w_df = pd.DataFrame(out["regime_weights"])

    mean_df.to_csv(OUT_DIR / "pred_mean.csv", index=False)
    ql_df.to_csv(OUT_DIR / "pred_q05.csv", index=False)
    qh_df.to_csv(OUT_DIR / "pred_q95.csv", index=False)
    w_df.to_csv(OUT_DIR / "regime_weights.csv", index=False)

    print(f"Output dir: {OUT_DIR}")
    print("Files written:")
    for p in OUT_DIR.iterdir():
        print(p)

    # -----------------------------
    # Evaluation (COMPARE ONLY)
    # -----------------------------
    print("\nSelected distributions per regime:")
    print(out["selected_distributions"])

    y_true = compare[UNKNOWN_COLS].values

    coverage = np.mean(
        (y_true >= q_low) & (y_true <= q_high),
        axis=0,
    )

    coverage_dict = {
        name: float(c) for name, c in zip(UNKNOWN_COLS, coverage)
    }

    print("\nCoverage q05–q95 per parameter:")
    for name, c in coverage_dict.items():
        print(f"{name:<30}: {c:.3f}")

    mae = np.mean(np.abs(y_true - mean), axis=0)

    mae_dict = {
        name: float(e) for name, e in zip(UNKNOWN_COLS, mae)
    }

    print("\nMAE (mean prediction):")
    for name, e in mae_dict.items():
        print(f"{name:<30}: {e:.3f}")

    # -----------------------------
    # Metrics export (Stage 3)
    # -----------------------------
    metrics = {
        "delta": {
            name: float(d)
            for name, d in zip(UNKNOWN_COLS, calib_delta)
        } if USE_CALIBRATION else {},
        "coverage": coverage_dict,
        "mae": mae_dict,
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to: {metrics_path}")
    print(">>> END DEMO")

# --------------------------------------------------
# CLI Entry
# --------------------------------------------------

if __name__ == "__main__":
    run_demo()
