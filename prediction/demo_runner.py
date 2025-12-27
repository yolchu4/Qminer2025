from pathlib import Path
from typing import Dict
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prediction.model import DistributionPredictor


# =========================================================
# Demo Contract
# =========================================================

TIME_COL = "timestamp"

KNOWN_COLS = [
    # Process Inputs
    "inlet_flow_A",
    "inlet_flow_B",
    "inlet_flow_C",
    "inlet_conc_A",
    "inlet_conc_B",
    "inlet_conc_C",
    # Process Outputs
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


# =========================================================
# Utilities
# =========================================================

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_numpy(df: pd.DataFrame, cols):
    return df[cols].to_numpy(dtype=float, copy=True)


def _validate_contract(df: pd.DataFrame, required_cols, name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def assign_sample_distributions(n_samples: int) -> np.ndarray:
    """
    Deterministic, balanced distribution assignment (DEMO-only).
    Activates multi-distribution engine for training the classifier.
    """
    distribution_cycle = [
        "uniform",
        "normal",
        "exponential",
        "lognormal",
        "gamma",
        "weibull",
        "chiSquared",
        "beta",
    ]
    return np.array(
        [distribution_cycle[i % len(distribution_cycle)] for i in range(n_samples)],
        dtype=object,
    )


def build_regression_targets(y_unknown: np.ndarray) -> np.ndarray:
    """
    Flattened regression targets:
    [mean_1, scale_1, mean_2, scale_2, ...]  -> shape (n_samples, n_targets*2)
    """
    eps = 1e-6
    mean = y_unknown
    scale = np.std(y_unknown, axis=0, keepdims=True)
    scale = np.maximum(scale, eps)
    scale = np.repeat(scale, y_unknown.shape[0], axis=0)

    params = np.stack([mean, scale], axis=-1)
    return params.reshape(params.shape[0], -1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    per_param = {}
    maes, rmses = [], []

    for i, name in enumerate(UNKNOWN_COLS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)

        per_param[name] = {"MAE": float(mae), "RMSE": float(rmse)}
        maes.append(mae)
        rmses.append(rmse)

    return {
        "per_parameter": per_param,
        "overall": {
            "MAE_mean": float(np.mean(maes)),
            "RMSE_mean": float(np.mean(rmses)),
        },
    }


# =========================================================
# Main Demo
# =========================================================

def run_demo(
    train_path: Path,
    predict_known_path: Path,
    predict_truth_path: Path,
    out_dir: Path,
    random_state: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    train_df = _load_csv(train_path)
    known_df = _load_csv(predict_known_path)
    truth_df = _load_csv(predict_truth_path)

    # ---------- Validate contract ----------
    _validate_contract(train_df, [TIME_COL] + KNOWN_COLS + UNKNOWN_COLS, "train_full_70.csv")
    _validate_contract(known_df, [TIME_COL] + KNOWN_COLS, "predict_known_30.csv")
    _validate_contract(truth_df, [TIME_COL] + KNOWN_COLS + UNKNOWN_COLS, "predict_truth_30.csv")

    # ---------- Training ----------
    X_train = train_df[KNOWN_COLS]
    y_true_train = _to_numpy(train_df, UNKNOWN_COLS)

    y_dist = assign_sample_distributions(len(train_df))  # 1D labels
    y_reg = build_regression_targets(y_true_train)       # 2D targets (n, 10)

    model = DistributionPredictor(
        process_name="process_control_demo",
        random_state=random_state,
    )
    model.fit(X_train, y_dist, y_reg)

    # ---------- Prediction ----------
    X_pred = known_df[KNOWN_COLS]
    raw_pred = model.predict(X_pred)["parameter_predictions"]  # shape (n, 10)

    means = raw_pred[:, ::2]    # (n, 5)
    scales = raw_pred[:, 1::2]  # (n, 5)

    preds_df = pd.DataFrame(means, columns=UNKNOWN_COLS)
    preds_df.insert(0, TIME_COL, known_df[TIME_COL])
    preds_path = out_dir / "predictions_30.csv"
    preds_df.to_csv(preds_path, index=False)

    # ---------- Evaluation ----------
    y_true = _to_numpy(truth_df, UNKNOWN_COLS)
    metrics = compute_metrics(y_true, means)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- Error report ----------
    error_rows = []
    for i, param in enumerate(UNKNOWN_COLS):
        for j in range(len(y_true)):
            abs_err = abs(y_true[j, i] - means[j, i])
            rel_err = abs_err / (abs(y_true[j, i]) + 1e-6)

            error_rows.append({
                "timestamp": truth_df.loc[j, TIME_COL],
                "parameter": param,
                "true": float(y_true[j, i]),
                "predicted": float(means[j, i]),
                "abs_error": float(abs_err),
                "rel_error_percent": float(100 * rel_err),
            })

    error_path = out_dir / "error_report_30.csv"
    pd.DataFrame(error_rows).to_csv(error_path, index=False)

    # ---------- Uncertainty ----------
    k = 2.0  # ~95% band (model-derived)
    unc_rows = []
    for i, param in enumerate(UNKNOWN_COLS):
        for j in range(len(means)):
            unc_rows.append({
                "timestamp": known_df.loc[j, TIME_COL],
                "parameter": param,
                "mean": float(means[j, i]),
                "lower_95": float(means[j, i] - k * scales[j, i]),
                "upper_95": float(means[j, i] + k * scales[j, i]),
            })

    unc_path = out_dir / "predictions_with_uncertainty.csv"
    pd.DataFrame(unc_rows).to_csv(unc_path, index=False)

    # ---------- Summary ----------
    summary = {
        "evaluation_scope": {
            "train_fraction": 0.7,
            "test_fraction": 0.3,
            "n_train_samples": int(len(train_df)),
            "n_test_samples": int(len(y_true)),
        },
        "overall_accuracy": metrics["overall"],
        "interpretation": {
            "reliability": "moderate",
            "recommended_usage": "decision support",
            "not_recommended_for": "direct closed-loop control",
            "notes": "Uncertainty bounds are model-derived estimates (demo-grade), not guaranteed statistical confidence.",
        },
        "outputs": {
            "predictions_csv": str(preds_path),
            "error_report_csv": str(error_path),
            "uncertainty_csv": str(unc_path),
            "metrics_json": str(metrics_path),
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DEMO COMPLETED ===")
    print("Predictions:", preds_path)
    print("Error report:", error_path)
    print("Uncertainty:", unc_path)
    print("Metrics:", metrics_path)
    print("Summary:", summary_path)
    print(json.dumps(metrics["overall"], indent=2))


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process Control Model demo runner (train -> predict -> evaluate)."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to train_full_70.csv")
    parser.add_argument("--known", type=str, required=True, help="Path to predict_known_30.csv")
    parser.add_argument("--truth", type=str, required=True, help="Path to predict_truth_30.csv")
    parser.add_argument("--out", type=str, default="data/demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_demo(
        train_path=Path(args.train),
        predict_known_path=Path(args.known),
        predict_truth_path=Path(args.truth),
        out_dir=Path(args.out),
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
