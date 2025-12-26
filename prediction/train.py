"""
Training entry point for supervised distribution and parameter prediction.
"""

import pandas as pd

from prediction.datasets import PredictionDataset
from prediction.model import DistributionPredictor
from prediction.evaluation import (
    evaluate_distribution_classification,
    evaluate_parameter_regression,
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    """
    return pd.read_csv(path)


# --------------------------------------------------
# Dataset preparation (kept for future use)
# --------------------------------------------------
def prepare_datasets(
    df: pd.DataFrame,
    process_name: str = "default",
    val_ratio: float = 0.3,
):
    """
    Split data into train/validation and apply masking.
    (Currently not used in run_training, kept intentionally)
    """

    dataset = PredictionDataset(
        df=df,
        process_name=process_name,
        mask_ratio=0.3,
        random_state=42,
    )

    train_df, val_df = dataset.train_val_split(val_ratio=val_ratio)

    X_train = dataset.apply_mask(train_df)
    X_val = dataset.apply_mask(val_df)

    y_train_dist = train_df["distribution"]

    y_train_params = train_df[dataset.target_params_cols].copy()
    y_val_params = val_df[dataset.target_params_cols].copy()

    # 🔥 NaN FIX (critical for beta / multi-distribution)
    y_train_params = y_train_params.fillna(0.0)
    y_val_params = y_val_params.fillna(0.0)

    return X_train, X_val, y_train_dist, y_train_params, y_val_params


# --------------------------------------------------
# Training entry point
# --------------------------------------------------
def run_training(
    data_path: str,
    process_name: str = "default",
):
    """
    Run supervised training for distribution and parameter prediction.
    """

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    df = load_data(data_path)

    # --------------------------------------------------
    # 2. Build dataset handler
    # --------------------------------------------------
    dataset = PredictionDataset(
        df=df,
        process_name=process_name,
        mask_ratio=0.3,      # only for validation / inference
        random_state=42,
    )

    # --------------------------------------------------
    # 3. Split into train / validation (REAL data)
    # --------------------------------------------------
    train_df, val_df = dataset.train_val_split(val_ratio=0.3)

    # --------------------------------------------------
    # 4. Training data (NO MASK)
    # --------------------------------------------------
    X_train = train_df.copy()

    y_train_dist = train_df["distribution"]

    y_train_params = train_df[dataset.target_params_cols].copy()

    # 🔥 CRITICAL FIX: no NaN allowed in regression targets
    y_train_params = y_train_params.fillna(0.0)

    assert not y_train_params.isna().any().any(), "NaN still exists in y_train_params"

    # --------------------------------------------------
    # 5. Validation data (WITH MASK)
    # --------------------------------------------------
    X_val = dataset.apply_mask(val_df)

    # --------------------------------------------------
    # 6. Build model
    # --------------------------------------------------
    model = DistributionPredictor(
        process_name=process_name,
        random_state=42,
    )

    # --------------------------------------------------
    # 7. Train model
    # --------------------------------------------------
    model.fit(X_train, y_train_dist, y_train_params)

    # --------------------------------------------------
    # 8. Predict on masked validation data
    # --------------------------------------------------
    preds = model.predict(X_val)

    # --------------------------------------------------
    # 9. Evaluation
    # --------------------------------------------------
    print("\n--- EVALUATION START ---")

    # 9.1 Distribution classification
    dist_eval = evaluate_distribution_classification(
        y_true=val_df["distribution"],
        y_pred_probs=preds["distribution_probs"],
        model_classes=model.dist_model.named_steps["clf"].classes_,
    )

    print("\nDistribution classification:")
    print("Accuracy:", dist_eval["accuracy"])
    print("Confusion matrix:\n", dist_eval["confusion_matrix"])

    # 9.2 Parameter regression
    param_eval = evaluate_parameter_regression(
        y_true_params=val_df[model.all_parameters].fillna(0.0),
        y_pred_params=preds["parameter_predictions"],
    )

    print("\nParameter regression:")
    print("MAE (overall):", param_eval["mae_overall"])
    print("RMSE (overall):", param_eval["rmse_overall"])

    print("\n--- EVALUATION END ---")

    # --------------------------------------------------
    # 10. Final info
    # --------------------------------------------------
    print("Training finished.")
    print("Distribution probabilities shape:", preds["distribution_probs"].shape)
    print("Parameter predictions shape:", preds["parameter_predictions"].shape)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    run_training(
        data_path="data/raw/test_10000_multi.csv",
        process_name="default",
    )
