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


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    """
    return pd.read_csv(path)

def prepare_datasets(
    df: pd.DataFrame,
    process_name: str = "default",
    val_ratio: float = 0.3,
):
    """
    Split data into train/validation and apply masking.
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
    y_train_params = train_df[dataset.target_params_cols]

    return X_train, X_val, y_train_dist, y_train_params

def run_training(
    data_path: str,
    process_name: str = "default",
):
    """
    Run supervised training for distribution and parameter prediction.
    """
    # 1. load data
    df = load_data(data_path)

    # 2. prepare datasets
    X_train, X_val, y_train_dist, y_train_params = prepare_datasets(
        df=df,
        process_name=process_name,
        val_ratio=0.3,
    )

    # 1. load data
    df = load_data(data_path)

    # 2. build dataset handler
    dataset = PredictionDataset(
    df=df,
    process_name=process_name,
    mask_ratio=0.3,
    random_state=42,
    )

    # 3. split into train / validation (REAL data)
    train_df, val_df = dataset.train_val_split(val_ratio=0.3)

    # 4. apply masking ONLY for model input
    X_train = dataset.apply_mask(train_df)
    X_val = dataset.apply_mask(val_df)

    # 5. ground truth (NO MASK)
    y_train_dist = train_df["distribution"]
    y_train_params = train_df[dataset.target_params_cols]

    # 3. build model
    model = DistributionPredictor(
        process_name=process_name,
        random_state=42,
    )

    # 4. train model
    model.fit(X_train, y_train_dist, y_train_params)

    # 5. quick sanity check prediction
    preds = model.predict(X_val)
        # --- Evaluation ---

    # 1. distribution classification evaluation
    class_names = list(model.distributions)
    dist_eval = evaluate_distribution_classification(
        y_true=val_df["distribution"],
        y_pred_probs=preds["distribution_probs"],
        class_names=class_names,
    )

    print("\nDistribution classification:")
    print("Accuracy:", dist_eval["accuracy"])
    print("Confusion matrix:\n", dist_eval["confusion_matrix"])

    # 2. parameter regression evaluation
    param_eval = evaluate_parameter_regression(
        y_true_params=val_df[model.all_parameters],
        y_pred_params=preds["parameter_predictions"],
    )

    print("\nParameter regression:")
    print("MAE (overall):", param_eval["mae_overall"])
    print("RMSE (overall):", param_eval["rmse_overall"])


    print("Training finished.")
    print("Distribution probabilities shape:", preds["distribution_probs"].shape)
    print("Parameter predictions shape:", preds["parameter_predictions"].shape)

if __name__ == "__main__":
    run_training(
        data_path="data/raw.csv",
        process_name="default",
    )

