"""
Evaluation utilities for distribution classification
and parameter regression.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
def evaluate_distribution_classification(
    y_true,
    y_pred_probs,
    model_classes,
):
    """
    Evaluate distribution classification using model class order.
    """

    # map true labels to indices using model class order
    class_to_index = {c: i for i, c in enumerate(model_classes)}
    y_true_idx = y_true.map(class_to_index).values

    # predicted indices
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }

def evaluate_parameter_regression(
    y_true_params,
    y_pred_params,
):
    """
    Evaluate regression performance for distribution parameters.

    Parameters
    ----------
    y_true_params : ndarray or DataFrame
        True parameter values.
    y_pred_params : ndarray
        Predicted parameter values.

    Returns
    -------
    dict
        MAE and RMSE for each parameter and overall.
    """
    y_true = np.asarray(y_true_params)
    y_pred = np.asarray(y_pred_params)

    errors = y_pred - y_true

    mae_per_param = np.mean(np.abs(errors), axis=0)
    rmse_per_param = np.sqrt(np.mean(errors ** 2, axis=0))

    mae_overall = np.mean(mae_per_param)
    rmse_overall = np.mean(rmse_per_param)

    return {
        "mae_per_param": mae_per_param,
        "rmse_per_param": rmse_per_param,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
    }
