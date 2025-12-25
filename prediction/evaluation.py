"""
Evaluation utilities for distribution classification
and parameter regression.
"""

import numpy as np
def evaluate_distribution_classification(
    y_true,
    y_pred_probs,
    class_names,
):
    """
    Evaluate distribution classification performance.

    Parameters
    ----------
    y_true : array-like
        True distribution labels.
    y_pred_probs : ndarray
        Predicted probabilities for each distribution.
    class_names : list
        Ordered list of distribution class names.

    Returns
    -------
    dict
        Accuracy and confusion matrix.
    """
    # predicted class index
    y_pred_idx = y_pred_probs.argmax(axis=1)

    # map class names to indices
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y_true_idx = [class_to_idx[y] for y in y_true]

    # accuracy
    accuracy = np.mean(np.array(y_true_idx) == y_pred_idx)

    # confusion matrix
    n_classes = len(class_names)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true_idx, y_pred_idx):
        conf_matrix[t, p] += 1

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
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
