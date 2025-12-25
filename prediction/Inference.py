"""
Inference utilities for predicting distribution and parameters
from incomplete input data.
"""

import pandas as pd

from prediction.model import DistributionPredictor
from prediction.config import PROCESS_CONFIG


def run_inference(
    model: DistributionPredictor,
    df_incomplete: pd.DataFrame,
):
    """
    Run inference on incomplete input data.
    """
    preds = model.predict(df_incomplete)

    return {
        "distribution_probs": preds["distribution_probs"],
        "parameter_predictions": preds["parameter_predictions"],
    }


def make_decision(
    distribution_probs,
    parameter_predictions,
    process_name: str = "default",
):
    """
    Convert raw model outputs into human-readable decisions.
    """
    cfg = PROCESS_CONFIG[process_name]
    distribution_classes = list(cfg["parameter_map"].keys())
    parameter_map = cfg["parameter_map"]

    decisions = []

    for i in range(len(distribution_probs)):
        # 1. choose most likely distribution
        dist_idx = distribution_probs[i].argmax()
        chosen_dist = distribution_classes[dist_idx]

        # 2. extract relevant parameters
        params = {}
        offset = 0
        for dist, param_names in parameter_map.items():
            if dist == chosen_dist:
                for j, name in enumerate(param_names):
                    params[name] = float(parameter_predictions[i][offset + j])
                break
            offset += len(param_names)

        decisions.append(
            {
                "distribution": chosen_dist,
                "parameters": params,
            }
        )

    return decisions
