import numpy as np
from typing import Dict

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import PROCESS_CONFIG
class DistributionPredictor:
    def __init__(
        self,
        process_name: str = "default",
        hidden_layer_sizes=(64, 64),
        random_state: int = 42,
    ):
        if process_name not in PROCESS_CONFIG:
            raise ValueError(f"Unknown process: {process_name}")

        self.cfg = PROCESS_CONFIG[process_name]

        self.numerical_cols = self.cfg["numerical_features"]
        self.categorical_cols = self.cfg["categorical_features"]

        self.distributions = list(self.cfg["parameter_map"].keys())
        self.all_parameters = []
        for p in self.cfg["parameter_map"].values():
            self.all_parameters.extend(p)

        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes

        self._build_models()
