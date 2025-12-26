import numpy as np
import pandas as pd
from typing import Tuple

from .config import PROCESS_CONFIG


class PredictionDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        process_name: str = "default",
        mask_ratio: float = 0.3,
        random_state: int = 42,
    ):
        if process_name not in PROCESS_CONFIG:
            raise ValueError(f"Unknown process: {process_name}")

        self.cfg = PROCESS_CONFIG[process_name]
        self.df = df.copy()

        self.numerical_cols = self.cfg["numerical_features"]
        self.categorical_cols = self.cfg["categorical_features"]
        self.target_dist_col = "distribution"
        self.target_params_cols = []

        for params in self.cfg["parameter_map"].values():
            self.target_params_cols.extend(params)

        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(random_state)

    def train_val_split(self, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        indices = np.arange(len(self.df))
        self.rng.shuffle(indices)

        split = int(len(indices) * (1 - val_ratio))
        train_idx, val_idx = indices[:split], indices[split:]

        return self.df.iloc[train_idx], self.df.iloc[val_idx]

    def apply_mask(self, df):
        df_masked = df.copy()

        # build feature list explicitly
        feature_cols = self.numerical_cols + self.categorical_cols

        for col in feature_cols:
        # IMPORTANT: never mask x1 (key decision feature)
            if col == "x1":
                continue

        if np.random.rand() < self.mask_ratio:
            df_masked.loc[:, col] = np.nan

        return df_masked

    

