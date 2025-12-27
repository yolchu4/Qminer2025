import numpy as np
from typing import Dict

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from prediction.config import PROCESS_CONFIG


class DistributionPredictor:
    """
    Predict distribution class and corresponding parameters
    from process input/output data.
    """

    def __init__(
        self,
        process_name: str = "default",
        hidden_layer_sizes=(64, 64),
        random_state: int = 42,
    ):
        if process_name not in PROCESS_CONFIG:
            raise ValueError(f"Unknown process name: {process_name}")

        self.cfg = PROCESS_CONFIG[process_name]

        # input features
        self.numerical_cols = self.cfg["numerical_features"]
        self.categorical_cols = self.cfg["categorical_features"]

        if len(self.numerical_cols) == 0:
            raise ValueError("At least one numerical feature is required.")

        # IMPORTANT:
        # classifier only sees ONE reference numerical feature
        self.clf_numeric_col = self.numerical_cols[0]

        # output structure
        self.distributions = list(self.cfg["parameter_map"].keys())

        self.all_parameters = []
        for params in self.cfg["parameter_map"].values():
            self.all_parameters.extend(params)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        self._build_models()

    def _build_models(self):
        """Build preprocessing and prediction models."""

        # ---------- shared transformers ----------
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # ---------- classification preprocessor ----------
        # classifier sees ONLY one reference numeric feature (+ categorical)
        clf_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, [self.clf_numeric_col]),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        # ---------- regression preprocessor ----------
        # regressor sees ALL numerical features (+ categorical)
        reg_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        # ---------- distribution classifier ----------
        self.dist_model = Pipeline(
            steps=[
                ("preprocess", clf_preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # ---------- parameter regressor ----------
        self.param_model = Pipeline(
            steps=[
                ("preprocess", reg_preprocessor),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        random_state=self.random_state,
                        max_iter=500,
                    ),
                ),
            ]
        )

    def fit(self, X, y_dist, y_params):
        """
        Train both classifier and regressor.

        X : pandas.DataFrame
        y_dist : array-like (n_samples, n_targets)
        y_params : array-like (n_samples, n_targets, n_params)
        """
        self.dist_model.fit(X, y_dist)
        self.param_model.fit(X, y_params)

    def predict(self, X) -> Dict[str, np.ndarray]:
        """
        Predict distribution probabilities and control parameters.
        """
        dist_probs = self.dist_model.predict_proba(X)
        param_preds = self.param_model.predict(X)

        return {
            "distribution_probs": dist_probs,
            "parameter_predictions": param_preds,
        }
