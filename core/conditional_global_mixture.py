from __future__ import annotations

import numpy as np
from typing import Dict, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from scipy.optimize import bisect
from scipy.stats import (
    norm, lognorm, gamma as gamma_dist,
    expon, uniform, laplace
)

EPS = 1e-8


class ConditionalGlobalMixtureModel:
    """
    Soft multi-regime, global-distribution-per-regime model.

    U | X = sum_r w_r(X) * D_r( theta_r(X) )
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str] | None = None,
        n_regimes: int = 3,
        random_state: int = 42,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols or []
        self.n_regimes = n_regimes
        self.random_state = random_state

        # Keep demo set compact & stable
        self.distributions = [
            "normal",
            "laplace",
            "lognormal",
            "gamma",
            "exponential",
            # "uniform",
        ]

        self._build_preprocessor()

        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
        )

        # per regime: selected dist + per-target parameter models
        self.regime_models_: List[Dict] = []

    # -----------------------------
    # preprocessing
    # -----------------------------
    def _build_preprocessor(self):
        num = Pipeline([("imp", SimpleImputer(strategy="mean"))])

        if self.categorical_cols:
            cat = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ])
            self.preprocess = ColumnTransformer([
                ("num", num, self.numerical_cols),
                ("cat", cat, self.categorical_cols),
            ])
        else:
            self.preprocess = ColumnTransformer([
                ("num", num, self.numerical_cols),
            ])

    def _regressor(self):
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.08,
            max_iter=60,
            random_state=self.random_state,
        )

    # -----------------------------
    # parameter models
    # -----------------------------
    def _fit_loc_scale(self, X, y):
        loc = Pipeline([("prep", self.preprocess), ("reg", self._regressor())])
        loc.fit(X, y)

        resid = y - loc.predict(X)

        log_s = Pipeline([("prep", self.preprocess), ("reg", self._regressor())])
        log_s.fit(X, np.log(np.abs(resid) + 1e-3))

        return {"loc": loc, "log_s": log_s}

    def _params_loc_scale(self, m, X):
        loc = m["loc"].predict(X)
        # FIX: correct inverse of log(|resid|)
        scale = np.exp(m["log_s"].predict(X)) + 1e-3
        return loc, scale

    # -----------------------------
    # distribution NLL
    # -----------------------------
    def _nll(self, dist, params, X, y):
        mu, s = self._params_loc_scale(params, X)

        if dist == "normal":
            return -np.mean(norm.logpdf(y, mu, s))

        if dist == "laplace":
            return -np.mean(laplace.logpdf(y, mu, s))

        if dist == "exponential":
            # exponential support is y>=0; clamp y for likelihood stability
            scale = np.maximum(mu, 1e-3)
            y_pos = np.maximum(y, 0.0)
            return -np.mean(expon.logpdf(y_pos, scale=scale))

        if dist == "gamma":
            # gamma support is y>0; clamp mean & y for stability
            mu_pos = np.maximum(mu, 1e-3)
            var = s ** 2
            k = (mu_pos ** 2) / (var + EPS)
            theta = var / (mu_pos + EPS)
            k = np.maximum(k, 1e-6)
            theta = np.maximum(theta, 1e-6)
            return -np.mean(gamma_dist.logpdf(np.maximum(y, 1e-6), k, scale=theta))

        if dist == "lognormal":
            # lognormal support is y>0; clamp y, keep parameterization consistent with scipy
            y_pos = np.maximum(y, 1e-6)
            s = np.maximum(s, 1e-6)
            return -np.mean(lognorm.logpdf(y_pos, s=s, scale=np.exp(mu)))

        if dist == "uniform":
            a, b = mu - s, mu + s
            width = np.maximum(b - a, 1e-6)
            return -np.mean(uniform.logpdf(y, a, width))

        raise ValueError(dist)

    # -----------------------------
    # regime CDF helper
    # -----------------------------
    def _regime_cdf(self, dist: str, mu: float, s: float, y: float) -> float | None:
        if dist == "normal":
            return float(norm.cdf(y, loc=mu, scale=max(s, 1e-6)))
        if dist == "laplace":
            return float(laplace.cdf(y, loc=mu, scale=max(s, 1e-6)))
        if dist == "gamma":
            mu_pos = max(mu, 1e-3)
            var = s ** 2
            k = (mu_pos ** 2) / (var + EPS)
            th = var / (mu_pos + EPS)
            k = float(np.maximum(k, 1e-6))
            th = float(np.maximum(th, 1e-6))
            return float(gamma_dist.cdf(max(y, 1e-6), k, scale=th))
        if dist == "lognormal":
            y_pos = float(max(y, 1e-6))
            s_ = float(max(s, 1e-6))
            return float(lognorm.cdf(y_pos, s=s_, scale=float(np.exp(mu))))
        if dist == "exponential":
            return float(expon.cdf(max(y, 0.0), scale=float(max(mu, 1e-3))))
        if dist == "uniform":
            a = mu - s
            b = mu + s
            width = float(max(b - a, 1e-6))
            return float(uniform.cdf(y, loc=float(a), scale=width))
        return None

    # -----------------------------
    # FIT
    # -----------------------------
    def fit(self, X, U):
        Xtr, Xva, Utr, Uva = train_test_split(
            X, U, test_size=0.2, random_state=self.random_state
        )

        # regime weights (on features)
        self.gmm.fit(Xtr)
        Wtr = self.gmm.predict_proba(Xtr)

        # FIX: hard regime assignment so each regime gets its own parameter models
        Z = np.argmax(Wtr, axis=1)

        n_targets = U.shape[1]
        self.regime_models_ = []

        for r in range(self.n_regimes):
            best_dist = None
            best_nll = np.inf
            best_params = None

            idx = (Z == r)
            # fallback if too few samples in a regime
            if int(np.sum(idx)) < 20:
                idx = slice(None)

            for dist in self.distributions:
                params = []
                total = 0.0

                for j in range(n_targets):
                    m = self._fit_loc_scale(Xtr[idx], Utr[idx, j])
                    nll = self._nll(dist, m, Xva, Uva[:, j])
                    total += nll
                    params.append(m)

                if total < best_nll:
                    best_nll = total
                    best_dist = dist
                    best_params = params

            self.regime_models_.append({
                "distribution": best_dist,
                "params": best_params,
            })

        return self

    # -----------------------------
    # mean per regime (distribution-aware)
    # -----------------------------
    def _regime_mean(self, dist: str, mu: float, s: float) -> float:
        if dist in ("normal", "laplace"):
            return float(mu)
        if dist == "lognormal":
            return float(np.exp(mu + 0.5 * s * s))
        if dist == "gamma":
            # constructed from mean/var -> mean is mu (clamped)
            return float(max(mu, 1e-6))
        if dist == "exponential":
            # in this model, exponential scale is derived from mu
            return float(max(mu, 1e-3))
        if dist == "uniform":
            return float(mu)  # symmetric around mu by our parametrization
        return float(mu)

    # -----------------------------
    # PREDICT
    # -----------------------------
    def predict(self, X, q=(0.05, 0.95)):
        """
        Returns:
            mean: [n, n_targets]
            q_low/q_high: mixture quantiles computed by root-finding on mixture CDF
            regime_weights: [n, n_regimes]
            selected_distributions: list[str] length n_regimes
        """
        W = self.gmm.predict_proba(X)
        n, R = W.shape
        n_targets = len(self.regime_models_[0]["params"])

        mean = np.zeros((n, n_targets), dtype=float)
        ql = np.zeros_like(mean)
        qh = np.zeros_like(mean)

        # Precompute per-sample, per-regime, per-target mu/s for speed
        # mu_s[i][r][j] = (mu, s)
        mu_s = [[[None] * n_targets for _ in range(R)] for _ in range(n)]
        for i in range(n):
            Xi = X.iloc[[i]]
            for r in range(R):
                for j in range(n_targets):
                    mu, s = self._params_loc_scale(self.regime_models_[r]["params"][j], Xi)
                    mu_s[i][r][j] = (float(mu[0]), float(s[0]))

        # mean (mixture of regime means)
        for i in range(n):
            for j in range(n_targets):
                m = 0.0
                for r in range(R):
                    dist = self.regime_models_[r]["distribution"]
                    mu, s = mu_s[i][r][j]
                    m += W[i, r] * self._regime_mean(dist, mu, s)
                mean[i, j] = m

        # mixture quantiles via bisection on stabilized mixture CDF
        for i in range(n):
            for j in range(n_targets):
                center = float(mean[i, j])

                # pick a data-dependent initial spread to reduce bracket failures
                scales = [mu_s[i][r][j][1] for r in range(R)]
                spread = float(max(5.0, 6.0 * (np.median(scales) if scales else 1.0)))

                def mix_cdf_minus(y: float, alpha: float) -> float:
                    cdf = 0.0
                    wsum = 0.0
                    for r in range(R):
                        dist = self.regime_models_[r]["distribution"]
                        mu, s = mu_s[i][r][j]
                        F = self._regime_cdf(dist, mu, s, y)
                        if F is None:
                            continue
                        wr = W[i, r]
                        cdf += wr * F
                        wsum += wr

                    if wsum < 1e-12:
                        # degenerate fallback
                        return y - center

                    # Stabilize to a proper CDF even if some regimes skipped
                    return (cdf / wsum) - alpha

                # Expand bracket to try to ensure sign change
                def find_bracket(alpha: float):
                    lo = center - spread
                    hi = center + spread
                    f_lo = mix_cdf_minus(lo, alpha)
                    f_hi = mix_cdf_minus(hi, alpha)

                    tries = 0
                    local_spread = spread
                    while f_lo * f_hi > 0 and tries < 8:
                        local_spread *= 2.0
                        lo = center - local_spread
                        hi = center + local_spread
                        f_lo = mix_cdf_minus(lo, alpha)
                        f_hi = mix_cdf_minus(hi, alpha)
                        tries += 1

                    return lo, hi, f_lo, f_hi

                # ---- q_low
                lo, hi, f_lo, f_hi = find_bracket(q[0])

                # FIX: guard bisect when no sign change -> fallback (prevents crash)
                if f_lo * f_hi > 0:
                    ql[i, j] = center
                else:
                    ql[i, j] = bisect(lambda y: mix_cdf_minus(y, q[0]), lo, hi, maxiter=60)

                # ---- q_high
                lo, hi, f_lo, f_hi = find_bracket(q[1])

                # FIX: guard bisect when no sign change -> fallback (prevents crash)
                if f_lo * f_hi > 0:
                    qh[i, j] = center
                else:
                    qh[i, j] = bisect(lambda y: mix_cdf_minus(y, q[1]), lo, hi, maxiter=60)

        return {
            "mean": mean,
            "q_low": ql,
            "q_high": qh,
            "regime_weights": W,
            "selected_distributions": [m["distribution"] for m in self.regime_models_],
        }
