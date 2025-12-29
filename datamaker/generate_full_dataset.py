"""
Process Control Demo - Raw Data Generator (Nonlinear + Dynamic)

Generates a realistic time-series dataset for a chemical process control problem:
- Nonlinear relations (Arrhenius-like term, saturation, interactions)
- Dynamic behavior (AR(1) / lag influence)
- Noise + slow drift
- Includes implicit coverage of multiple distributions:
  Normal, Lognormal, Beta, Truncated, Uniform, Exponential, Gamma, Mixture/Regime

Output:
    data/full_dataset.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
N_SAMPLES = 1000
TIME_STEP_HOURS = 1
RANDOM_SEED = 42

OUTPUT_PATH = "data/full_dataset.csv"


# -----------------------------
# Column Definitions (DEMO CONTRACT)
# -----------------------------
TIME_COL = "timestamp"

KNOWN_COLS = [
    "inlet_flow_A",
    "inlet_flow_B",
    "inlet_flow_C",
    "inlet_conc_A",
    "inlet_conc_B",
    "inlet_conc_C",
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


# -----------------------------
# Ranges (FINAL ENFORCEMENT)
# -----------------------------
# این رنج‌ها «قرارداد فیزیکی/عملیاتی» هستند.
# حتی اگر در فرمول‌ها یا نویزها تغییر ایجاد شود، خروجی نهایی همیشه در این رنج‌ها می‌ماند.
RANGES = {
    # KNOWN
    "inlet_flow_A": (2.0, 40.0),
    "inlet_flow_B": (2.0, 40.0),
    "inlet_flow_C": (1.0, 30.0),

    "inlet_conc_A": (0.02, 0.98),
    "inlet_conc_B": (0.02, 0.98),
    "inlet_conc_C": (0.02, 0.98),

    "outlet_flow_1": (0.5, 35.0),
    "outlet_flow_2": (0.5, 35.0),

    "outlet_conc_1": (0.01, 0.99),
    "outlet_conc_2": (0.01, 0.99),

    # UNKNOWN
    "reactor_temperature": (300.0, 380.0),
    "reactor_pressure": (5.0, 14.0),
    "agitator_speed": (80.0, 420.0),
    "reactant_feed_flow": (1.0, 20.0),
    "reactant_feed_concentration": (0.02, 0.98),
}


# -----------------------------
# Helpers
# -----------------------------
def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x

def _smooth_drift(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, scale, size=n)
    return np.cumsum(steps)

def apply_ranges(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Enforce final operational ranges on all defined columns.
    This is the last safety layer to prevent out-of-range values.
    """
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# -----------------------------
# Core Generator
# -----------------------------
def generate_process_control_data(n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    # Time axis
    timestamp = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq=f"{TIME_STEP_HOURS}H",
    )

    t = np.arange(n_samples)

    # Uniform disturbance (explicit uniform)
    uniform_noise = rng.uniform(-1.0, 1.0, size=n_samples)

    # ======= Inputs (disturbances) =======
    daily = np.sin(2 * np.pi * t / 24.0)
    weekly = np.sin(2 * np.pi * t / (24.0 * 7.0))

    inlet_flow_A = rng.lognormal(mean=2.25, sigma=0.28, size=n_samples) * (1.0 + 0.08 * daily + 0.05 * weekly)
    inlet_flow_B = rng.lognormal(mean=2.05, sigma=0.32, size=n_samples) * (1.0 + 0.06 * daily - 0.04 * weekly)
    inlet_flow_C = rng.lognormal(mean=1.95, sigma=0.22, size=n_samples) * (1.0 + 0.04 * daily + 0.03 * weekly)

    inlet_flow_A = _clip(inlet_flow_A, 2.0, 40.0)
    inlet_flow_B = _clip(inlet_flow_B, 2.0, 40.0)
    inlet_flow_C = _clip(inlet_flow_C, 1.0, 30.0)

    inlet_flow_C = inlet_flow_C * (1.0 + 0.03 * uniform_noise)

    cA = rng.beta(a=2.2, b=4.8, size=n_samples)
    cB = rng.beta(a=3.0, b=3.8, size=n_samples)
    cC = rng.beta(a=2.6, b=3.6, size=n_samples)

    cA = _clip(cA + 0.06 * _ar1_noise(n_samples, phi=0.92, sigma=0.12, rng=rng), 0.02, 0.98)
    cB = _clip(cB + 0.05 * _ar1_noise(n_samples, phi=0.90, sigma=0.10, rng=rng), 0.02, 0.98)
    cC = _clip(cC + 0.05 * _ar1_noise(n_samples, phi=0.88, sigma=0.10, rng=rng), 0.02, 0.98)

    inlet_conc_A, inlet_conc_B, inlet_conc_C = cA, cB, cC

    # Regime / mixture
    regime = (np.sin(2 * np.pi * t / (24.0 * 10.0)) > 0).astype(float)
    quality_factor = 1.0 + 0.25 * regime + 0.08 * _smooth_drift(n_samples, scale=0.002, rng=rng)
    quality_factor = _clip(quality_factor, 0.7, 1.5)

    # ======= Controls (Unknown) =======
    u_noise_T = _ar1_noise(n_samples, phi=0.85, sigma=1.8, rng=rng)
    u_noise_P = _ar1_noise(n_samples, phi=0.82, sigma=0.25, rng=rng)
    u_noise_S = _ar1_noise(n_samples, phi=0.80, sigma=6.0, rng=rng)

    mix_index = (
        0.55 * inlet_conc_A
        + 0.30 * inlet_conc_B
        + 0.15 * inlet_conc_C
        + 0.10 * (inlet_conc_A * inlet_conc_B)
    )
    mix_index = _clip(mix_index, 0.0, 1.2)

    base_feed_flow = 0.55 * inlet_flow_A + 0.35 * inlet_flow_B + 0.15 * inlet_flow_C
    reactant_feed_flow = 6.0 * np.tanh(base_feed_flow / 15.0) * quality_factor + rng.normal(0, 0.8, n_samples)
    reactant_feed_flow = _clip(reactant_feed_flow, 1.0, 20.0)

    reactant_feed_concentration = _clip(
        0.40 * inlet_conc_A + 0.35 * inlet_conc_B + 0.25 * inlet_conc_C
        + 0.10 * _sigmoid(4 * (inlet_conc_A - inlet_conc_B))
        + rng.normal(0, 0.03, n_samples),
        0.02, 0.98
    )

    reactor_temperature = np.zeros(n_samples)
    reactor_temperature[0] = 330.0 + 8.0 * mix_index[0] + 10.0 * (quality_factor[0] - 1.0) + u_noise_T[0]
    for i in range(1, n_samples):
        target_T = (
            330.0
            + 18.0 * mix_index[i]
            + 14.0 * (quality_factor[i] - 1.0)
            + 6.0 * np.tanh((reactant_feed_flow[i] - 8.0) / 6.0)
        )
        reactor_temperature[i] = 0.88 * reactor_temperature[i - 1] + 0.12 * target_T + u_noise_T[i]
    reactor_temperature = _clip(reactor_temperature, 300.0, 380.0)

    total_in = inlet_flow_A + inlet_flow_B + inlet_flow_C

    # Gamma variability (explicit gamma)
    gamma_var = rng.gamma(shape=2.0, scale=0.35, size=n_samples)
    gamma_var = np.clip(gamma_var, 0.0, 2.0)

    reactor_pressure = (
        7.5
        + 0.010 * total_in
        + 0.0012 * (reactor_temperature - 330.0) ** 2 / 100.0
        + 0.25 * np.tanh((total_in - 40.0) / 10.0)
        + 0.15 * gamma_var
        + u_noise_P
    )
    reactor_pressure = _clip(reactor_pressure, 5.0, 14.0)

    viscosity_proxy = 1.0 + 1.8 * (inlet_conc_C ** 1.4) + 0.8 * (mix_index ** 1.2)
    agitator_speed = (
        180.0
        + 55.0 * np.tanh((viscosity_proxy - 1.5) / 0.7)
        + 25.0 * (quality_factor - 1.0)
        + u_noise_S
    )
    agitator_speed = _clip(agitator_speed, 80.0, 420.0)

    # ======= Outputs =======
    kT = _clip(np.exp(0.018 * (reactor_temperature - 330.0)), 0.2, 8.0)

    residence = 1.0 / (1.0 + (total_in / 50.0)) * (1.0 + 0.04 * (reactor_pressure - 8.0))
    residence = _clip(residence, 0.2, 1.5)

    mixing_eff = np.tanh((agitator_speed / 220.0)) * np.tanh((reactant_feed_flow / 10.0))
    mixing_eff = _clip(mixing_eff, 0.1, 1.0)

    conversion_score = (
        1.8 * np.log1p(kT)
        + 1.2 * mixing_eff
        + 0.7 * residence
        + 0.9 * (reactant_feed_concentration - 0.4)
        - 0.25 * np.maximum(0.0, reactor_pressure - 11.0)
        - 0.35 * (quality_factor - 1.0)
    )
    conversion = _clip(_sigmoid(conversion_score), 0.02, 0.98)

    outlet_conc_1 = _clip(
        0.15 + 0.75 * conversion * reactant_feed_concentration
        + 0.10 * np.sqrt(inlet_conc_A * inlet_conc_B)
        + rng.normal(0, 0.02, n_samples),
        0.01, 0.99
    )
    outlet_conc_2 = _clip(
        0.10 + 0.65 * (conversion ** 0.85) * reactant_feed_concentration
        + 0.08 * np.sqrt(inlet_conc_B * inlet_conc_C)
        + rng.normal(0, 0.02, n_samples),
        0.01, 0.99
    )

    # Exponential shock (explicit exponential)
    exp_shock = rng.exponential(scale=0.6, size=n_samples)
    exp_shock = np.clip(exp_shock, 0.0, 3.0)

    gas_term = (
        3.0 * np.tanh((reactor_temperature - 335.0) / 12.0)
        * np.tanh((conversion - 0.4) / 0.2)
        + 0.4 * exp_shock
    )
    gas_term = _clip(gas_term, -2.0, 6.0)

    outlet_flow_1 = (
        0.78 * reactant_feed_flow
        + 0.06 * np.tanh(agitator_speed / 200.0) * total_in
        - 0.12 * np.tanh((reactor_pressure - 9.0) / 2.0) * 5.0
        + gas_term
        + rng.normal(0, 1.8, n_samples)
    )
    outlet_flow_2 = (
        0.62 * reactant_feed_flow
        + 0.04 * (reactor_temperature - 320.0)
        + 0.03 * np.tanh((total_in - 35.0) / 10.0) * total_in
        + 0.6 * gas_term
        + rng.normal(0, 1.8, n_samples)
    )

    outlet_flow_1 = _clip(outlet_flow_1, 0.5, 35.0)
    outlet_flow_2 = _clip(outlet_flow_2, 0.5, 35.0)

    df = pd.DataFrame({
        TIME_COL: timestamp,

        "inlet_flow_A": inlet_flow_A,
        "inlet_flow_B": inlet_flow_B,
        "inlet_flow_C": inlet_flow_C,
        "inlet_conc_A": inlet_conc_A,
        "inlet_conc_B": inlet_conc_B,
        "inlet_conc_C": inlet_conc_C,

        "reactor_temperature": reactor_temperature,
        "reactor_pressure": reactor_pressure,
        "agitator_speed": agitator_speed,
        "reactant_feed_flow": reactant_feed_flow,
        "reactant_feed_concentration": reactant_feed_concentration,

        "outlet_flow_1": outlet_flow_1,
        "outlet_flow_2": outlet_flow_2,
        "outlet_conc_1": outlet_conc_1,
        "outlet_conc_2": outlet_conc_2,
    })

    # ---- Final enforcement (single source of truth) ----
    df = apply_ranges(df, RANGES)

    return df


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_process_control_data(N_SAMPLES)
    df.to_csv(output_path, index=False)

    print("\n--- FULL DATASET GENERATED ---")
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("-----------------------------------------")
