import pandas as pd

TIME_COL = "timestamp"

KNOWN_COLS = [
    "inlet_flow_A","inlet_flow_B","inlet_flow_C",
    "inlet_conc_A","inlet_conc_B","inlet_conc_C",
    "outlet_flow_1","outlet_flow_2",
    "outlet_conc_1","outlet_conc_2",
]

UNKNOWN_COLS = [
    "reactor_temperature",
    "reactor_pressure",
    "agitator_speed",
    "reactant_feed_flow",
    "reactant_feed_concentration",
]

def _check(df, cols, name):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"{name}: missing column '{c}'")

def validate_datasets(paths: dict):
    for name, path in paths.items():
        df = pd.read_csv(path)

        if TIME_COL not in df.columns:
            raise ValueError(f"{name}: missing '{TIME_COL}'")

        _check(df, KNOWN_COLS, name)

        if name == "predict":
            continue

        _check(df, UNKNOWN_COLS, name)
