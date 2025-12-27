PROCESS_CONFIG = {
    "process_control_demo": {
        # --------------------------------------------------
        # Input features (KNOWN parameters)
        # --------------------------------------------------
        # IMPORTANT:
        # The FIRST numerical feature is used as the
        # reference feature for distribution classification
        "numerical_features": [
            # Process Outputs (often most informative → put first)
            "outlet_flow_1",

            # Process Inputs
            "inlet_flow_A",
            "inlet_flow_B",
            "inlet_flow_C",
            "inlet_conc_A",
            "inlet_conc_B",
            "inlet_conc_C",

            # Remaining Outputs
            "outlet_flow_2",
            "outlet_conc_1",
            "outlet_conc_2",
        ],

        # No categorical features in this demo
        "categorical_features": [],

        # --------------------------------------------------
        # Distribution classes supported by the model
        # --------------------------------------------------
        "distribution_classes": [
            "uniform",
            "normal",
            "exponential",
            "lognormal",
            "gamma",
            "weibull",
            "chiSquared",
            "beta",
        ],

        # --------------------------------------------------
        # Parameter mapping per distribution
        # (USED INTERNALLY — NOT USER OUTPUT)
        # --------------------------------------------------
        "parameter_map": {
            "uniform": ["low", "high"],
            "normal": ["mu", "sigma"],
            "exponential": ["lambda"],
            "lognormal": ["mu", "sigma"],
            "gamma": ["k", "theta"],
            "weibull": ["k", "lambda_w"],
            "chiSquared": ["df"],
            "beta": ["alpha", "beta"],
        },
    }
}

