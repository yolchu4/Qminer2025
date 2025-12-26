PROCESS_CONFIG = {
    "default": {
        # input features
        "numerical_features": ["x1", "x2", "x3"],
        "categorical_features": ["mode"],

        # output distributions
        "distribution_classes": ["uniform", "normal","exponential","lognormal",],

        # parameter mapping
        "parameter_map" : {
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
