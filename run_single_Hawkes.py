import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from models import ConformalQR, GradientBoostingQR, LinearQR, RandomForestQR, mlpQR, lstmQR, TransformerQR
from utils.data import load_event_data, generate_X_y_multitype, pad_sequences
from utils.eval import empirical_coverage, average_interval_size

# Configuration parameters
CONFIG = {
    "data": {
        "path": "1dim_Hawkes.csv",
        "m": 1,                  # Number of event types
        "n": 5000,               # Training samples
        "n_test": 100,           # Test samples
        "k": 2,                  # Historical periods used for prediction
        "period_length": 10,
        "train_size": 4000,
        "calib_size": 1000,
        "test_size": 100
    },
    "model_params": {
        "alpha": 0.05,
        "seed": 0,
        "model_specific": {
            "RandomForest": {"max_depth": 20, 
                             "min_samples_split": 10, 
                             "min_samples_leaf": 5},
            "MLP": {"batch_size": 128, 
                    "learning_rate": 1e-2, 
                    "epochs": 300},
            "LSTM": {"batch_size": 128, 
                     "learning_rate": 1e-2, 
                     "epochs": 50, 
                     "hidden_size": 128, 
                     "num_layers": 1},
            "Transformer": {"batch_size": 128, 
                            "learning_rate": 1e-3, 
                            "epochs": 30, 
                            "hidden_size": 128, 
                            "num_layers": 1, 
                            "nhead": 4, 
                            "dropout": 0.1}
        }
    }
}

def load_and_preprocess_data(config):
    """Load and preprocess event data"""
    if not os.path.exists(config["data"]["path"]):
        raise FileNotFoundError(f"Data file missing: {config['data']['path']}")
    
    event_times = load_event_data(config["data"]["path"])
    X, y = generate_X_y_multitype(
        event_times,
        config["data"]["m"],
        0,
        config["data"]["n"] + config["data"]["k"] + config["data"]["n_test"],
        config["data"]["k"],
        config["data"]["period_length"]
    )
    return pad_sequences(X, config["data"]["m"]), y

def prepare_targets(y_data, m):
    """Prepare target variables for both prediction tasks"""
    return {
        "count": np.sum(y_data[:, 0:m], axis=1),
        "time": y_data[:, -2]
    }

def run_experiment(model_class, X_train, y_train, X_calib, y_calib, 
                 X_test, reshape_input=False, **model_kwargs):
    """Run full CQR pipeline for a given model"""
    cqr = ConformalQR(
        Model=model_class,
        alpha=CONFIG["model_params"]["alpha"],
        seed=CONFIG["model_params"]["seed"],
        model_kwargs=model_kwargs
    )
    
    # Reshape data if needed (for non-sequential models)
    reshape_func = lambda x: x.reshape(x.shape[0], -1) if reshape_input else x
    
    cqr.fit(reshape_func(X_train), y_train)
    cqr.calibrate(reshape_func(X_calib), y_calib)
    y_pred_lower, y_pred_upper = cqr.predict(reshape_func(X_test))
    
    return y_pred_lower, y_pred_upper

def main():
    # Load and split data
    X, y = load_and_preprocess_data(CONFIG)
    split = CONFIG["data"]
    
    # Dataset splitting
    split_points = [
        split["train_size"], 
        split["train_size"] + split["calib_size"]
    ]
    X_train, X_calib, X_test = np.split(X, split_points)
    y_train, y_calib, y_test = np.split(y, split_points)

    # Prepare targets
    targets = {
        "train": prepare_targets(y_train, CONFIG["data"]["m"]),
        "calib": prepare_targets(y_calib, CONFIG["data"]["m"]),
        "test": prepare_targets(y_test, CONFIG["data"]["m"])
    }

    # Model experiments
    models = {
        "LinearQR": (LinearQR, True),
        "GradientBoostingQR": (GradientBoostingQR, True),
        "RandomForestQR": (RandomForestQR, True),
        "MLP": (mlpQR, True),
        "LSTM": (lstmQR, False),
        "TransformerQR": (TransformerQR, False)
    }

    for model_name, (model_class, needs_reshape) in models.items():
        print(f"\n=== Running {model_name} ===")
        
        # Get model-specific parameters
        model_kwargs = CONFIG["model_params"]["model_specific"].get(
            model_name.replace("QR", ""),
            {}
        ).copy()

        # Run for event count prediction
        lower, upper = run_experiment(
            model_class, 
            X_train, targets["train"]["count"],
            X_calib, targets["calib"]["count"],
            X_test, needs_reshape,
            **model_kwargs
        )
        print(f"\nEvent Count Results ({model_name}):")
        print(f"Coverage: {empirical_coverage(targets['test']['count'], lower, upper)}")
        print(f"Interval Size: {average_interval_size(lower, upper)}")

        # Run for event time prediction
        lower, upper = run_experiment(
            model_class,
            X_train, targets["train"]["time"],
            X_calib, targets["calib"]["time"],
            X_test, needs_reshape,
            **model_kwargs
        )
        print(f"\nEvent Time Results ({model_name}):")
        print(f"Coverage: {empirical_coverage(targets['test']['time'], lower, upper)}")
        print(f"Interval Size: {average_interval_size(lower, upper)}")

if __name__ == "__main__":
    main()