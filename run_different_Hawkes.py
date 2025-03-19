import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models import ConformalQR, GradientBoostingQR, LinearQR, RandomForestQR, mlpQR, lstmQR, TransformerQR
from utils.data import load_event_data, generate_X_y_multitype, pad_sequences
from utils.eval import empirical_coverage, average_interval_size

# Configuration parameters
CONFIG = {
    "data": {
        "path": "",
        "m": 5,
        "n": 5000,
        "n_test": 1,
        "k": 2,
        "period_length": 10,
        "train_size": 4000,
        "calib_size": 1000,
        "test_size": 1
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
    
    reshape_func = lambda x: x.reshape(x.shape[0], -1) if reshape_input else x
    
    cqr.fit(reshape_func(X_train), y_train)
    cqr.calibrate(reshape_func(X_calib), y_calib)
    y_pred_lower, y_pred_upper = cqr.predict(reshape_func(X_test))

    y_pred_lower = np.atleast_1d(y_pred_lower)
    y_pred_upper = np.atleast_1d(y_pred_upper)
    
    return y_pred_lower, y_pred_upper

def run_simulation(file_path):
    """Process a single simulation file and return results"""
    config = CONFIG.copy()
    config["data"]["path"] = file_path
    
    X, y = load_and_preprocess_data(config)
    
    # Dataset splitting
    split_points = [
        CONFIG["data"]["train_size"], 
        CONFIG["data"]["train_size"] + CONFIG["data"]["calib_size"]
    ]
    X_train, X_calib, X_test = np.split(X, split_points)
    y_train, y_calib, y_test = np.split(y, split_points)

    return {
        "test_count": prepare_targets(y_test, CONFIG["data"]["m"])["count"],
        "test_time": prepare_targets(y_test, CONFIG["data"]["m"])["time"],
        "results": process_models(X_train, X_calib, X_test, y_train, y_calib, y_test)
    }

def process_models(X_train, X_calib, X_test, y_train, y_calib, y_test):
    """Process all models for a single simulation"""
    model_configs = {
        "LinearQR": (LinearQR, True),
        "GradientBoostingQR": (GradientBoostingQR, True),
        "RandomForestQR": (RandomForestQR, True),
        "MLP": (mlpQR, True),
        "LSTM": (lstmQR, False),
        "TransformerQR": (TransformerQR, False)
    }

    results = {}
    targets = {
        "train_count": prepare_targets(y_train, CONFIG["data"]["m"])["count"],
        "calib_count": prepare_targets(y_calib, CONFIG["data"]["m"])["count"],
        "train_time": prepare_targets(y_train, CONFIG["data"]["m"])["time"],
        "calib_time": prepare_targets(y_calib, CONFIG["data"]["m"])["time"],
    }

    for model_name, (model_class, needs_reshape) in model_configs.items():
        model_kwargs = CONFIG["model_params"]["model_specific"].get(
            model_name.replace("QR", ""), {}
        ).copy()

        # Event count prediction
        count_lower, count_upper = run_experiment(
            model_class, X_train, targets["train_count"],
            X_calib, targets["calib_count"], X_test, needs_reshape,
            **model_kwargs
        )

        # Event time prediction
        time_lower, time_upper = run_experiment(
            model_class, X_train, targets["train_time"],
            X_calib, targets["calib_time"], X_test, needs_reshape,
            **model_kwargs
        )

        results[model_name] = {
            "count": (count_lower, count_upper),
            "time": (time_lower, time_upper)
        }

    return results

def main():
    """Main execution flow with aggregated evaluation"""
    # Generate file paths
    data_dir = "results2"
    file_paths = [os.path.join(data_dir, f"simulation_{i:03d}.csv") for i in range(1, 101)]
    
    # Verify file existence
    missing = [fp for fp in file_paths if not os.path.exists(fp)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files, e.g.: {missing[:3]}")

    # Initialize storage
    aggregated = {
        model: {
            "count_lower": [],
            "count_upper": [],
            "time_lower": [],
            "time_upper": [],
            "test_count": [],
            "test_time": []
        } 
        for model in ["LinearQR", "GradientBoostingQR", "RandomForestQR", 
                      "MLP", "LSTM", "TransformerQR"]
    }

    # Process all simulations
    for file_path in tqdm(file_paths, desc="Processing simulations"):
        try:
            sim_results = run_simulation(file_path)
            for model_name, model_data in sim_results["results"].items():
                aggregated[model_name]["count_lower"].append(model_data["count"][0])
                aggregated[model_name]["count_upper"].append(model_data["count"][1])
                aggregated[model_name]["time_lower"].append(model_data["time"][0])
                aggregated[model_name]["time_upper"].append(model_data["time"][1])
                aggregated[model_name]["test_count"].extend(sim_results["test_count"])
                aggregated[model_name]["test_time"].extend(sim_results["test_time"])
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # Evaluate results
    evaluation_results = {}
    for model_name, data in aggregated.items():
        # Convert to numpy arrays
        count_lower = np.concatenate(data["count_lower"])
        count_upper = np.concatenate(data["count_upper"])
        time_lower = np.concatenate(data["time_lower"])
        time_upper = np.concatenate(data["time_upper"])
        test_count = np.array(data["test_count"])
        test_time = np.array(data["test_time"])

        evaluation_results[model_name] = {
            "count_coverage": empirical_coverage(test_count, count_lower, count_upper),
            "count_interval": average_interval_size(count_lower, count_upper),
            "time_coverage": empirical_coverage(test_time, time_lower, time_upper),
            "time_interval": average_interval_size(time_lower, time_upper)
        }

    # Print formatted results
    print("\nFinal Evaluation Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"Count Coverage: {metrics['count_coverage']:.3f}")
        print(f"Count Interval: {metrics['count_interval']:.3f}")
        print(f"Time Coverage: {metrics['time_coverage']:.3f}")
        print(f"Time Interval: {metrics['time_interval']:.3f}")

if __name__ == "__main__":
    main()