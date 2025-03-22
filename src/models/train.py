# ✅ FILE: src/models/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.classification import *
from pathlib import Path
import sys

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import (
    PYCARET_SETUP_CONFIG,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    RAW_DATA_FILE,
    BEST_MODEL_PATH
)

def load_data() -> pd.DataFrame:
    print(f"Loading data from {RAW_DATA_FILE}")
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded {len(df)} records")
    return df

def train_model(data: pd.DataFrame):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        setup(data=data,
              target=PYCARET_SETUP_CONFIG['target'],
              train_size=PYCARET_SETUP_CONFIG['train_size'],
              fold=PYCARET_SETUP_CONFIG['fold'],
              normalize=PYCARET_SETUP_CONFIG['normalize'],
              transformation=PYCARET_SETUP_CONFIG['transformation'],
              ignore_features=PYCARET_SETUP_CONFIG['ignore_features'],
              data_split_shuffle=True, silent=True, verbose=False)

        best_model = compare_models(n_select=1)
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)

        mlflow.log_param("session_id", 123)
        mlflow.log_param("data_split_shuffle", True)
        mlflow.sklearn.log_model(final_model, "best_model")

        save_model(final_model, str(BEST_MODEL_PATH))
        print(f"Model saved to: {BEST_MODEL_PATH}")

def main():
    print("Starting training pipeline...")
    data = load_data()
    train_model(data)

if __name__ == "__main__":
    main()
