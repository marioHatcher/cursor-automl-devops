"""
Model training module using PyCaret and MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pycaret.classification import *
from pathlib import Path
import joblib
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import (
    PYCARET_SETUP_CONFIG,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    RAW_DATA_FILE,
    BEST_MODEL_PATH
)
from src.fairness.metrics import FairnessMetrics

def load_data() -> pd.DataFrame:
    """Load and validate the raw data."""
    print(f"Loading data from {RAW_DATA_FILE}")
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {RAW_DATA_FILE}")
    
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded {len(df)} records from {RAW_DATA_FILE}")
    return df

def train_model(data: pd.DataFrame):
    """
    Train the model using PyCaret with MLflow tracking.
    
    Args:
        data: Input DataFrame with preprocessed features
    """
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    try:
        # Directly use the data without preprocessing
        processed_data = data
        
        # Log preprocessing parameters
        mlflow.log_param("preprocessing", "None")
        
        # Set up PyCaret
        setup(data=processed_data,
              target=PYCARET_SETUP_CONFIG['target'],
              train_size=PYCARET_SETUP_CONFIG['train_size'],
              fold=PYCARET_SETUP_CONFIG['fold'],
              normalize=PYCARET_SETUP_CONFIG['normalize'],
              transformation=PYCARET_SETUP_CONFIG['transformation'],
              ignore_features=PYCARET_SETUP_CONFIG['ignore_features'],
              data_split_shuffle=True,)
        
        # Compare models and select best
        best_model = compare_models(n_select=1)
        
        # Tune model
        tuned_model = tune_model(best_model)
        
        # Finalize model
        final_model = finalize_model(tuned_model)
        
        # Log model
        mlflow.sklearn.log_model(final_model, "best_model")
        mlflow.log_param("session_id", 123)
        mlflow.log_param("data_split_shuffle", True)
        
        # Save the model
        save_model(final_model, str(BEST_MODEL_PATH))
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {BEST_MODEL_PATH}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

def main():
    """Main training pipeline."""
    print("Starting model training pipeline...")
    
    try:
        # Load data
        data = load_data()
        
        # Train model
        train_model(data)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
