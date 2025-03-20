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
from src.preprocessing.preprocessor import LoanPreprocessor
from src.fairness.metrics import FairnessMetrics

def load_data() -> pd.DataFrame:
    """Load and validate the raw data."""
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
    
    with mlflow.start_run(run_name="pycaret_automl"):
        # Initialize preprocessor
        preprocessor = LoanPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        # Log preprocessing parameters
        mlflow.log_param("preprocessing", "LoanPreprocessor")
        
        # Set up PyCaret
        setup(data=processed_data,
              target=PYCARET_SETUP_CONFIG['target'],
              train_size=PYCARET_SETUP_CONFIG['train_size'],
              fold=PYCARET_SETUP_CONFIG['fold'],
              normalize=PYCARET_SETUP_CONFIG['normalize'],
              transformation=PYCARET_SETUP_CONFIG['transformation'],
              ignore_features=PYCARET_SETUP_CONFIG['ignore_features'],
              silent=True,
              verbose=False)
        
        # Compare models and select best
        best_model = compare_models(n_select=1)
        
        # Finalize model
        final_model = finalize_model(best_model)
        
        # Get predictions for fairness evaluation
        y_pred = predict_model(final_model, data=processed_data)['prediction_label']
        
        # Calculate fairness metrics
        fairness = FairnessMetrics()
        fairness_metrics = fairness.calculate_metrics(
            processed_data[PYCARET_SETUP_CONFIG['target']].values,
            y_pred.values,
            processed_data['income_variability'].values
        )
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": get_metrics()['Accuracy'],
            "auc": get_metrics()['AUC'],
            "precision": get_metrics()['Precision'],
            "recall": get_metrics()['Recall'],
            "f1": get_metrics()['F1'],
            "disparate_impact": fairness_metrics['disparate_impact'],
            "equal_opportunity_diff": fairness_metrics['equal_opportunity_difference']
        })
        
        # Log fairness report
        mlflow.log_text(fairness.get_fairness_report(), "fairness_report.txt")
        
        # Save the model and preprocessor
        save_model(final_model, str(BEST_MODEL_PATH))
        joblib.dump(preprocessor, BEST_MODEL_PATH.parent / "preprocessor.pkl")
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {BEST_MODEL_PATH}")
        print("\nFairness Report:")
        print(fairness.get_fairness_report())

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