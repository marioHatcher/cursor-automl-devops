"""
Configuration settings for the ML pipeline.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data settings
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = DATA_DIR / "Data.csv"

# MLflow settings
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "loan_approval_fairness"

# Model settings
MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

# PyCaret setup configuration
PYCARET_SETUP_CONFIG = {
    "target": "loan_status",
    "train_size": 0.8,
    "fold": 5,
    "normalize": True,
    "transformation": True,
    "ignore_features": ["id"],
    "categorical_features": [
        "employment_status",
        "loan_purpose",
        "income_variability"
    ],
    "numeric_features": [
        "age",
        "income",
        "loan_amount",
        "loan_term",
        "credit_score",
        "existing_loans"
    ]
}

# Fairness settings
PROTECTED_ATTRIBUTE = "income_variability"
FAVORABLE_CLASSES = [1]  # Loan approved
PRIVILEGED_GROUPS = [{"income_variability": "Low"}]
UNPRIVILEGED_GROUPS = [{"income_variability": "High"}]

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) 