"""
FastAPI service for loan approval predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import BEST_MODEL_PATH

app = FastAPI(
    title="Fair Loan Approval API",
    description="API for fair loan approval predictions with bias mitigation",
    version="1.0.0"
)

class LoanApplication(BaseModel):
    """Loan application input schema."""
    age: int
    income: float
    loan_amount: float
    loan_term: int
    credit_score: int
    employment_status: str
    loan_purpose: str
    existing_loans: int
    income_variability: str

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    loan_approved: bool
    approval_probability: float
    fairness_metrics: dict

# Set the MLflow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')

# Load model from MLflow
try:
    model_uri = 'runs:/263669e8bdaa41ff8f7ffed5dfdf513d/best_model'
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Error loading model from MLflow: {str(e)}")
    sys.exit(1)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fair Loan Approval API",
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """
    Make a loan approval prediction.
    
    Args:
        application: Loan application data
        
    Returns:
        Prediction response with approval decision and fairness metrics
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Directly use the input data without preprocessing
        processed_input = input_data
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Check if the model has predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_input)[0]
            approval_probability = float(probabilities[1])
        else:
            # If predict_proba is not available, use a default probability
            approval_probability = float(prediction)
        
        # Get fairness metrics for the protected group
        fairness_metrics = {
            "group": "High Income Variability" if application.income_variability == "High" else "Low Income Variability",
            "group_approval_rate": approval_probability
        }
        
        return PredictionResponse(
            loan_approved=bool(prediction),
            approval_probability=approval_probability,
            fairness_metrics=fairness_metrics
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 