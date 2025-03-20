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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import BEST_MODEL_PATH
from src.preprocessing.preprocessor import LoanPreprocessor

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

# Load model and preprocessor
try:
    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(BEST_MODEL_PATH.parent / "preprocessor.pkl")
except Exception as e:
    print(f"Error loading model: {str(e)}")
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
        
        # Preprocess input
        processed_input = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        probabilities = model.predict_proba(processed_input)[0]
        
        # Get fairness metrics for the protected group
        fairness_metrics = {
            "group": "High Income Variability" if application.income_variability == "High" else "Low Income Variability",
            "group_approval_rate": float(probabilities[1])
        }
        
        return PredictionResponse(
            loan_approved=bool(prediction),
            approval_probability=float(probabilities[1]),
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
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 