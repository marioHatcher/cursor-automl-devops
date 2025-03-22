# ✅ FILE: src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys
from pycaret.classification import load_model

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import BEST_MODEL_PATH

app = FastAPI(
    title="Fair Loan Approval API",
    description="API for fair loan approval predictions",
    version="1.0.0"
)

class LoanApplication(BaseModel):
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
    loan_approved: bool
    approval_probability: float
    fairness_metrics: dict

try:
    model = load_model(str(BEST_MODEL_PATH))
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

@app.get("/")
async def root():
    return {"message": "API running", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    try:
        input_df = pd.DataFrame([application.dict()])
        prediction = model.predict(input_df)[0]
        approval_probability = float(prediction)
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
