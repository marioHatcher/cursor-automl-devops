# Fair Loan Approval API (ML + DevOps)

This project builds a machine learning pipeline for fair loan approval using PyCaret and FastAPI, integrated into a DevOps pipeline with Jenkins.

## 🎯 Objectives

1. Use **PyCaret** to automate model selection.
2. Register experiments using **MLflow**.
3. Detect the effect of **income variability** on loan approval.
4. Save the best model and deploy it via **FastAPI**.
5. Use **Jenkins** for CI/CD automation.

## 🧪 Project Structure

```
project/
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI server
│   ├── models/
│   │   └── train.py         # Model training with PyCaret
│   └── config/
│       └── config.py        # Project configuration
├── requirements.txt
├── Jenkinsfile              # CI/CD pipeline
└── README.md
```

## 🚀 Jenkins Pipeline

1. Checks out the code from GitHub
2. Sets up a Python virtual environment
3. Installs dependencies
4. Trains the model with PyCaret
5. Starts FastAPI as a background service
6. Validates the `/health` endpoint

## 🔮 Prediction API

- `POST /predict` with JSON input of loan application
- `GET /health` to check API status

## ✅ Example Input
```json
{
  "age": 35,
  "income": 55000,
  "loan_amount": 20000,
  "loan_term": 24,
  "credit_score": 700,
  "employment_status": "Employed",
  "loan_purpose": "Home",
  "existing_loans": 1,
  "income_variability": "Low"
}
```

## 📦 Output
```json
{
  "loan_approved": true,
  "approval_probability": 0.89,
  "fairness_metrics": {
    "group": "Low Income Variability",
    "group_approval_rate": 0.89
  }
}
```

---
That's it! Push to GitHub and Jenkins will handle the rest 🚀
