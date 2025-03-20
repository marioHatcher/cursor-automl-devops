# Cursor AutoML DevOps - Fair Loan Approval System

This project implements an automated machine learning pipeline for fair loan approval decisions, with a special focus on ensuring fairness for applicants with variable income sources.

## Project Overview

The system uses PyCaret for AutoML and MLflow for experiment tracking to build a loan approval model that maintains both high performance and fairness across different income variability groups.

### Key Features

- Automated ML pipeline with PyCaret
- Fairness metrics and bias detection
- MLflow experiment tracking
- FastAPI-based deployment
- Continuous Integration/Deployment with Jenkins
- Comprehensive testing suite

## Project Structure

```
cursor-automl-devops/
├── src/
│   ├── api/            # FastAPI service
│   ├── config/         # Configuration files
│   ├── data/           # Data processing
│   ├── fairness/       # Fairness metrics
│   ├── models/         # Model training
│   ├── preprocessing/  # Feature engineering
│   └── utils/         # Helper functions
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── notebooks/         # Jupyter notebooks
├── docs/             # Documentation
├── Jenkinsfile       # CI/CD pipeline
└── requirements.txt  # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cursor-automl-devops.git
cd cursor-automl-devops
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python src/models/train.py
```

### Running the API

```bash
uvicorn src.api.main:app --reload
```

### Running Tests

```bash
pytest tests/
```

## Model Fairness

The project implements several fairness metrics and debiasing techniques to ensure fair treatment of applicants with variable income:

- Group fairness metrics
- Equal opportunity difference
- Disparate impact analysis
- Bias mitigation techniques

## API Endpoints

- `POST /predict`: Get loan approval prediction
- `GET /metrics`: Get model performance metrics
- `GET /fairness`: Get fairness metrics

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
