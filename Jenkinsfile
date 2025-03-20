pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.8'
        VENV_NAME = 'venv'
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout code from the public Git repository
                git url: 'https://github.com/yourusername/cursor-automl-devops.git', branch: 'main'
            }
        }
        
        stage('Setup') {
            steps {
                script {
                    // Create and activate virtual environment
                    bat """
                        python -m venv ${VENV_NAME}
                        ${VENV_NAME}\\Scripts\\activate.bat
                        python -m pip install --upgrade pip
                        pip install -r requirements.txt
                    """
                }
            }
        }
        
        stage('Lint') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        pip install flake8
                        flake8 src/ tests/ --max-line-length=100
                    """
                }
            }
        }
        
        stage('Unit Tests') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        pytest tests/unit/ -v --junitxml=test-results/unit-tests.xml
                    """
                }
            }
        }
        
        stage('Integration Tests') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        pytest tests/integration/ -v --junitxml=test-results/integration-tests.xml
                    """
                }
            }
        }
        
        stage('Train Model') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        python src/models/train.py
                    """
                }
            }
        }
        
        stage('Evaluate Fairness') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        python -c "
import sys
from src.fairness.metrics import FairnessMetrics
import joblib

model = joblib.load('models/best_model.pkl')
fairness = FairnessMetrics()
is_fair, reason = fairness.is_fair()

if not is_fair:
    print(f'Fairness check failed: {reason}')
    sys.exit(1)
print('Fairness check passed')
"
                    """
                }
            }
        }
        
        stage('Build API') {
            steps {
                script {
                    bat """
                        ${VENV_NAME}\\Scripts\\activate.bat
                        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
                        timeout /t 5 /nobreak
                        curl http://localhost:8000/health
                    """
                }
            }
        }
    }
    
    post {
        always {
            // Clean up
            cleanWs()
            
            // Archive test results
            junit 'test-results/*.xml'
            
            // Archive MLflow artifacts
            archiveArtifacts artifacts: 'mlruns/**/*', fingerprint: true
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            echo 'Pipeline failed!'
        }
    }
} 