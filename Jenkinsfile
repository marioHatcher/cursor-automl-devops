pipeline {
    agent any

    environment {
        PYTHON = 'python'                // Tu ejecutable ya es Python 3.11.7 ✅
        VENV_NAME = 'venv'
        MLFLOW_URI = 'http://localhost:5000'
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/marioHatcher/cursor-automl-devops.git', branch: 'master'
            }
        }

        stage('Setup Python Environment') {
            steps {
                bat """
                    %PYTHON% -m venv %VENV_NAME%
                    call %VENV_NAME%\\Scripts\\activate.bat
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }

        stage('Start MLflow Tracking Server') {
            steps {
                bat """
                    mkdir mlruns
                    start /B cmd /c "call %VENV_NAME%\\Scripts\\activate.bat && mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"
                    ping 127.0.0.1 -n 11 > nul
                """
            }
        }

        stage('Train Model') {
            steps {
                bat """
                    call %VENV_NAME%\\Scripts\\activate.bat
                    python src/models/train.py
                """
            }
        }

        stage('Start FastAPI Server') {
            steps {
                bat """
                    start /B cmd /c "call %VENV_NAME%\\Scripts\\activate.bat && uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
                    ping 127.0.0.1 -n 6 > nul
                """
            }
        }

        stage('Health Check API') {
            steps {
                bat """
                    curl http://localhost:8000/health
                """
            }
        }
    }

    post {
        always {
            echo "🧹 Limpiando workspace..."
            cleanWs()
        }
        success {
            echo '✅ Pipeline completado exitosamente 🎉'
        }
        failure {
            echo '❌ Pipeline falló 😢'
        }
    }
}
