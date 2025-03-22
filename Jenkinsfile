pipeline {
    agent any

    environment {
        PYTHON = 'python'
        VENV_NAME = 'venv'
        MLFLOW_URI = 'http://localhost:5000'
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/marioHatcher/cursor-automl-devops.git', branch: 'master'
            }
        }

        stage('Setup Environment') {
            steps {
                bat '''
                    %PYTHON% -m venv %VENV_NAME%
                    call %VENV_NAME%\Scripts\activate.bat
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                bat '''
                    call %VENV_NAME%\Scripts\activate.bat
                    python src/models/train.py
                '''
            }
        }

        stage('Start API') {
            steps {
                bat '''
                    start /B cmd /c "call %VENV_NAME%\Scripts\activate.bat && uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
                    ping 127.0.0.1 -n 6 > nul
                '''
            }
        }

        stage('Health Check') {
            steps {
                bat '''
                    for /L %%i in (1,1,10) do (
                        curl http://localhost:8000/health && exit /b 0
                        ping 127.0.0.1 -n 3 > nul
                    )
                    exit /b 1
                '''
            }
        }
    }

    post {
        always {
            echo '🛑 Cleaning up...'
            bat 'taskkill /IM uvicorn.exe /F || echo uvicorn not running'
            cleanWs()
        }
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed.'
        }
    }
}