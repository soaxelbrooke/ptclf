pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'python -m pipenv install --dev'
            }
        }
        stage('Test') {
            steps {
                sh 'PYTHONPATH=$(pwd) python -m pipenv run pytest'
            }
        }
    }
}