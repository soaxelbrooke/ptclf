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
                sh 'RNN_TEST_DRAWS=50 PYTHONPATH=$(pwd) python -m pipenv run pytest'
            }
        }
    }
}