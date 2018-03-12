pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'python -m pipenv install'
            }
        }
        stage('Test') {
            steps {
                sh 'python -m pipenv run pytest'
            }
        }
    }
}