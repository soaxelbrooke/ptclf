pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'pipenv install'
            }
        }
        stage('Test') {
            steps {
                sh 'pipenv run pytest'
            }
        }
    }
}