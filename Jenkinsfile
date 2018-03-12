pipeline {
    agent any

    stages {
        stage('Build') {
            sh 'pipenv install'
        }
        stage('Test') {
            sh 'pipenv run pytest'
        }
    }
}