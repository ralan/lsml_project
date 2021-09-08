#!/bin/bash 

while ! curl -s -u guest:guest ${BROKER_HOST:-lsml_project_broker_dev}:15672/api/healthchecks/node | grep "ok"; do
    echo "Waiting for message broker connection"
    sleep 1
done

case $1 in
workers)
    if [ "$ENV_TYPE" = "dev" ];
    then
        echo "Starting Celery workers with autoreload"
        exec watchmedo auto-restart --recursive -d lsml_project -p '*.py' -- celery -A lsml_project.tasks worker --loglevel=DEBUG --uid nobody --concurrency=4 ${@:2}
    else
        echo "Starting Celery workers"
        exec celery -A lsml_project.tasks worker --loglevel=INFO --uid nobody --concurrency=4 ${@:2}
    fi
    ;;
mlflow_invocator)
    until $(curl --output /dev/null --silent --head --fail ${MLFLOW_SERVER_HOST:-lsml_project_mlflow_server}:${MLFLOW_SERVER_PORT:-5000}); do
        echo "Waiting for MLflow server"
        sleep 1
    done

    echo "Starting MLflow invocator"
    export MLFLOW_TRACKING_URI=http://${MLFLOW_SERVER_HOST:-lsml_project_mlflow_server}:${MLFLOW_SERVER_PORT:-5000}
    exec mlflow models serve -m "models:/boolq-random-forrest/Production" -p 5005 --no-conda --host 0.0.0.0
    ;;
tests)
    echo "Running tests"
    exec pytest tests/${@:2}
    ;;
*)
    while ! celery -A lsml_project.tasks inspect ping | grep "pong"; do
        echo "Waiting for Celery workers"
        sleep 1
    done

    if [ "$ENV_TYPE" = "dev" ];
    then
        echo "Starting application using Flask server (development environment)"
        export FLASK_ENV=development
        export FLASK_RUN_EXTRA_FILES=lsml_project/api/api.yaml
        exec flask run --host=0.0.0.0 --port=80 ${@:2}
    else
        echo "Starting application using Gunicorn server (production environment)"
        exec gunicorn --bind 0.0.0.0:80 --workers=4 ${@:2} app:app
    fi
    ;;
esac
