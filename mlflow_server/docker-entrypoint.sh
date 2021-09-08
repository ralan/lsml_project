#!/bin/bash

while ! mysqladmin ping -h ${DB_HOST:-lsml_project_db} --silent; do
    echo "Waiting for database connection"
    sleep 1
done

python /opt/create_db.py

echo "Starting MLflow server"
exec mlflow server --backend-store-uri mysql://${DB_USER:-root}:${DB_PASSWORD}@${DB_HOST:-lsml_project_db}:3306/mlflow --default-artifact-root file:/opt/app/mlruns --host 0.0.0.0
