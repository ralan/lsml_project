import mysql.connector
import os

db_host = os.getenv("DB_HOST", "lsml_project_db")
db_user = os.getenv("DB_USER", "root")
db_password = os.getenv("DB_PASSWORD", "")

mydb = mysql.connector.connect(host=db_host, user=db_user, password=db_password, database="mysql")

mydb.cursor().execute("CREATE DATABASE IF NOT EXISTS mlflow")
