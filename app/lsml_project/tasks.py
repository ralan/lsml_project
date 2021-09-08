import gensim
import json
import mlflow
import mlflow.sklearn
import nltk
import numpy as np
import os
import pandas as pd
import requests
import sentry_sdk

from celery import Celery
from functools import cache
from nltk.corpus import stopwords
from sentry_sdk.integrations.celery import CeleryIntegration
from sklearn.ensemble import RandomForestClassifier


if os.getenv("ENV_TYPE") == "prod":
    sentry_dsn = os.getenv("SENTRY_DSN")

    if sentry_dsn:
        sentry_sdk.init(sentry_dsn, integrations=[CeleryIntegration()])

if os.getenv("BROKER_URL"):
    broker_url = os.getenv("BROKER_URL")
else:
    broker_host = os.getenv("BROKER_HOST", "lsml_project_broker")
    broker_url = f"pyamqp://guest@{broker_host}//"

redis_host = os.getenv("REDIS_HOST", "lsml_project_redis")

celery = Celery("lsml_project", backend=f"redis://{redis_host}", broker=broker_url)

if os.getenv("MLFLOW_SERVER_URL"):
    MLFLOW_SERVER_URL = os.getenv("MLFLOW_SERVER_URL")
else:
    MLFLOW_SERVER_URL = "http://lsml_project_mlflow_server:5000/"

if os.getenv("MLFLOW_INVOCATOR_URL"):
    MLFLOW_INVOCATOR_URL = os.getenv("MLFLOW_INVOCATOR_URL")
else:
    MLFLOW_INVOCATOR_URL = "http://lsml_project_mlflow_invocator:5005/invocations"

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = nltk.stem.WordNetLemmatizer()

REG_MODEL_NAME = "boolq-random-forrest"


@celery.task()
def predict(passage, question):
    df = pd.DataFrame.from_dict({"passage": [passage], "question": [question]})

    df["question_tok"] = df.question.apply(tokenize_text)
    df["passage_tok"] = df.passage.apply(tokenize_text)

    vectorizer = get_vecorizer()

    X = np.concatenate((vectorizer.transform(df.question_tok), vectorizer.transform(df.passage_tok)), axis=1)

    data = json.dumps(X.tolist())
    response = requests.post(url=MLFLOW_INVOCATOR_URL, headers={"Content-Type": "application/json"}, data=data)
    result = response.json()[0]

    return "yes" if result else "no"


@celery.task()
def train(params):
    mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
    mlflow.set_experiment(params["experiment_name"])

    X_train, y_train = get_train_data()
    X_val, y_val = get_val_data()

    for train_params in params["train_params"]:
        with mlflow.start_run():
            model = RandomForestClassifier(**train_params)
            model.fit(X_train, y_train)

            acc = model.score(X_val, y_val)

            for param in train_params:
                mlflow.log_param(param, train_params[param])

            mlflow.log_metric("acc", acc)

            mlflow.sklearn.log_model(model, "model")


@celery.task()
def deploy_best_model(experiment_name):
    client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URL)
    experiment = client.get_experiment_by_name(experiment_name)
    client.create_registered_model(REG_MODEL_NAME)

    current_prod_acc = 0

    model_versions = client.search_model_versions(f"name='{REG_MODEL_NAME}'")
    prods = [v for v in model_versions if v.current_stage == "Production"]

    if prods:
        current_prod = prods[-1]

        if current_prod:
            current_prod_metrics = client.get_run(current_prod.run_id).data.metrics
            current_prod_acc = current_prod_metrics["acc"]

    run_infos = client.list_run_infos(experiment.experiment_id, order_by=["metric.acc DESC"])

    if not run_infos:
        return

    best_run_info = run_infos[0]
    best_run = mlflow.get_run(best_run_info.run_id)

    if current_prod_acc < best_run.data.metrics["acc"]:
        new_model = client.create_model_version(
            name=REG_MODEL_NAME, source=f"{best_run_info.artifact_uri}/model", run_id=best_run_info.run_id
        )

        client.transition_model_version_stage(name=REG_MODEL_NAME, version=new_model.version, stage="Production")


def tokenize_text(text):
    tokens = nltk.word_tokenize(text.lower(), language="english")
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token.isalpha() and token not in STOPWORDS]
    return tokens


class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = word2vec[word2vec.index_to_key[0]].shape[0]

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(
            [
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
                for words in X
            ]
        )


@cache
def get_vecorizer():
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        "/opt/train_data/GoogleNews-vectors-negative300.bin", binary=True
    )

    return EmbeddingVectorizer(word2vec)


@cache
def get_train_data():
    df_train = pd.read_json("/opt/train_data/BoolQ/train.jsonl", lines=True)

    df_train["question_tok"] = df_train.question.apply(tokenize_text)
    df_train["passage_tok"] = df_train.passage.apply(tokenize_text)

    vectorizer = get_vecorizer()

    X_train = np.concatenate(
        (vectorizer.transform(df_train.question_tok), vectorizer.transform(df_train.passage_tok)), axis=1
    )

    y_train = df_train.label.map({True: 1, False: 0}).values

    return X_train, y_train


@cache
def get_val_data():
    df_val = pd.read_json("/opt/train_data/BoolQ/val.jsonl", lines=True)

    df_val["question_tok"] = df_val.question.apply(tokenize_text)
    df_val["passage_tok"] = df_val.passage.apply(tokenize_text)

    vectorizer = get_vecorizer()

    X_val = np.concatenate(
        (vectorizer.transform(df_val.question_tok), vectorizer.transform(df_val.passage_tok)), axis=1
    )

    y_val = df_val.label.map({True: 1, False: 0}).values

    return X_val, y_val
