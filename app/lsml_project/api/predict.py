from connexion import request
from celery.result import AsyncResult

from lsml_project.tasks import predict, celery


def post():
    params = request.json
    task_id = predict.delay(passage=params["passage"], question=params["question"])
    return {"task_id": str(task_id)}, 200


def get(task_id):
    r = AsyncResult(task_id, app=celery)

    if r.ready():
        return {"status": "DONE", "answer": r.result}, 200
    else:
        return {"status": "IN_PROGRESS"}, 200
