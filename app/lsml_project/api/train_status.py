from celery.result import AsyncResult
from lsml_project.tasks import celery


def get(task_id):
    task = AsyncResult(task_id, app=celery)

    if task.ready():
        return {"status": "DONE"}, 200
    else:
        return {"status": "IN_PROGRESS"}, 200
