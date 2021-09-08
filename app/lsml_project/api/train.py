from connexion import request
from lsml_project.tasks import train


def post():
    task_id = train.delay(params=request.json)
    return {"status": "ok", "msg": f"Train has been started, Celery task ID: {task_id}"}, 200
