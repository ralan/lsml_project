from connexion import request
from lsml_project.tasks import deploy_best_model


def post():
    task_id = deploy_best_model.delay(experiment_name=request.json["experiment_name"])
    return {"status": "ok", "msg": f"Deploying has been started, Celery task ID: {task_id}"}, 200