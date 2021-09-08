import connexion
from connexion.resolver import RestyResolver
from flask import render_template


def create_app():
    app = connexion.FlaskApp(__name__, specification_dir="api")
    app.add_api("api.yaml", base_path="/api", resolver=RestyResolver("lsml_project.api"))

    app = app.app

    @app.route("/")
    def home():
        return render_template("home.jinja2")

    return app
