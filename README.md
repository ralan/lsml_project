# LSML2 Final Project API

Student final project of the LSML2 course

This project is devoted to the question-answering task. The models are trained using the BoolQ dataset from SuperGLUE. The goal is to predict yes/no answer depending on passage and question. 

Yes/no question answering looks simple from the first glance, but actually it is not. By trying different classifiers and their hyper params, I found out that Random Forest classifier for word2vec embedings gives good accuracy (0.67).
Definitely, BERT-like models can give a better performance, but they require GPU on hosting box, that is not always possible. As usually, it is a question of tradeoffs.

This final project satisfies all the requiremnt:

- MLFlow used for train, selecting the best model, deploying to production, and predictions
- Dockerfile & docker-compose are used
- This is asynchronous project: Celery used (RabbitMQ as a broker, and Redis as a backend db)
- Both REST API (using Open API) and HTML Frontend (Bootstrap 5, JQuery) are implemented
- Models trained from scratch

### Prerequisites

The easiest way to run this app is by using `Docker` and `Docker Compose.`

To install Docker and Docker Compose, follow instructions at https://docs.docker.com/engine/install/
and https://docs.docker.com/compose/install/.


### Installing the LSML2 Final Project app

1. Clone the repo

2. Copy `dotenv` file to `.env` and set values

3. Build the app

```
$ docker-compose build
```


### Run the app in production mode

```
$ docker-compose up
```

- http://localhost - HTML Frontend
- http://localhost/api/ui - API
- http://localhost:5000 - MLFlow server


### Already deployed public service

- http://lsml-project.fun

