FROM python:3.9

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y
RUN apt-get install -y default-mysql-client

COPY requirements.txt /tmp/
RUN pip install -Ur /tmp/requirements.txt && rm /tmp/requirements.txt

COPY docker-entrypoint.sh /opt
COPY create_db.py /opt

EXPOSE 5000

ENTRYPOINT ["/opt/docker-entrypoint.sh"]