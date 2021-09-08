FROM python:3.9

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y
RUN apt-get install -y default-mysql-client

RUN mkdir /opt/train_data

RUN cd /opt/train_data && wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip && \
    unzip BoolQ.zip && rm BoolQ.zip

RUN cd /opt/train_data && wget -c https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz && \
    gzip -d GoogleNews-vectors-negative300.bin.gz

RUN pip install -U nltk && python -m nltk.downloader -d /usr/local/nltk_data popular 

COPY requirements.txt /tmp/
RUN pip install -Ur /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . /opt/app
WORKDIR /opt/app

ENV ENV_TYPE prod
EXPOSE 80

ENTRYPOINT ["./docker-entrypoint.sh"]