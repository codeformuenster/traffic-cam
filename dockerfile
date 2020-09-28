FROM python:3.7

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="$PYTHONPATH:/"

RUN apt-get update && apt-get -y install ffmpeg

RUN mkdir app
WORKDIR app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY bin bin
COPY cfg cfg
COPY output output
COPY traffic_cam traffic_cam
COPY setup.py setup.py
RUN touch is_docker_container
RUN pip install -e .
