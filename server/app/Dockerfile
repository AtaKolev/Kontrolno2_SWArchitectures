FROM python:3.7

WORKDIR    /app

ENV TZ=Europe/Sofia
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY app/* ./
RUN apt-get update \
&& pip3 install --upgrade pip \
&& apt-get update \
&& apt-get clean

RUN pip3 install -r requirements.txt

EXPOSE 5000