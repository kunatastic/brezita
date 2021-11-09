FROM python:3.8

WORKDIR /app

COPY . .

RUN ["pip3","install","pipenv"]

RUN ["pip3", "install", "--upgrade", "pip"]

RUN ["pipenv", "install"]

ENV FLASK_APP=main.py

CMD ["pipenv","run","flask","run","--host=0.0.0.0"]