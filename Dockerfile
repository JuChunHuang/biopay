FROM python:3.12.5-slim

RUN pip install -U pip pipenv

WORKDIR /app

COPY src/ ./src
COPY data/ ./data
COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

ENV PYTHONPATH="/app"

ENTRYPOINT [ "python", "./src/predict/predict.py" ]