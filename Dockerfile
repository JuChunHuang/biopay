FROM python:3.12.5-slim

RUN pip install -U pip && pip install pipenv 

COPY src/ ./src
COPY data/ ./data
COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT [ "python", "./src/predict/predict.py" ]