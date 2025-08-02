# 💵 Biopay: Biotech Salary Predictor

Biopay is a machine learning project that predicts the **total compensation** (base salary + bonuses + equity, etc.) of an employee in the **biotech/pharmaceutical industry** in the United States.

This project is built using Python and leverages the [2024 Biotech Salary and Company Survey](https://www.reddit.com/r/biotech/comments/18vq4rw/rbiotech_salary_and_company_survey_2024/) shared by the Reddit community on r/biotech.

## Tech Stack

- **Pipenv** for dependency management
- **MLflow** for experiment tracking and model registry
- **Docker** for containerization
- **Airflow** for orchestration
- **Localstack** for cloud service
- **Pylint** for code formatting
- **GitHub Action** for CI/CD pipeline, including unit tests and integration test
- **Pre-commit hook**

## Project Structure

```
biopay/
│
├── data/
│   ├── models/               # Model checkpoint
│   ├── processed_data/       # Processed data
│   └── raw_data/             # Raw data
├── src/
│   ├── data/
│   │   ├── preprocess.py     # Data processing script
│   │   └── utils.py          # Data utilities
│   ├── train/
│   │   ├── train.py          # Main training entrypoint
│   │   ├── hpo.py            # Hyperparameter tuning using Hyperopt
│   │   └── utils.py          # Training utilities
│   └── predict/
│       ├── predict.py        # Generate predictions on new data
│       └── utils.py          # Predicting utilities
├── tests/                    # Pytest unit tests
├── Dockerfile                # Docker configuration
├── docker-compose.yaml       # Localstack setup
├── Pipfile / Pipfile.lock    # Pipenv dependencies
└── README.md                 # You are here
```

## Installation

```bash
pip install pipenv
pipenv install --dev
pipenv shell
```

For the entire model development, please run:

```bash
pipenv run python src/data/preprocess.py
pipenv run python src/train/train.py
pipenv run python src/predict/predict.py
```

## Docker Support

You can also run the project in a container:

```bash
docker build -t biopay .
docker run --rm biopay
```

## Airflow Setup

Please go to [`src/train/AIRFLOW.md`](https://github.com/JuChunHuang/biopay/blob/main/src/train/AIRFLOW.md) for more instruction.

## Localstack Setup

Please go to [`tests/integration_test/LOCALSTACK.md`](https://github.com/JuChunHuang/biopay/blob/main/tests/integration_test/LOCALSTACK.md) for more instruction.

## Testing

Unit tests are written using `pytest`. Run them with:

```bash
pipenv run python -m pytest
```