# Airflow 3.0.1 Setup with Python 3.12 on macOS

This guide explains how to install and run [Apache Airflow](https://airflow.apache.org/) 3.0.1 using **Python 3.12** and `pipenv`.

## Prerequisites

- Python 3.12 (use `pyenv` or system Python)
- pipenv (`brew install pipenv`)
- macOS

## Install Airflow 3.0.1 with Constraints

```bash
pip install "apache-airflow==3.0.1" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.1/constraints-3.12.txt"
```

## Initialize Airflow

1. Set AIRFLOW_HOME

    ```bash
    export AIRFLOW_HOME=~/airflow
    ```

    You can also add this to ~/.zshrc:

    ```bash
    echo 'export AIRFLOW_HOME=~/airflow' >> ~/.zshrc
    source ~/.zshrc
    ```

2. Initialize the Metadata Database

    ```bash
    airflow db migrate
    ```

3. Run Airflow

    ```bash
    airflow standalone
    ```

    This will:
    - Start the webserver at http://localhost:8080
    - Start the scheduler
    - Create a default user:
    - Username: admin
    - Password: check terminal output

    Note: the Username and Password will be saved in ~/airflow/simple_auth_manager_passwords.json.generated

## Add DAGs

Save your DAG scripts in the Airflow DAGs folder (default: ~/airflow/dags/). Then visit http://localhost:8080 to trigger and monitor DAGs.

Note: if you cannot find your DAGs in the web UI nor through `airflow dags list`, restart the airflow.

## Stop Airflow

```bash
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
```

## Optional Cleanup

```bash
rm -rf ~/airflow
```

## Template

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'ivy',
    'start_date': datetime(2025, 7, 21),
}

with DAG(
    dag_id='train_biopay_model',
    default_args=default_args,
    schedule=None,
    catchup=False,
    description="Run my biopay ML training script",
    tags=["ml", "biopay"],
) as dag:

    train_model = BashOperator(
        task_id='run_training_script',
        bash_command='cd your_biopay_directory && pipenv run python your_training_script.py'
    )
```