import os
import utils
import click
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("biopay-experiment")
# mlflow ui --backend-store-uri sqlite:///mlflow.db

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


@click.command()
@click.option(
    "--data_path",
    default="data/processed_data",
    help="Location where the processed biopay data was saved"
)

def run_train(data_path: str):
    mlflow.autolog()

    X_train, y_train = utils.load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = utils.load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.set_tag("developer", "Ivy")

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(rf, 'rf-model')


if __name__ == '__main__':
    run_train()
