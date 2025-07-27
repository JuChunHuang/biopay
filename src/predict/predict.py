import click
import pandas as pd
import os
from src.predict import utils
import joblib


@click.command()
@click.option(
    "--data_path",
    default="data/processed_data",
    help="Location where the processed biopay data was saved"
)
@click.option(
    "--model_path",
    default="data/models",
    help="Location where the trained model was saved"
)
@click.option(
    "--output_path",
    default="data/outputs",
    help="Location where the predicted result should be saved"
)
def make_prediction(data_path: str, model_path: str, output_path: str):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    model = utils.load_pickle(os.path.join(model_path, "model.pkl"))
    dv = utils.load_pickle(os.path.join(data_path, "dv.pkl"))
    scaler = joblib.load(os.path.join(data_path, "scaler.save"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    transformed_df = utils.standarize(test_df, scaler, train=False)
    X, _ = utils.vectorize(transformed_df, dv)

    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"], index=test_df.index)

    if s3_endpoint_url:
        options = {
            "client_kwargs": {
                "endpoint_url": s3_endpoint_url
            },
            "key": os.getenv('AWS_ACCESS_KEY_ID'),
            "secret": os.getenv('AWS_SECRET_ACCESS_KEY')
        }
        y_pred.to_csv(os.path.join(output_path, "predictions.csv"), storage_options=options)
    else:
        y_pred.to_csv(os.path.join(output_path, "predictions.csv"))

    print(y_pred)


if __name__ == '__main__':
    make_prediction()