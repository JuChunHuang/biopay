import os
import pytest
import pandas as pd
import numpy as np
from unittest import mock
from click.testing import CliRunner

from src.predict.predict import make_prediction


@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "feature1": [0.1, 0.2],
        "feature2": [1.0, 1.1]
    })


@pytest.fixture
def dummy_model():
    model = mock.Mock()
    model.predict.return_value = np.array([100, 200])
    return model


@mock.patch("src.predict.predict.utils")
@mock.patch("src.predict.predict.joblib")
@mock.patch("src.predict.predict.pd.read_csv")
@mock.patch("src.predict.predict.pd.DataFrame.to_csv")
def test_make_prediction(mock_to_csv, mock_read_csv, mock_joblib, mock_utils, dummy_df, dummy_model, tmp_path):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model"
    output_path = tmp_path / "output"
    data_path.mkdir()
    model_path.mkdir()
    output_path.mkdir()

    mock_read_csv.return_value = dummy_df
    mock_utils.standarize.return_value = dummy_df
    mock_utils.vectorize.return_value = (np.array([[0, 1], [1, 0]]), None)
    mock_utils.load_pickle.side_effect = [dummy_model, mock.Mock()]
    mock_joblib.load.return_value = mock.Mock()

    runner = CliRunner()
    result = runner.invoke(make_prediction, [
        "--data_path", str(data_path),
        "--model_path", str(model_path),
        "--output_path", str(output_path)
    ])

    assert result.exit_code == 0
    mock_utils.load_pickle.assert_any_call(os.path.join(model_path, "model.pkl"))
    mock_to_csv.assert_called_once()