import os
import pytest
from click.testing import CliRunner
from unittest import mock
import numpy as np
from src.train.train import run_train


@pytest.fixture
def fake_data():
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    return X, y


@mock.patch("src.train.train.mlflow")
@mock.patch("src.train.train.utils")
def test_run_train(mock_utils, mock_mlflow, fake_data, tmp_path):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    mock_utils.load_pickle.side_effect = [fake_data, fake_data]

    mock_utils.save_pickle.return_value = None

    runner = CliRunner()
    result = runner.invoke(run_train, [
        "--data_path", str(data_path),
        "--model_path", str(model_path)
    ])

    assert result.exit_code == 0
    assert mock_utils.load_pickle.call_count == 2
    assert mock_utils.save_pickle.called
    assert mock_mlflow.log_metric.called
    assert mock_mlflow.sklearn.log_model.called