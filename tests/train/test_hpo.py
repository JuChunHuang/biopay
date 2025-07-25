import os
import pytest
from unittest import mock
import numpy as np
from click.testing import CliRunner

from src.train.hpo import run_optimization


@pytest.fixture
def fake_data():
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    return X, y


@mock.patch("src.train.hpo.mlflow")
@mock.patch("src.train.hpo.utils")
@mock.patch("src.train.hpo.fmin")
def test_run_optimization(mock_fmin, mock_utils, mock_mlflow, fake_data, tmp_path):
    mock_utils.load_pickle.side_effect = [fake_data, fake_data]
    mock_fmin.return_value = {"loss": 0.1}

    data_path = tmp_path / "data"
    os.makedirs(data_path, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(run_optimization, [
        "--data_path", str(data_path),
        "--num_trials", "5"
    ])

    assert result.exit_code == 0
    assert mock_utils.load_pickle.call_count == 2
    assert mock_fmin.called
    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_metric.called