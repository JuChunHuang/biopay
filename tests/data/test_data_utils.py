import pytest
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import src.data.preprocess as preprocess
import src.data.utils as utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import tempfile
import pickle

@pytest.fixture
def data():
    return preprocess.read_data(os.path.join(os.getcwd(), "data", "raw_data",  "r_biotech salary and company survey.xlsx"))


def test_drop_counts(data):
    df = utils.drop_counts(data, "country")
    assert isinstance(df, pd.DataFrame)


def test_split_data(data):
    train_df, val_df, test_df = utils.split_data(data)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) + len(val_df) + len(test_df) == len(data)


def test_vectorize():
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
        "total_compensation": [1000, 2000, 3000]
    })
    dv = DictVectorizer()
    X, y, dv = utils.vectorize(test_data, dv)
    print(type(X))
    assert isinstance(X, csr_matrix)
    assert isinstance(y, np.ndarray)
    assert isinstance(dv, DictVectorizer)


def test_vectorize_data():
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
        "total_compensation": [1000, 2000, 3000]
    })
    X_train, y_train, X_val, y_val, test_df, dv, scaler = utils.vectorize_data(test_data)
    assert isinstance(X_train, csr_matrix)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_val, csr_matrix)
    assert isinstance(y_val, np.ndarray)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(dv, DictVectorizer)


def test_dump_pkl():
    test_data = {"key": "value"}
    filename = "test.pkl"
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        utils.dump_pkl(test_data, filepath)
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == test_data


def test_standarize():
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4.0, 5.0, 6.0]
    })
    scaler = StandardScaler()
    standardized_df = utils.standarize(test_data, scaler, train=True)
    assert isinstance(standardized_df, pd.DataFrame)