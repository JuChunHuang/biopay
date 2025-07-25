import src.data.preprocess as preprocess
import pytest
import os
import pandas as pd

@pytest.fixture
def data_path():
    return os.path.join(os.getcwd(), "data", "raw_data",  "r_biotech salary and company survey.xlsx")

@pytest.fixture
def data(data_path):
    return preprocess.read_data(data_path)


def test_read_data(data_path):
    df = preprocess.read_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert "base_salary" in df.columns


def test_filter_country(data):
    df = preprocess.filter_country(data)
    assert isinstance(df, pd.DataFrame)
    assert "country" not in df.columns


def test_process_industry(data):
    df = preprocess.process_industry(data)
    assert isinstance(df, pd.DataFrame)


def test_process_location(data):
    df = preprocess.process_location(data)
    assert isinstance(df, pd.DataFrame)
    assert "new england (ma, ct, ri, nh, vt, me)" in df["location"].values
    assert "boston" not in df["location"].values


def test_process_degree(data):
    df = preprocess.process_degree(data)
    assert isinstance(df, pd.DataFrame)
    assert "degree" in df.columns
    assert "BS" in df["degree"].values


def test_process_wlb(data):
    df = preprocess.process_wlb(data)
    assert isinstance(df, pd.DataFrame)
    assert df["wlb"].nunique() == 5


def test_process_bonus(data):
    df = preprocess.process_bonus(data, "bonus")
    assert isinstance(df, pd.DataFrame)
    assert df["bonus"].isnull().sum() == 0


def test_process_salary(data):
    df = preprocess.process_salary(data)
    assert isinstance(df, pd.DataFrame)
    assert "total_compensation" in df.columns
    assert "base_salary" not in df.columns
    assert df["total_compensation"].isnull().sum() == 0