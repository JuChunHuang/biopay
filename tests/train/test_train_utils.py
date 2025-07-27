import src.train.utils as utils
import tempfile
import pickle
import os


def test_load_pickle():
    test_data = {"key": "value"}
    filename = "test.pkl"
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        utils.save_pickle(test_data, filepath)
        loaded_data = utils.load_pickle(filepath)
        assert loaded_data == test_data


def test_save_pickle():
    test_data = {"key": "value"}
    filename = "test.pkl"
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        utils.save_pickle(test_data, filepath)
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == test_data