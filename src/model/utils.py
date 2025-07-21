import pickle


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)