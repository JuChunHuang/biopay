import pickle


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def save_pickle(model, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)