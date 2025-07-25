import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def standarize(df: pd.DataFrame, scaler: StandardScaler, train: bool=True) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if train:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def vectorize(df: pd.DataFrame, dv: DictVectorizer):
    X = df.drop(columns="total_compensation").to_dict(orient='records')
    X = dv.transform(X)
    y = df["total_compensation"].values

    return X, y