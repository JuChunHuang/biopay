import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def drop_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts()
    to_keep = counts[counts >= 5].index
    df = df[df[column].isin(to_keep)]

    return df


def split_data(df: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, val_df


def vectorize(df: pd.DataFrame, dv: DictVectorizer, train: bool=True):
    X = df.drop(columns="total_compensation").to_dict(orient='records')
    y = df["total_compensation"].values

    if train:
        X = dv.fit_transform(X)
    else:
        X = dv.transform(X)

    return X, y, dv


def vectorize_data(df: pd.DataFrame) -> tuple:
    dv = DictVectorizer()
    scaler = StandardScaler()
    train_df, val_df = split_data(df)
    train_df = standarize(train_df, scaler, train=True)
    val_df = standarize(val_df, scaler, train=False)
    X_train, y_train, dv = vectorize(train_df, dv, train=True)
    X_val, y_val, _ = vectorize(val_df, dv, train=False)

    return X_train, y_train, X_val, y_val, dv


def dump_pkl(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def standarize(df: pd.DataFrame, scaler: StandardScaler, train: bool=True) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if train:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df