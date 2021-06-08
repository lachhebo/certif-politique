import pickle
from typing import Dict

import pandas as pd

from app import ROOT_PATH


def get_month(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: x.month)


def get_week(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: x.week)


def remove_unused_columns(df):
    if "NIVEAU DE SECURITE" in df.columns:
        df = df.drop(columns=["NIVEAU DE SECURITE"])
    return df


def get_hour(df):
    return df.apply(lambda x: x // 100)


def get_start_hour(df):
    return df.apply(lambda x: x[:-2])


def __get_dict_of_average_plane_by_day(df: pd.DataFrame, airport_type: str) -> Dict:
    min_date = df["DATE"].min()
    max_date = df["DATE"].max()
    number_of_days = (max_date - min_date).days + 1
    return (
        df[[airport_type, "IDENTIFIANT", "DATE"]]
        .groupby([airport_type, "DATE"])
        .count()
        .reset_index()[[airport_type, "IDENTIFIANT"]]
        .groupby([airport_type])
        .sum()
        .apply(lambda x: x / number_of_days)["IDENTIFIANT"]
        .to_dict()
    )


def apply_time_transformation(df):
    df.loc[:, "DATE"] = pd.to_datetime(df["DATE"])
    df.loc[:, "MOIS"] = get_month(df["DATE"])
    df.loc[:, "SEMAINE"] = get_week(df["DATE"])


def load_feature_engineering(path=None):
    """
    Load file in an instance
    """
    if path is None:
        path = (ROOT_PATH / "data" / "output" / "feature_engineering.pkl").resolve()
    with open(path, "rb") as file:
        pickle_fe = pickle.load(file)

    return pickle_fe
