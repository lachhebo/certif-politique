import math
import pickle
from typing import Dict

import pandas as pd

from app import ROOT_PATH


def get_airport_dict(df_airport: pd.DataFrame) -> dict:
    if df_airport is None:
        return {}
    df_airport.drop_duplicates(inplace=True)

    duplicate_airport = df_airport.loc[
        df_airport.duplicated(subset=['CODE IATA'], keep=False),
        ['CODE IATA', 'PRIX RETARD PREMIERE 20 MINUTES', 'PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES']].groupby(
        by=['CODE IATA']).mean().to_dict(orient='index')

    df_airport.drop_duplicates(inplace=True, subset=['CODE IATA'])

    for code_iata in duplicate_airport.keys():
        df_airport.loc[
            df_airport['CODE IATA'] == code_iata,
            ['PRIX RETARD PREMIERE 20 MINUTES']
        ] = df_airport.loc[
            df_airport['CODE IATA'] == code_iata,
            ['PRIX RETARD PREMIERE 20 MINUTES']
        ].apply(lambda x: duplicate_airport[code_iata]['PRIX RETARD PREMIERE 20 MINUTES'],
                axis=1)

        df_airport.loc[
            df_airport['CODE IATA'] == code_iata,
            ['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES']
        ] = df_airport.loc[
            df_airport['CODE IATA'] == code_iata,
            ['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES']
        ].apply(lambda x: duplicate_airport[code_iata]['PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES'],
                axis=1)

    df_airport['LONGITUDE'] = df_airport['LONGITUDE'].astype('float')
    df_airport['LATITUDE'] = df_airport['LATITUDE'].astype('float')
    df_airport['LONGITUDE TRONQUEE'] = df_airport['LONGITUDE'].apply(round)
    df_airport['LATITUDE TRONQUEE'] = df_airport['LATITUDE'].apply(round)
    return df_airport.set_index("CODE IATA").to_dict(orient='index')


def get_month(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: x.month)


def get_week(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: x.week)


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "NIVEAU DE SECURITE" in df.columns:
        df = df.drop(columns=["NIVEAU DE SECURITE"])
    return df


def get_hour(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: x // 100)


def apply_sqrt(df: pd.Series) -> pd.Series:
    return df.apply(lambda x: math.sqrt(x))


def get_dict_of_average_plane_by_day(df: pd.DataFrame, airport_type: str) -> Dict:
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


def parse_date(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X.loc[:, "DATE"] = pd.to_datetime(X["DATE"])
    X.loc[:, "MOIS"] = get_month(X["DATE"])
    X.loc[:, "SEMAINE"] = get_week(X["DATE"])
    X.loc[:, "HEURE DEPART PROGRAMME"] = get_hour(X["DEPART PROGRAMME"])
    X.loc[:, "HEURE ARRIVEE PROGRAMMEE"] = get_hour(X["ARRIVEE PROGRAMMEE"])
    return X


def load_feature_engineering(path=None):
    """
    Load file in an instance
    """
    if path is None:
        path = (ROOT_PATH / "data" / "output" / "feature_engineering.pkl").resolve()
    with open(path, "rb") as file:
        pickle_fe = pickle.load(file)

    return pickle_fe
