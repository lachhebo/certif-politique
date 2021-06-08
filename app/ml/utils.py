import pandas as pd

from app import ROOT_PATH
from app.data_engineering.data_cleaning import DataCleaning
from app.data_engineering.utils import load_feature_engineering
from app.ml.model import Model


def _apply_prediction(df_engineered: pd.DataFrame, df_prediction: pd.DataFrame):
    training = Model().load_model(path="models/rf_model.pkl")
    y_pred = training.predict(df_engineered)
    df_prediction.loc[:, "PREDICTION RETARD A L'ARRIVEE"] = pd.Series(
        data=y_pred, name="PREDICTION"
    )


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feature_engineering = load_feature_engineering(
        path="data/output/feature_engineering.pkl"
    )
    df_engineered = feature_engineering.transform(df)
    df_engineered = df_engineered.drop(columns=["DATE", "IDENTIFIANT"])
    return df_engineered


def load_csv(file):
    return pd.read_csv(file)


def prediction(df, form):
    df_prediction = df[["IDENTIFIANT"]].copy()

    if bool(form.get("retard_arrivee")):
        # Load data cleaning
        cleaning = DataCleaning(
            features_columns=[
                "IDENTIFIANT",
                "VOL",
                "CODE AVION",
                "AEROPORT DEPART",
                "AEROPORT ARRIVEE",
                "DEPART PROGRAMME",
                "TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE",
                "TEMPS PROGRAMME",
                "DISTANCE",
                "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
                "ARRIVEE PROGRAMMEE",
                "COMPAGNIE AERIENNE",
                "NOMBRE DE PASSAGERS",
                "DATE",
                "NIVEAU DE SECURITE",
            ],
            label="RETARD A L'ARRIVEE",
        )
        cleaned_df = cleaning.cleaning(df)

        # Load feature Engineering
        feature_engineering = load_feature_engineering(
            path=f"{ROOT_PATH}/data/output/feature_engineering.pkl"
        )
        df_engineered = feature_engineering.transform(cleaned_df)

        # load training
        training = Model().load_model(path=f"{ROOT_PATH}/models/rf_model.pkl")
        y_pred = training.predict(df_engineered)

        df_prediction.loc[
            df_engineered.index, "PREDICTION RETARD A L'ARRIVEE"
        ] = pd.Series(data=y_pred, name="PREDICTION", index=df_engineered.index)
        df_prediction["PREDICTION RETARD A L'ARRIVEE"].fillna(
            "ANNULE / DETOURNE", inplace=True
        )

    return df_prediction
