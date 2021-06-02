import os

import pandas as pd
from flask import Flask, render_template, request

from app import ROOT_PATH
from app.data_engineering.feature_engineering import FeatureEngineering
from app.model import Model
from app.data_engineering.data_cleaning import DataCleaning

templates_path: str = (ROOT_PATH / "templates").resolve()  # type: ignore
app = Flask(__name__, template_folder=templates_path)


@app.route("/")
def root():
    return render_template("form.html")


@app.route("/data/", methods=["POST", "GET"])
def data():
    if request.method == "GET":
        return "The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == "POST":
        df = load_csv(request.files.get("file"))
        df_prediction = prediction(df, request.form)

        return render_template(
            "data.html", tables=[df_prediction.to_html(classes="data", header="true")]
        )


def load_csv(file):
    return pd.read_csv(file)


def prediction(df, form):
    df_prediction = df[['IDENTIFIANT']].copy()

    if bool(form.get('retard_arrivee')):
        # Load data cleaning
        cleaning = DataCleaning(features_columns=['IDENTIFIANT',
                                                  'VOL',
                                                  'CODE AVION',
                                                  'AEROPORT DEPART',
                                                  'AEROPORT ARRIVEE',
                                                  'DEPART PROGRAMME',
                                                  'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
                                                  'TEMPS PROGRAMME',
                                                  'DISTANCE',
                                                  "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
                                                  'ARRIVEE PROGRAMMEE',
                                                  'COMPAGNIE AERIENNE',
                                                  'NOMBRE DE PASSAGERS',
                                                  'DATE',
                                                  'NIVEAU DE SECURITE'],
                                label="RETARD A L'ARRIVEE")
        cleaned_df = cleaning.cleaning(df)

        # Load feature Engineering
        feature_engineering = FeatureEngineering().load_feature_engineering(path="data/output/feature_engineering.pkl")
        df_engineered = feature_engineering.transform(cleaned_df)

        # load training
        training = Model().load_model(path="models/rf_model.pkl")
        y_pred = training.predict(df_engineered)

        df_prediction.loc[df_engineered.index, "PREDICTION RETARD A L'ARRIVEE"] = pd.Series(
            data=y_pred,
            name="PREDICTION",
            index=df_engineered.index
        )
        df_prediction["PREDICTION RETARD A L'ARRIVEE"].fillna('ANNULE / DETOURNE', inplace=True)

    return df_prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("DEBUG", "False") == "True", port=80)
