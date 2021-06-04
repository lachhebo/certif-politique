import os

import pandas as pd
from flask import Flask, render_template, request

from app import ROOT_PATH
from app.data_engineering.feature_engineering import FeatureEngineering
from app.model import Model

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
    df_prediction = df[["IDENTIFIANT"]].copy()

    if bool(form.get("retard_arrivee")):
        # Load feature Engineering
        feature_engineering = FeatureEngineering().load_feature_engineering(
            path="data/output/feature_engineering.pkl"
        )
        df_engineered = feature_engineering.transform(df)
        df_engineered = df_engineered.drop(columns=["DATE", "IDENTIFIANT"])
        # load training
        training = Model().load_model(path="models/rf_model.pkl")
        y_pred = training.predict(df_engineered)
        df_prediction.loc[:, "PREDICTION RETARD A L'ARRIVEE"] = pd.Series(
            data=y_pred, name="PREDICTION"
        )
    return df_prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("DEBUG", "False") == "True", port=80)
