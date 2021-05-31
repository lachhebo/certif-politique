from flask import Flask, render_template, request
import pandas as pd

from certifia.feature_engineering import FeatureEngineering
from certifia.training import Training

app = Flask(__name__)


@app.route("/prediction")
def predict():
    # TODO: run application
    return "<p>Hello, World!</p>"


@app.route("/")
def root():
    return render_template('form.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return "The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        df = load_csv(request.files.get('file'))
        df_prediction = prediction(df, request.form)

        return render_template('data.html', tables=[df_prediction.to_html(classes='data', header="true")])


def load_csv(file):
    return pd.read_csv(file)


def prediction(df, form):
    df_prediction = df[['IDENTIFIANT']].copy()

    # temp
    if bool(form.get('retard_arrivee')):
        # Load feature Engineering
        feature_engineering = FeatureEngineering().load_feature_engineering(path="data/output/feature_engineering.pkl")
        df_engineered = feature_engineering.transform(df)
        df_engineered = df_engineered.drop(columns=['DATE', 'IDENTIFIANT'])
        # load training
        training = Training().load_model(path="models/rf_model.pkl")
        y_pred = training.predict(df_engineered)
        # df_prediction.loc[:, "PREDICTION RETARD A L'ARRIVEE"] = df['IDENTIFIANT'].apply(lambda x: randrange(10))
        df_prediction.loc[:, "PREDICTION RETARD A L'ARRIVEE"] = pd.Series(data=y_pred, name="PREDICTION")
    return df_prediction
