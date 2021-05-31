from random import randrange, seed

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)


@app.route("/prediction")
def predict():
    # TODO: run application
    return "<p>Hello, World!</p>"


@app.route("/")
def root():
    return render_template('form.html', name='ismael')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return "The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        df = load_csv(request.files.get('file'))
        df_prediction = prediction(df)

        return render_template('data.html', tables=[df_prediction.to_html(classes='data', header="true")])


def load_csv(file):
    return pd.read_csv(file)


def prediction(df):
    # Load feature Engineering
    # feature_engineering = FeatureEngineering()
    # df_engineered = feature_engineering.transform(df)
    # load training
    # training = Training()
    # y_pred = training.predict(df_engineered)
    # concat y_pred and flight number
    # df_prediction = concat([df[['IDENTIFIANT']], pd.Series(data=y_pred, name="PREDICTION"])

    # temporairement
    seed(42)
    df_prediction = df[['IDENTIFIANT']].copy()
    df_prediction.loc[:, 'PREDICTION'] = df['IDENTIFIANT'].apply(lambda x: randrange(10))
    return df_prediction
