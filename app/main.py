import os

from flask import Flask, render_template, request

from app import ROOT_PATH
from app.ml.utils import load_csv, prediction

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("DEBUG", "False") == "True", port=80)
