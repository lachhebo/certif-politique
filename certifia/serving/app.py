from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route("/prediction")
def predict():
    # TODO: run application
    return "<p>Hello, World!</p>"


@app.route("/")
def root():
    return render_template('app.html', name='ismael')


@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('answer.html',form_data = form_data)
 