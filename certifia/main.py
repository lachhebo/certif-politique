from flask import Flask, render_template, request

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
        print(request)
        form_data = request.form
        dummy_prediction(request.form)

        return render_template('data.html', form=form_data)


def dummy_prediction(request):
    return {'retard': 4}


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
