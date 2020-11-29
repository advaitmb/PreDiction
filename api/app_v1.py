# Importing necessary packages
from flask import Flask, request, render_template, jsonify
from fastai.text import *
import os
import sys


# Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/models'

# Initializing the FLASK API
app = Flask(__name__)

# Loading the saved model using fastai's load_learner method
model = load_learner(path, '5epochslearner_v1.pkl')

# Defining the home page for the web service


@app.route('/')
def home():
    return render_template('mobile.html')

# Writing api for inference using the loaded model


@app.route('/predict', methods=['POST', 'GET'])
# Defining the predict method get input from the html page and to predict using the trained model
def predict():

    text = request.form['text']
    # n_words = int(request.form['nwords'])
    prediction = model.beam_search(text, n_words=1)
    prediction = prediction[2*len(text)+2:]

    predicted = {
        "predicted": prediction
    }
    # predicted = {str(key): value for key, value in result.items()}
    return jsonify(predicted=predicted)


if __name__ == "__main__":
    app.run(debug=True)
