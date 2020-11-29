from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word
from pathlib import Path
import pandas as pd
from random import choice
import re
import os
import sys

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
learn = load_learner('../models/design/4epochslearner.pkl')

# Defining the home page for the web service
@app.route('/')
def home():
    return render_template('mobile.html')

# Writing api for inference using the loaded model
@app.route('/predict',methods=['POST', 'GET'])
# Defining the predict method get input from the html page and to predict using the trained model
def predict():
    text = request.form['text']
    # n_words = int(request.form['nwords'])
    prediction = get_next_word(learn, text)
    if len(prediction) == 0:
        predicted = {
            "word1": "",
            "word2": "",
            "word3": "",
        }
    elif len(prediction) == 1:
        predicted = {
            "word1": prediction[0],
            "word2": "",
            "word3": "",
        }
    elif len(prediction) == 2:
        predicted = {
            "word1": prediction[0],
            "word2": prediction[1],
            "word3": "",
        }
    else:
        predicted = {
            "word1": prediction[0],
            "word2": prediction[1],
            "word3": prediction[2],
        }

    return jsonify(predicted=predicted)

@app.route('/autocomplete', methods=['POST', 'GET'])
def autocomplete():
    text = request.form['text']
    matches = [s for s in learn.dls.vocab if s and s.startswith(text)]
    if len(matches) == 0:
        autocompleted = {
            "word1": "",
            "word2": "",
            "word3": "",
        }
    elif len(matches) == 1:
        autocompleted = {
            "word1": matches[0],
            "word2": "",
            "word3": "",
        }
    elif len(matches) == 2:
        autocompleted = {
            "word1": matches[0],
            "word2": matches[1],
            "word3": "",
        }
    else:
        autocompleted = {
            "word1": matches[0],
            "word2": matches[1],
            "word3": matches[2],
        }
    return jsonify(autocompleted=autocompleted)

if __name__ == "__main__":
    app.run(debug=True)