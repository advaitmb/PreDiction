from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word, beam_search, beam_search_modified
from pathlib import Path
import pandas as pd
from random import choice
import re
import os
import sys

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
learn_pos = load_learner('../models/movie/positive.pkl')
learn_neg = load_learner('../models/movie/negative.pkl')

def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/a')
def home():
    return render_template('negative.html')

@app.route('/b')
def home():
    return render_template('positive.html')


@app.route('/predict_pos', methods=['GET', 'POST'])
def predict():
    text = request.form['text']

    text_arr = text.split(" ")


    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn_pos, text, confidence=0.008, temperature=1.)
    
    prediction = re.sub('\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
    prediction = prediction.replace(" - ", "-")
    prediction = prediction.replace(" / ", "/")
    prediction = prediction.replace(" ( ", " (")
    prediction = prediction.replace(" ) ", ") ")
    # prediction = prediction[base_string_length:]
    
    prediction_arr = prediction.split(" ")
    prediction = " ".join(prediction_arr[len(text_arr):])
    
    predicted = {
        "predicted": prediction
    }
    return jsonify(predicted=predicted)

@app.route('/predict_neg', methods=['GET', 'POST'])
def predict():
    text = request.form['text']

    text_arr = text.split(" ")


    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn_neg, text, confidence=0.008, temperature=1.)
    
    prediction = re.sub('\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
    prediction = prediction.replace(" - ", "-")
    prediction = prediction.replace(" / ", "/")
    prediction = prediction.replace(" ( ", " (")
    prediction = prediction.replace(" ) ", ") ")
    # prediction = prediction[base_string_length:]
    
    prediction_arr = prediction.split(" ")
    prediction = " ".join(prediction_arr[len(text_arr):])
    
    predicted = {
        "predicted": prediction
    }
    return jsonify(predicted=predicted)


@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    text = request.form['text']
    matches = [s for s in learn.dls.vocab if s and s.startswith(text)]
    if len(matches) == 0:
        prediction = ""
    else:
        prediction = choice(matches)
        prediction = prediction[len(text):]
    print(prediction, file=sys.stderr)
    
    predicted = {
        "predicted": prediction
    }
    # predicted = {str(key): value for key, value in result.items()}
    return jsonify(predicted=predicted)


if __name__ == "__main__":
    app.run(debug=True)
