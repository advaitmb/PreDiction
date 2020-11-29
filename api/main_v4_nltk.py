from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from .inference import get_next_word, beam_search, beam_search_modified
from pathlib import Path
import pandas as pd
from random import choice
import re
import os
import sys
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
learn = load_learner('api/movie_reviews_pos_5epochs.pkl')


def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/')
def home():
    return render_template('compare_with_submit.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    text = text.replace("-", " - ")
    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    # text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    # text = re.sub('\s{2,}', ' ', text)
    text_arr = word_tokenize(text)
    # text_arr_considered = text_arr[-20:]
    # text = " ".join(text_arr_considered)
    prediction = beam_search_modified(learn, text, confidence=0.01, temperature=0.7)
    
    
    # prediction = prediction[base_string_length:]
   
    prediction_arr = word_tokenize(prediction)
    print(prediction_arr, sys.stderr)
    print(text_arr, sys.stderr)
    prediction = " ".join(prediction_arr[len(text_arr):])
    prediction = re.sub('\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
    prediction = prediction.replace(" - ", "-")
    prediction = prediction.replace(" / ", "/")
    prediction = prediction.replace(" ( ", " (")
    prediction = prediction.replace(" ) ", ") ")
    
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

@app.route('/s', methods=['GET', 'POST'])
def thanks():
    submitted_text = request.form['text']
    return " ", 204


if __name__ == "__main__":
    app.run(debug=True)
