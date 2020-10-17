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
learn = load_learner('./app/4epochslearner.pkl')

def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/')
def home():
    return render_template('compare_with_autocomplete.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    print('input_text: '+text, file=sys.stderr)
    text_arr = text.split(" ")
    for i in range(len(text_arr)):
        if text_arr[i] not in learn.dls.vocab:
            text_arr[i] = UNK
    text = ' '.join(text_arr)

    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn, text, confidence=0.001, temperature=1.)
    
    prediction = re.sub('\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
    prediction = prediction.replace(" - ", "-")
    prediction = prediction.replace(" / ", "/")
    prediction = prediction.replace(" ( ", " (")
    prediction = prediction.replace(" ) ", ") ")
    prediction = prediction[base_string_length:]
    prediction = prediction.rstrip()
    print('prediction: '+prediction, file=sys.stderr)
    predicted = {
        "predicted": prediction
    }
    return jsonify(predicted=predicted)

@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    text = request.form['text']
    text=text.lower()
    matches = [s for s in learn.dls.vocab if s and s.startswith(text)]
    if len(matches) == 0:
        prediction = ""
    else:
        prediction = choice(matches)
        prediction = prediction[len(text):]
        
    if prediction == UNK:
        prediction = ""
    print('autocomplete: ' + prediction, file=sys.stderr)
    predicted = {
        "predicted": prediction
    }
    # predicted = {str(key): value for key, value in result.items()}
    return jsonify(predicted=predicted)


if __name__ == "__main__":
    app.run(debug=True)
