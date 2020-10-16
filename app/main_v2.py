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
learn = load_learner('../models/4epochslearner_without_punct.pkl')

def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/')
def home():
    return render_template('compare.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    prediction = beam_search_modified(learn, text, confidence=0.1, temperature=1.)
    
    prediction = prediction[len(text):]
    print(prediction, file=sys.stderr)
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
