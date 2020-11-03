from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word, beam_search, beam_search_modified
from pathlib import Path
import pandas as pd
from random import choice
import csv
import re
import os
import sys
import datetime

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
learn = load_learner('../models/design/4epochslearner.pkl')
learn_neg = load_learner('../models/movies/negative.pkl')
learn_pos = load_learner('../models/movies/positive.pkl')


def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/')
def home():
    return render_template('compare.html')

@app.route('/a')
def a():
    return render_template('neg.html')

@app.route('/b')
def b():
    return render_template('pos.html')

@app.route('/c')
def c():
    return render_template('none.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']

    text_arr = text.split(" ")


    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn, text, confidence=0.008, temperature=1.)
    
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

@app.route('/a/predict', methods=['GET', 'POST'])
def a_predict():
    text = request.form['text']

    text_arr = text.split(" ")


    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn, text, confidence=0.008, temperature=1.)
    
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

@app.route('/b/predict', methods=['GET', 'POST'])
def b_predict():
    text = request.form['text']

    text_arr = text.split(" ")


    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    prediction = beam_search_modified(learn, text, confidence=0.008, temperature=1.)
    
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


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    submitted_text = request.form['text']
    bias = request.form['bias']
    print(submitted_text)

    with open(r'reviews.csv', 'a', newline='') as csvfile:
        fieldnames = ['Session','Bias','Review']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Session': datetime.datetime.now(),'Bias': bias, 'Review': '"' + submitted_text + '"' })
    return '',204


if __name__ == "__main__":
    app.run(debug=True)
