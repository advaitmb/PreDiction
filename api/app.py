from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word, beam_search, beam_search_modified, beam_search_modified_with_clf
from pathlib import Path
import pandas as pd
from random import choice
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, TweetTokenizer
import csv
import re
import os
import sys
import datetime

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
# learn = load_learner('../models/design/4epochslearner.pkl')
learn_lm = load_learner('5epochs_imdb_lm.pkl')
clf = load_learner('imdb_sentiment_classifier.pkl')

def subtract(a, b):                              
    return "".join(a.rsplit(b))

@app.route('/')
def home():
    return render_template('movie_reviews.html')

@app.route('/a')
def a():
    return render_template('pos.html')  

@app.route('/b')
def b():
    return render_template('neg.html')

@app.route('/c')
def c():
    return render_template('none.html')


@app.route('/a/predict', methods=['GET', 'POST'])
def a_predict():
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
    prediction = beam_search_modified_with_clf(learn=learn_lm, clf=clf, bias='pos', text=text, confidence=0.01, temperature=0.7)
    
    
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

@app.route('/b/predict', methods=['GET', 'POST'])
def b_predict():
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
    prediction = beam_search_modified_with_clf(learn=learn_lm, clf=clf, bias='neg', text=text, confidence=0.01, temperature=0.7)
    
    
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


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    submitted_text = request.form['text']
    bias = request.form['bias']
    print(submitted_text)

    with open(r'api/reviews.csv', 'a', newline='') as csvfile:
        fieldnames = ['Session','Bias','Review']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Session': datetime.datetime.now(),'Bias': bias, 'Review': '"' + submitted_text + '"' })
    return '',204


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
