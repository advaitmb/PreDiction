from pathlib import Path
import random
import datetime
import string
import sys
import os
import re
import csv

import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer

import pandas as pd

from flask import Flask, request, render_template, jsonify

from fastai.text.all import *
from inference import beam_search_modified, beam_search_modified_with_clf, complete_word

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object
language_model = load_learner('/Users/Saaket/Documents/Machine Learning/PreDiction/api/5epochs_imdb_lm.pkl')
classifier = load_learner('/Users/Saaket/Documents/Machine Learning/PreDiction/api/imdb_sentiment_classifier.pkl')

# Hidden bias mapping
bias_mapping = {
    'a': 'pos',
    'b': 'neg',
    'c': 'neu'
}

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/<string:bias_id>/')
def render(bias_id):
    return render_template('index.html')

@app.route('/<string:bias_id>/word_complete_api', methods=['GET', 'POST'])
def word_complete_api(bias_id):
    return ""

@app.route('/<string:bias_id>/phrase_complete_api', methods=['GET', 'POST'])
def phrase_complete_api(bias_id):

    # Get the json query
    query_text = request.form['text']
    
    
    # Extract last 50 words from query text
    text = " ".join(query_text.split(" ")[-50:])
  
    # Replace hyphens as they are not handled by word_tokenize
    text = text.replace("-", " - ")
  
    # Tokenize text 
    tokenized_text = word_tokenize(text)

    # Complete the last word, if the last word is a space the user has finishe writing the word so do not try to autocomplete
    if text[-1] != " ":
        last_word = complete_word(language_model=language_model, text = " ".join(tokenized_text[:-1]), final_word=tokenized_text[-1])
    else:
        last_word = ""

    word_completed_text = text + last_word
    print(last_word)
    # Pass the text and the completed last word through the language model for phrase completion
    try:
        # Pass through a neutral beam search engine 
        if bias_mapping[bias_id] == 'neu':
            phrase = beam_search_modified(
                language_model, word_completed_text, confidence=0.05, temperature=1.)
        # Pass through a postive or negative beam_search engine
        else:
            phrase = beam_search_modified_with_clf(
                language_model, classifier, bias_mapping[bias_id], text=word_completed_text, confidence=0.05)
    except:
        print("Failed to get an output from beam search")
        phrase = text
        pass
    # Replace full stops, commas, hyphens, slashes, inverted commas
    print(phrase)
    phrase = phrase.replace(" .", ".")
    phrase = phrase.replace(" ,", ",")
    phrase = phrase.replace(" /", "/")
    phrase = phrase.replace(" '", "'")
    phrase = phrase.replace(" - ", "-")
    phrase = phrase.replace(" n't", "n't")
    phrase = phrase.replace(" ?", "?")
    
    prediction = "empty"
    if last_word != "":
        prediction =  last_word + " " + phrase
    else:
        prediction = phrase

    # When the user finishes typing the whole word, add a space before the output
    if text[-1] != " ":
        if last_word == "":
            prediction = " " + prediction
    return prediction


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, use_reloader=False)