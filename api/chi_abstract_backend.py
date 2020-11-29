import datetime
import sys
import os
import re
import csv
import string
from nltk.tokenize import word_tokenize, TweetTokenizer
from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import beam_search_modified
from pathlib import Path
import pandas as pd
from random import choice
import nltk
nltk.download('punkt')

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object
learn_pos = load_learner('../models/movie/movie_reviews_pos_5epochs.pkl')
learn_neg = load_learner('../models/movie/movie_reviews_neg_5epochs.pkl')
learn_neut = load_learner('../models/movie/5epochs_imdb_lm.pkl')

bias_mapping = {
    'a': 'pos',
    'b': 'neg',
    'c': 'neu'
}

priority_list = []


def subtract(a, b):
    return "".join(a.rsplit(b))


@app.route('/')
def home():
    return render_template('compare.html')


@app.route('/<string:bias_id>/')
def a(bias_id):
    return render_template('chi_abstract_writer.html')


@app.route('/<string:bias_id>/predict', methods=['GET', 'POST'])
def predict(bias_id):
    text = request.form['text']

    # Consider only 300 characters
    text = text[-600:]

    # Saved for final processing
    text_later = text

    # Replace hyphens as they are not handled by word_tokenize
    text = text.replace("-", " - ")

    text_arr = word_tokenize(text)
    print("old text: " + text, sys.stderr)

    if (text[-1] == " "):
        rem_word = ""
    else:
        rem_word = autocomplete(text_arr[-1])

    text = text + rem_word
    print("old text + autocomplete: " + text, sys.stderr)

    try:
        # if bias_mapping[bias_id] == 'neu':
        #     prediction = beam_search_modified(
        #         learn_lm, text, confidence=0.05, temperature=1.)
        # else:
        #     prediction = beam_search_modified_with_clf(
        #         learn_lm, clf, bias_mapping[bias_id], text=text, confidence=0.0005)
        print(bias_id)
        if bias_mapping[bias_id] == 'neg':
            prediction = beam_search_modified(
                learn_neg, text, confidence=0.01, temperature=0.6)
        elif bias_mapping[bias_id] == 'pos':
            prediction = beam_search_modified(
                learn_pos, text, confidence=0.001, temperature=1.)
        else:
            prediction = beam_search_modified(
                learn_neut, text, confidence=0.01, temperature=0.6)

        prediction_arr = word_tokenize(prediction)
        print(prediction_arr, sys.stderr)
        # for item in prediction_arr:
        #     if item in string.punctuation:
        #         prediction_arr.remove(item)
        print(prediction_arr, sys.stderr)

        prediction = " ".join(prediction_arr[len(text_arr):])
        print(prediction, sys.stderr)
        if rem_word == "":
            prediction = " " + prediction
        else:
            prediction = rem_word + " " + prediction
        prediction = prediction.replace("`", "")
        for ele in prediction:
            if ele in string.punctuation:
                prediction = prediction.replace(ele, "")
#         prediction = re.sub("\s\s+" , " ", prediction)
#         prediction = ' '.join(word_tokenize(prediction))
        prediction = prediction.replace("  ", " ")
        if (text_later[-1] == " "):
            prediction = prediction[1:]
    except:
        prediction = ""

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
        fieldnames = ['Session', 'Bias', 'Review']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Session': datetime.datetime.now(),
                         'Bias': bias, 'Review': '"' + submitted_text + '"'})
    return '', 204


@app.route('/thanks')
def thanks():
    return render_template('thanks.html')


def autocomplete(text):
    text = text.lower()
#     if bias_mapping[biasx] == 'neg':
#         matches = [s for s in learn_neg.dls.vocab if s and s.startswith(text)]
#     elif bias_mapping[biasx] == 'pos':
#         matches = [s for s in learn_pos.dls.vocab if s and s.startswith(text)]
#     else:
    matches = [s for s in learn_neut.dls.vocab if s and s.startswith(text)]
    if len(matches) == 0:
        prediction = ""
    else:
        prediction = matches[0]
        prediction = prediction[len(text):]

    print("autocomplete: " + prediction, sys.stderr)

    if prediction == UNK:
        prediction = ""

    if prediction == "":
        return prediction

    return prediction


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=True)
