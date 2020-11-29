import datetime
from os import remove
import sys
import os
import re
import csv
from nltk.tokenize import word_tokenize, TweetTokenizer
from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word, beam_search, beam_search_modified
from pathlib import Path
import pandas as pd
from random import choice
import nltk
nltk.download('punkt')

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object
learn = load_learner('../models/design/4epochslearner.pkl')


def subtract(a, b):
    return "".join(a.rsplit(b))


@app.route('/')
def home():
    return render_template('simplify_new.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    text_later = text
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
        prediction = beam_search_modified(
            learn, text, confidence=0.01, temperature=1)

        prediction_arr = word_tokenize(prediction)

        prediction = " ".join(prediction_arr[len(text_arr):])
        print(prediction, sys.stderr)
        prediction = re.sub(
            '\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
        prediction = prediction.replace(" - ", "-")
        prediction = prediction.replace(" -", "")
        prediction = prediction.replace(" / ", "/")
        prediction = prediction.replace(" ( ", " (")
        prediction = prediction.replace(" ) ", ") ")
        prediction = prediction.replace(" .", ".")
        prediction = prediction.replace(" ' ", "'")
        prediction = prediction.replace(' " ', '"')
        prediction = prediction.replace('  ', ' ')

        prediction = rem_word + " " + prediction

        if (text_later[-1] == " "):
            prediction = prediction[1:]
    except:
        prediction = ""

    predicted = {
        "predicted": prediction
    }

    return jsonify(predicted=predicted)


# @app.route('/b/predict', methods=['GET', 'POST'])
# def predict():
#     text = request.form['text']
#     text_later = text
#     text = text.replace("-", " - ")

#     text_arr = word_tokenize(text)
#     print("old text: " + text, sys.stderr)

#     if (text[-1] == " "):
#         rem_word = ""
#     else:
#         rem_word = autocomplete(text_arr[-1])

#     text = text + rem_word
#     print("old text + autocomplete: " + text, sys.stderr)

#     try:
#         prediction = beam_search_modified(
#             learn, text, confidence=0.01, temperature=1)

#         prediction_arr = word_tokenize(prediction)

#         prediction = " ".join(prediction_arr[len(text_arr):])
#         prediction = re.sub(
#             '\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
#         prediction = prediction.replace(" - ", "-")
#         prediction = prediction.replace(" / ", "/")
#         prediction = prediction.replace(" ( ", " (")
#         prediction = prediction.replace(" ) ", ") ")
#         prediction = prediction.replace(" .", ".")
#         prediction = prediction.replace(" ' ", "'")
#         prediction = prediction.replace(' " ', '"')

#         prediction = rem_word + " " + prediction

#         if (text_later[-1] == " "):
#             prediction = prediction[1:]
#     except:
#         prediction = ""

#     predicted = {
#         "predicted": prediction
#     }

#     return jsonify(predicted=predicted)


def autocomplete(text):
    text = text.lower()
    matches = [s for s in learn.dls.vocab if s and s.startswith(text)]
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
