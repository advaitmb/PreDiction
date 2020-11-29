import datetime
import sys
import os
import re
import csv
import string
from nltk.tokenize import word_tokenize, TweetTokenizer
from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import beam_search_modified, beam_search_modified_with_clf
from pathlib import Path
import pandas as pd
from random import choice
import nltk
nltk.download('punkt')

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object
learn_lm = load_learner('../models/movie/5epochs_imdb_lm.pkl')
clf = load_learner('../models/movie/imdb_sentiment_classifier.pkl')
learn_autocomplete = load_learner('../models/movie/char_lm.pkl')

bias_mapping = {
    'a': 'pos',
    'b': 'neg',
    'c': 'neu'
}


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

    # try:
    #     if bias_mapping[bias_id] == 'neu':
    #         prediction = beam_search_modified(
    #             learn_lm, text, confidence=0.05, temperature=1.)
    #     else:
    #         prediction = beam_search_modified_with_clf(
    #             learn_lm, clf, bias_mapping[bias_id], text=text, confidence=0.0005)
    #     print(bias_id)

    # prediction_arr = word_tokenize(prediction)
    # print(prediction_arr, sys.stderr)
    # # for item in prediction_arr:
    # #     if item in string.punctuation:
    # #         prediction_arr.remove(item)
    # print(prediction_arr, sys.stderr)

    # prediction = " ".join(prediction_arr[len(text_arr):])
    # print(prediction, sys.stderr)

#         if rem_word == "":
#             prediction = " " + prediction
#         else:
#             prediction = rem_word + " " + prediction
#         prediction = prediction.replace("`", "")

#         for ele in prediction:
#             if ele in string.punctuation:
#                 prediction = prediction.replace(ele, "")
# #         prediction = re.sub("\s\s+" , " ", prediction)
# #         prediction = ' '.join(word_tokenize(prediction))
#         prediction = prediction.replace("  ", " ")
#         if (text_later[-1] == " "):
#             prediction = prediction[1:]
#     except:
#         prediction = ""

    predicted = {
        "predicted": rem_word
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


def predict_autocomplete(learn, text, n_words=1, no_unk=True, temperature=1., min_p=None, no_bar=False,
                         decoder=decode_spec_tokens, only_last_word=False):
    "Return `text` and the `n_words` that come after"
    learn.model.reset()
    idxs = idxs_all = learn.dls.test_dl([text]).items[0].to(learn.dls.device)
    if no_unk:
        unk_idx = learn.dls.vocab.index(UNK)
    for _ in (range(n_words) if no_bar else progress_bar(range(n_words), leave=False)):
        with learn.no_bar():
            preds, _ = learn.get_preds(dl=[(idxs[None],)])
        res = preds[0][-1]
        if no_unk:
            res[unk_idx] = 0.
        if min_p is not None:
            if (res >= min_p).float().sum() == 0:
                warn(
                    f"There is no item with probability >= {min_p}, try a lower value.")
            else:
                res[res < min_p] = 0.
        if temperature != 1.:
            res.pow_(1 / temperature)
        idx = torch.multinomial(res, 1).item()
        idxs = idxs_all = torch.cat([idxs_all, idxs.new([idx])])
        if only_last_word:
            idxs = idxs[-1][None]

    num = learn.dls.train_ds.numericalize
    tokens = [num.vocab[i] for i in idxs_all if num.vocab[i] not in [BOS]]
    sep = learn.dls.train_ds.tokenizer.sep
    return sep.join(decoder(tokens))


def preprocess_autocomplete(text):
    return " ".join('%'.join(text.split())).replace('%', PAD)


def autocomplete(text):
    text = text.lower()

    preprocessed_text = preprocess_autocomplete(text)

    prediction = predict_autocomplete(
        learn_autocomplete, preprocessed_text, n_words=10)

    prediction = prediction.replace(' ', '')
    prediction = prediction.replace('xxpad', ' ')
    prediction = prediction[len(text):]
    prediction = prediction.split(" ")[0]

    if prediction == UNK:
        prediction = ""

    if prediction == "":
        return prediction

    return prediction


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080, debug=True)
