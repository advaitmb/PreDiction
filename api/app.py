from pathlib import Path
import random
import datetime
import string
import sys
import os
import re
import csv
import asyncio
import json

import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer

import pandas as pd
import time
from flask import Flask, request, render_template, jsonify

from fastai.text.all import *
# from inference import beam_search_modified, beam_search_modified_with_clf, complete_word

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
neutral_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
positive_model = AutoModelForCausalLM.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/api/models/gpt2-imdb-positive-sentiment")
negative_model = AutoModelForCausalLM.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/api/models/gpt2-imdb-negative-sentiment")
neutral_model.to('cuda')
positive_model.to('cuda')
negative_model.to('cuda')

logDump = []
maily = ''

# Initializing the FLASK API
app = Flask(__name__)

# Hidden bias mapping
bias_mapping = {
    'a': positive_model,
    'b': negative_model,
    'c': neutral_model
}

# Home Page
@app.route('/x')
def home():
    return render_template('index_baseline_mm.html')

@app.route('/done')
def done():
    return render_template('done.html')

@app.route('/')
def login():
    return render_template('email.html')

@app.route('/<string:bias_id>/')
def render(bias_id):
    return render_template('index.html')

@app.route('/<string:bias_id>/word_complete_api', methods=['GET', 'POST'])
def word_complete_api(bias_id):
    query_text = request.form['text']
    text = " ".join(query_text.split(" ")[-50:])
    tokenized_text = word_tokenize(text)
    word = complete_word_transformer(bias_mapping[bias_id], tokenizer, " ".join(tokenized_text[:-1]), tokenized_text[-1])
    return word 

@app.route('/<string:bias_id>/phrase_complete_api', methods=['GET', 'POST'])
def phrase_complete_api(bias_id):
    query_text = request.form['text']
    text = " ".join(query_text.split(" ")[-25:])
    tokenized_text = word_tokenize(text)

    # Replace hyphens as they are not handled by word_tokenize
    text = text.replace("-", " - ")
    phrase = generate_text_transformer(language_model=bias_mapping[bias_id], tokenizer=tokenizer, text=text, n_words_max=5)

    # Replace full stops, commas, hyphens, slashes, inverted commas
    phrase = phrase.replace(" .", ".")
    phrase = phrase.replace(" ,", ",")
    phrase = phrase.replace(" /", "/")
    phrase = phrase.replace(" '", "'")
    phrase = phrase.replace(" - ", "-")
    phrase = phrase.replace(" n't", "n't")
    phrase = phrase.replace(" ?", "?")
    phrase = phrase.replace(" !", "!")
    phrase = phrase.replace("!", "")
    phrase = phrase.replace("?", "")
    prediction = phrase

    return prediction

@app.route('/email', methods=['GET', 'POST'])
def email():
    global maily
    maily = request.form['mail']
    print(maily)
    return '', 204

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    data = request.get_json(force=True)
    with open('example_log.js', 'w') as outfile:
        data = 'let data = '+ str(data)
        json.dump(data, outfile)
    return '', 204
# def submit():
#     logDump = request.get_json()
#     print(logDump, sys.stdout)
#     global maily
#     logFile = maily + ".json"
#     print(logFile, sys.stdout)
#     write_json(logDump, logFile)
#     logDump = ['']
#     return '', 204

# ------------------ Functions & Inference --------------------- #

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def clean_html(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def clean_newlines(raw_text):
    return raw_text.replace('\n', '')

def complete_word_transformer(language_model, tokenizer, text, final_word):
    if len(text) == 0:
        return ''
    ids = tokenizer.encode(text)
    t = torch.LongTensor(ids)[None].to('cuda')
    logits = language_model.forward(t)[0][-1][-1]
    sorted_indices = torch.argsort(logits, descending=True)
    for tk_idx in sorted_indices:
        word = tokenizer.decode([tk_idx.cpu()]).strip()
        if word.lower().startswith(final_word):
            print(final_word,sys.stderr)
            if len(word.lower()) > len(final_word):
                return word[len(final_word):]
    return ""

def generate_text_transformer(language_model, tokenizer, text, n_words_max):
    text = text.strip()
    ids = tokenizer.encode(text)
    t = torch.LongTensor(ids)[None].to('cuda')
    phrase = language_model.generate(input_ids=t, num_beams=5, temperature=1.2,  max_length=(len(ids) + 10), skip_special_tokens=True, do_sample=True, repetition_penalty=1.2)
    prediction = phrase[0][t.size(1):].cpu()
    prediction = prediction[prediction!=50256]
    return clean_newlines(clean_html(tokenizer.decode(prediction.numpy())[1:]))

# ------------------ Functions & Inference --------------------- #

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000, debug=True)
