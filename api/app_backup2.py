from pathlib import Path
import random
import datetime
import string
import sys
import os
import re
import csv
import asyncio

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
positive_model = AutoModelForCausalLM.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/nbs/gpt2-imdb-positive-sentiment")
negative_model = AutoModelForCausalLM.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/nbs/gpt2-imdb-negative-sentiment")
neutral_model.to('cuda')
positive_model.to('cuda')
negative_model.to('cuda')

lstm_model = load_learner('5epochs_imdb_lm.pkl')
vocab = lstm_model.dls.vocab
logDump = []
maily = ''

# Initializing the FLASK API
app = Flask(__name__)

# Hidden bias mapping
bias_mapping = {
    'a': 'pos',
    'b': 'neg',
    'c': 'neu'
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
    # query_text = request.get_json()['text']

    text = " ".join(query_text.split(" ")[-50:])

    tokenized_text = word_tokenize(text)

#     word = complete_word(lstm_model, " ".join(text.split(" ")[:-1]), tokenized_text[-1])
#     print(word, sys.stdout)
    word = complete_word_transformer(negative_model, tokenizer, vocab, " ".join(tokenized_text[:-1]), tokenized_text[-1])

    return word 

@app.route('/<string:bias_id>/phrase_complete_api', methods=['GET', 'POST'])
def phrase_complete_api(bias_id):
    # Get the json query
    query_text = request.form['text']
    # query_text = request.get_json()['text']
    
    # Extract last 50 words from query text
    text = " ".join(query_text.split(" ")[-50:])
  
    # Tokenize text 
    tokenized_text = word_tokenize(text)

    # Replace hyphens as they are not handled by word_tokenize
    text = text.replace("-", " - ")
    phrase = generate_text_transformer(language_model=negative_model, tokenizer=tokenizer, text=text, n_words_max=5)

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



@app.route('/log', methods=['GET', 'POST'])
def log():
    logInstance = request.get_json()
    logDump.append(logInstance)
    logFile = maily + ".json"
    write_json(logDump, logFile)
    return '', 204


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    logDump = request.get_json()

    global maily
    logFile = maily + ".json"
    write_json(logDump, logFile)
    logDump.clear()
    return '', 204

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

def complete_word_transformer(language_model, tokenizer, vocab, text, final_word):
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
            # and word.lower() in vocab:
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
