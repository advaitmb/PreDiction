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
# from inference import beam_search_modified, beam_search_modified_with_clf, complete_word

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("gpt2")
neutral_model = AutoModelWithLMHead.from_pretrained("lvwerra/gpt2-imdb")
positive_model = AutoModelWithLMHead.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/nbs/gpt2-imdb-positive-sentiment")
negative_model = AutoModelWithLMHead.from_pretrained("/home/advaitmb/notebooks/projects/PreDiction/nbs/gpt2-imdb-negative-sentiment")
neutral_model.to('cuda')
positive_model.to('cuda')
negative_model.to('cuda')

lstm_model = load_learner('5epochs_imdb_lm.pkl')

# Initializing the FLASK API
app = Flask(__name__)

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
    logits = language_model.forward(t).logits[0][-1]
    sorted_indices = torch.argsort(logits, descending=True)
    
    for tk_idx in sorted_indices:
        word = tokenizer.decode(tk_idx.cpu().numpy())
        if word.lower().startswith(final_word):
            if word.lower() != final_word:
                return word[len(final_word):]
    return ""

def complete_word(language_model, text, final_word, no_unk=True, decoder=decode_spec_tokens, temperature=1.):
    
    language_model.model.reset()
    language_model.model.eval()
    numericalize = language_model.dls.train_ds.numericalize
    idxs = idxs_all = language_model.dls.test_dl([text]).items[0].to(language_model.dls.device)
    if no_unk: unk_idx = language_model.dls.vocab.index(UNK)
        
    with torch.no_grad():
        with language_model.no_bar(): preds,_ = language_model.get_preds(dl=[(idxs[None],)])
        res = preds[0][-1]
        sorted_indices = torch.argsort(res, descending=True)
        sorted_tokens = [numericalize.vocab[i] for i in sorted_indices if numericalize.vocab[i] not in ['xxbos', 'xxpad', 'xxunk', 'xxmaj']]
        candidate_tokens = [token for token in sorted_tokens if token.lower().startswith(final_word)]
        if (len(candidate_tokens) <= 0):
            return ""
        candidate_indices = [numericalize([token]).item() for token in candidate_tokens]
        candidate_scores = res[candidate_indices]
        if temperature != 1.: candidate_scores.div_(temperature)
        selected_index = torch.multinomial(candidate_scores, 1).item()
         
    return candidate_tokens[selected_index][len(final_word): ]

def generate_text_transformer(language_model, tokenizer, text, n_words_max):
    text = text.strip()
    ids = tokenizer.encode(text)
    t = torch.LongTensor(ids)[None].to('cuda')
    phrase = language_model.generate(input_ids=t, num_beams=5, temperature=1.2,  max_length=(len(ids) + 10), skip_special_tokens=True, do_sample=True, repetition_penalty=1.2)
    prediction = phrase[0][t.size(1):].cpu()
    prediction = prediction[prediction!=50256]
    return clean_newlines(clean_html(tokenizer.decode(prediction.numpy())[1:]))

@app.route('/<string:bias_id>/word_complete_api', methods=['GET', 'POST'])
def word_complete_api(bias_id):
    query_text = request.form['text']

    text = " ".join(query_text.split(" ")[-50:])

    tokenized_text = word_tokenize(text)

    word = complete_word(lstm_model, " ".join(text.split(" ")[:-1]), tokenized_text[-1])
    # word = complete_word_transformer(transformer_model, tokenizer, " ".join(tokenized_text[:-1]), tokenized_text[-1])
    print("word: ", word, sys.stdout)

    return word 

@app.route('/<string:bias_id>/phrase_complete_api', methods=['GET', 'POST'])
def phrase_complete_api(bias_id):
    # Get the json query
    query_text = request.form['text']
    
    # Extract last 50 words from query text
    text = " ".join(query_text.split(" ")[-50:])
  
    # Tokenize text 
    tokenized_text = word_tokenize(text)

    # Replace hyphens as they are not handled by word_tokenize
    text = text.replace("-", " - ")
    print("printing in the main loop:", text)
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
    print("phrase: ", phrase, sys.stdout)
    prediction = phrase

    return prediction

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000, debug=True)
