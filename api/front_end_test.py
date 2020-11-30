from pathlib import Path
import random
import datetime
import string
import sys
import os
import re
import csv
from time import sleep

import pandas as pd

from flask import Flask, request, render_template, jsonify


# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object


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
    sleep(0.1)
    
    random_phrases = ["is awesome", "ther than this", "is going well", "shouln't do this"]
    return random.choice(random_phrases)


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)