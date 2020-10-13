from flask import Flask, request, render_template, jsonify
from fastai.text.all import *
from inference import get_next_word, beam_search
from pathlib import Path
import pandas as pd
from random import choice
import re
import os
import sys

# Initializing the FLASK API
app = Flask(__name__)

#  Load learner object 
learn = load_learner('../models/design/4epochslearner.pkl')

@app.route('/')
def home():
    return render_template('faded.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # search = request.args.get('q')
    text = request.form['text']
#    query = db_session.query(Movie.title).filter(Movie.title.like('%' + str(search) + '%'))
#    results = [mv[0] for mv in query.all()]
    prediction = beam_search(learn, text, n_words=2, temperature=1.2)

    predicted = {
        "predicted": prediction
    }
    # predicted = {str(key): value for key, value in result.items()}
    return jsonify(predicted=predicted)


if __name__ == "__main__":
    app.run(debug=True)
