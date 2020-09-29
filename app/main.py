import numpy as np
from flask import Flask, request, render_template, jsonify, g, url_for, abort
import pickle
from fastai.text import *
import os

cwd = os.getcwd()
path = cwd + '/models'

#Initializing the FLASK API
app = Flask(__name__)

#Loading the saved model using fastai's load_learner method
model = load_learner(path, '5epochslearner_v1.pkl')

@app.route('/')
def home():
    return render_template('interactive.html')


@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    search = request.args.get('q')
#    query = db_session.query(Movie.title).filter(Movie.title.like('%' + str(search) + '%'))
#    results = [mv[0] for mv in query.all()]
    out = model.beam_search(str(search), n_words=1)
    results = [out[len(str(search)):]]
    return jsonify(matching_results=results)


if __name__ == "__main__":
    app.run(debug=True)
