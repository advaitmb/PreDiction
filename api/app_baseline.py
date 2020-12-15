from pathlib import Path
import random
import datetime
import string
import sys
import os
import re
import csv
import asyncio
import pandas as pd
import time
from flask import Flask, request, render_template, jsonify
import json

logDump = []
maily = ''

# # Initializing the FLASK API
app = Flask(__name__)

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

def clean_html(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def clean_newlines(raw_text):
    return raw_text.replace('\n', '')

@app.route('/email', methods=['GET', 'POST'])
def email():
    global maily 
    maily = request.form['mail']
    print(maily, sys.stdout)
    return '', 204

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    logDump = request.get_json()
    print(logDump, sys.stdout)
    global maily
    logFile = maily + ".json"
    print(logFile, sys.stdout)
    write_json(logDump, logFile)
    logDump = ['']
    return '', 204

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000, debug=True)
