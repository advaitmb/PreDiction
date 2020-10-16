from pathlib import Path
import pandas as pd
from random import choice
import re
import json
import sys

if __name__ == '__main__':
    # text = "Here, I/We think you robot-man are mistaken."
    text = "Something are best left unsaid (1)"
    print(text)
    base_string_length = len(text)

    # Add spaces before and after all of these punctuation marks 
    text = re.sub('([.,\/#!$%\^&\*;:{}=\-_`~()])', r' \1 ', text)
    print(text)

    # Replace any places with 2 spaces by one space 
    text = re.sub('\s{2,}', ' ', text)
    print(text)

    # prediction = beam_search_modified(learn, text, confidence=0.1, temperature=1.)
    prediction = text
    # Remove space before punctuations
    prediction = re.sub('\s([.,#!$%\^&\*;:{}=_`~](?:\s|$))', r'\1', prediction)
    print(prediction)
    # Special cases 
    prediction = prediction.replace(" - ", "-")
    prediction = prediction.replace(" / ", "/")
    prediction = prediction.replace(" ( ", " (")
    prediction = prediction.replace(" ) ", ") ")

    prediction = prediction[:base_string_length]
    print(prediction, file=sys.stderr)
