from fastai.text.all import *
from pathlib import Path
import pandas as pd
from random import choice
import re
import json

def get_next_word(learn, text, no_unk=True, temperature=1., decoder=decode_spec_tokens):
    learn.model.reset()
    idxs = idxs_all = learn.dls.test_dl([text]).items[0]
    if no_unk: unk_idx = learn.dls.vocab.index(UNK)
    preds,_ = learn.get_preds(dl=[(idxs[None],)])
    res = preds[0][-1]
    if no_unk: res[unk_idx] = 0.
    if temperature != 1.: res.pow_(1 / temperature)
    idx = torch.multinomial(res, 3)
    num = learn.dls.train_ds.numericalize
    tokens = [num.vocab[i] for i in idx if num.vocab[i] not in [BOS, PAD]]
    return decoder(tokens)

if __name__ == '__main__':
    learn = load_learner('../models/design/4epochslearner.pkl')
    
    vocab = learn.dls.vocab
    with open('data.json', 'w') as fp:
        json.dump(vocab, fp)