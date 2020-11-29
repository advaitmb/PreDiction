from fastai.text.all import *
from pathlib import Path
import pandas as pd
from random import choice
import re
import json


def get_next_word(learn, text, no_unk=True, temperature=1., decoder=decode_spec_tokens):
    learn.model.reset()
    idxs = idxs_all = learn.dls.test_dl([text]).items[0]
    if no_unk:
        unk_idx = learn.dls.vocab.index(UNK)
    preds, _ = learn.get_preds(dl=[(idxs[None],)])
    res = preds[0][-1]
    if no_unk:
        res[unk_idx] = 0.
    if temperature != 1.:
        res.pow_(1 / temperature)
    idx = torch.multinomial(res, 3)
    num = learn.dls.train_ds.numericalize
    tokens = [num.vocab[i] for i in idx if num.vocab[i] not in [BOS, PAD]]
    return decoder(tokens)


def beam_search_modified_with_clf(learn, clf, bias, text: str, confidence: float, no_unk: bool = True, no_punct: bool = True, top_k: int = 10, beam_sz: int = 100, temperature: float = 1.,
                                  sep: str = ' ', decoder=decode_spec_tokens):
    bias_map = {0: 'neg', 1: 'pos'}
    learn.model.reset()
    learn.model.eval()

    idx = learn.dls.test_dl([text]).items[0][None]
    nodes = None
    nodes = idx.clone()
    scores = idx.new_zeros(1).float()
    if no_unk:
        unk_idx = learn.dls.vocab.index(UNK)

    with torch.no_grad():
        while(torch.exp(-scores[0]) > confidence):
            out = F.log_softmax(learn.model(idx)[0][:, -1], dim=-1)

            if no_unk:
                out[:, unk_idx] = -float('Inf')

            values, indices = out.topk(top_k, dim=-1)

            scores = (-values + scores[:, None]).view(-1)
            indices_idx = torch.arange(0, nodes.size(0))[:, None].expand(
                nodes.size(0), top_k).contiguous().view(-1)
            sort_idx = scores.argsort()[:beam_sz]
            scores = scores[sort_idx]

            nodes = torch.cat([nodes[:, None].expand(nodes.size(0), top_k, nodes.size(1)),
                               indices[:, :, None].expand(nodes.size(0), top_k, 1), ], dim=2)
            nodes = nodes.view(-1, nodes.size(2))[sort_idx]
            learn.hidden = [(h[0][:, indices_idx[sort_idx], :], h[1]
                             [:, indices_idx[sort_idx], :]) for h in learn.model[0].hidden]
            idx = nodes[:, -1][:, None]

        num = learn.dls.train_ds.numericalize
        sep = learn.dls.train_ds.tokenizer.sep

        if temperature != 1.:
            scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 20)
#         node_idx = torch.exp(-scores).topk(20, dim=-1).indices
        phrases = []
        biased_idx = []
        for idx in node_idx:
            tokens = [num.vocab[i]
                      for i in nodes[idx] if num.vocab[i] not in [BOS, PAD]]
            phrase = sep.join(decoder(tokens))
            phrases.append(phrase)
            biased_idx.append(idx)

        dl = clf.dls.test_dl(phrases, rm_type_tfms=None, num_workers=0)
        inp, preds, _, dec_preds = clf.get_preds(
            dl=dl, with_input=True, with_decoded=True)

        for i in range(len(inp)):
            if bias_map[dec_preds[i].item()] == bias:
                tokens = [num.vocab[i]
                          for i in inp[i][1:] if num.vocab[i] not in [BOS, PAD]]
                return sep.join(decoder(tokens))

    return ""


def beam_search_modified(learn, text: str, confidence: float, no_unk: bool = True, no_punct: bool = True, top_k: int = 10, beam_sz: int = 100, temperature: float = 1.,
                         sep: str = ' ', decoder=decode_spec_tokens):
    learn.model.reset()
    learn.model.eval()
    idx = learn.dls.test_dl([text]).items[0][None]
    nodes = None
    nodes = idx.clone()
    scores = idx.new_zeros(1).float()
    if no_unk:
        unk_idx = learn.dls.vocab.index(UNK)

    with torch.no_grad():
        while(torch.exp(-scores[0]) > confidence):
            out = F.log_softmax(learn.model(idx)[0][:, -1], dim=-1)

            if no_unk:
                out[:, unk_idx] = -float('Inf')

            values, indices = out.topk(top_k, dim=-1)

            scores = (-values + scores[:, None]).view(-1)
            indices_idx = torch.arange(0, nodes.size(0))[:, None].expand(
                nodes.size(0), top_k).contiguous().view(-1)
            sort_idx = scores.argsort()[:beam_sz]
            scores = scores[sort_idx]

            nodes = torch.cat([nodes[:, None].expand(nodes.size(0), top_k, nodes.size(1)),
                               indices[:, :, None].expand(nodes.size(0), top_k, 1), ], dim=2)
            nodes = nodes.view(-1, nodes.size(2))[sort_idx]
            learn.hidden = [(h[0][:, indices_idx[sort_idx], :], h[1]
                             [:, indices_idx[sort_idx], :]) for h in learn.model[0].hidden]
            idx = nodes[:, -1][:, None]

        if temperature != 1.:
            scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        num = learn.dls.train_ds.numericalize
        tokens = [num.vocab[i]
                  for i in nodes[node_idx][1:] if num.vocab[i] not in [BOS, PAD]]
        sep = learn.dls.train_ds.tokenizer.sep
        return sep.join(decoder(tokens))


def beam_search(learn, text: str, n_words: int, no_unk: bool = True, top_k: int = 10, beam_sz: int = 100, temperature: float = 1.,
                sep: str = ' ', decoder=decode_spec_tokens):
    learn.model.reset()
    learn.model.eval()
    idx = learn.dls.test_dl([text]).items[0][None]
    nodes = None
    nodes = idx.clone()
    scores = idx.new_zeros(1).float()
    if no_unk:
        unk_idx = learn.dls.vocab.index(UNK)
    with torch.no_grad():
        for k in progress_bar(range(n_words), leave=False):
            out = F.log_softmax(learn.model(idx)[0][:, -1], dim=-1)
            if no_unk:
                out[:, unk_idx] = -float('Inf')
            values, indices = out.topk(top_k, dim=-1)
            scores = (-values + scores[:, None]).view(-1)
            indices_idx = torch.arange(0, nodes.size(0))[:, None].expand(
                nodes.size(0), top_k).contiguous().view(-1)
            sort_idx = scores.argsort()[:beam_sz]
            scores = scores[sort_idx]
            nodes = torch.cat([nodes[:, None].expand(nodes.size(0), top_k, nodes.size(1)),
                               indices[:, :, None].expand(nodes.size(0), top_k, 1), ], dim=2)
            nodes = nodes.view(-1, nodes.size(2))[sort_idx]
            learn.hidden = [(h[0][:, indices_idx[sort_idx], :], h[1]
                             [:, indices_idx[sort_idx], :]) for h in learn.model[0].hidden]
            idx = nodes[:, -1][:, None]
        if temperature != 1.:
            scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        num = learn.dls.train_ds.numericalize
        tokens = [num.vocab[i]
                  for i in nodes[node_idx][1:] if num.vocab[i] not in [BOS, PAD]]
        sep = learn.dls.train_ds.tokenizer.sep
        return sep.join(decoder(tokens))


if __name__ == '__main__':
    learn = load_learner('./movie_reviews_pos_5epochs.pkl')
