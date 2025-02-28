import random
import json as json
import numpy as np
import torch
from joblib import Parallel, delayed
import torch
# nlp = spacy.blank("en")
import re
import random


def _process_agreement(agreement, labels, omit_token):
    def _find_span(s_spans, o_start):
        l_s = 0
        r_s = len(s_spans)
        while r_s - l_s > 1:
            m = (l_s+r_s) // 2
            if s_spans[m][0] <= o_start:
                l_s = m
            else:
                r_s = m
        return l_s
    def _process_elipsis(text, spans, omit_token):
        omits = re.finditer('_+', text)
        text = re.sub('_+', omit_token, text)
        for omit in omits:
            o_span = omit.span()
            d = o_span[1] - o_span[0] - len(omit_token)
            o_span_idx = _find_span(spans, o_span[0])
            spans[o_span_idx][1] = spans[o_span_idx][1] - d
            for i in range(o_span_idx+1, len(spans)):
                spans[i][0], spans[i][1] = spans[i][0] - d, spans[i][1] - d
        return text, spans
    
    res = []
    text = agreement['text']
    spans = agreement['spans']
    if omit_token is not None:
        text, spans = _process_elipsis(text, spans, omit_token)
    for label, hyp_data in labels.items():
        context = ' '.join([hyp_data['hypothesis'], text])
        ans_data = agreement['annotation_sets'][0]['annotations'][label]
        ans = ans_data['choice']
        supp_spans = [(0, len(hyp_data['hypothesis'])+1)] + [(spans[i][0] + len(hyp_data['hypothesis'])+1, spans[i][1] + len(hyp_data['hypothesis'])+1) for i in ans_data['spans']]
        res.append({
            'context': context,
            'text_id': agreement['id'],
            'answer': ans,
            'hyp_id': int(label[4:]),
            'supp_spans': supp_spans
        })
    return res
def process_file(file_path, omit_token=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    examples = []
    eval_examples = {}
    # data = [data[i] for  i in range(30)]
    labels = data['labels']
    outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_agreement)(agreement, labels, omit_token=omit_token) for agreement in data['documents'])
    outputs = [item for agr in outputs for item in agr]
    random.shuffle(outputs)
    return outputs