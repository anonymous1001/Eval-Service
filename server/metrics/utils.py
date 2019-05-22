#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
stopset = frozenset(stopwords.words('english'))


# def to_unicode(object):
#     if isinstance(object, unicode):
#         return object
#     elif isinstance(object, bytes):
#         return object.decode("utf8")
#     else:
#         print(str(object))
#         return object


def stem_word(word):
    return stemmer.stem(normalize_word(word))


def normalize_word(word):
    return word.lower()


def get_len(element):
    return len(tokenizer.tokenize(element))


def sentence_tokenizer(sentence):
    return tokenizer.tokenize(sentence)


def get_ngrams(sentence, N):
    tokens = tokenizer.tokenize(sentence.lower())
    clean = [stemmer.stem(token) for token in tokens]
    return [gram for gram in ngrams(clean, N)]


def get_words(sentence, stem=True):
    if stem:
        words = [stemmer.stem(r) for r in tokenizer.tokenize(sentence)]
        return [normalize_word(w) for w in words]
    else:
        return [normalize_word(w) for w in tokenizer.tokenize(sentence)]

tokenizer = RegexpTokenizer(r'\w+')
stopset = set(stopwords.words('english'))
    
from metrics import S3
from metrics import ROUGE
from metrics import JS_eval
from metrics import word_embeddings
from metrics import wmd
from hausdorff import hausdorff

def ROUGE_N(summary, references, external_info):
    N = external_info['N']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_n(summary, references_text, N, alpha=0)

def ROUGE_L(summary, references, external_info=None):
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_l(summary, references_text, alpha=0)

def ROUGE_WE(summary, references, external_info):
    N = external_info['N']
    we = external_info['we']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_n_we(summary, references_text, we, N, alpha=0)

def JS_N(summary, references, external_info):
    N = external_info['N']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return -JS_eval.JS_eval(summary, references_text, N)


s3_model_folders = "metrics/s3_models"

def S3_metric(summary, references, external_info):
    we = external_info['we']
    pyr = external_info['pyr']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    if pyr: ## Return Pyramid version of S3: S3_pyr
        return S3.S3(references_text, summary, we, s3_model_folders)[0]
    else: ## Return responsivness version of S3: S3_resp
        return S3.S3(references_text, summary, we, s3_model_folders)[1]

def text_to_emb(text, we, N):
    if N == 1: 
        words = [w.lower() for sent in text for w in tokenizer.tokenize(sent) if w not in stopset]
        embs = [we[w] for w in words if w in we]
    else:
        words = [w.lower() for sent in text for w in tokenizer.tokenize(sent)]
        embs = []
        for i, w in enumerate(words):
            if i < len(words)-1:
                w_next = words[i+1]
                if w in we and w_next in we:
                    embs.append(np.concatenate((we[w], we[w_next])))
    return embs

def text_to_elmo(text, elmo, N):
    if N == 1: 
        words = [w.lower() for sent in text for w in tokenizer.tokenize(sent) if w not in stopset]
        embs = list(elmo.embed_sentence(words)[2])
    else:
        embs = None
    return embs    

def text_to_bert(text, bert, N):
    if N == 1: 
        words = [w.lower() for sent in text for w in tokenizer.tokenize(sent) if w not in stopset]
        embs = list(bert.encode(words))
    else:
        embs = None
    return embs

def WMS_N_AVG(summary, references, external_info, embedding_type):
    N = external_info['N']
    we = external_info['we']
    level = external_info['level']
    
    if level == 'words':
        if embedding_type == "elmo":
            summ_embs = text_to_elmo(summary, we, N)
        elif embedding_type == "bert":
            summ_embs = text_to_bert(summary, we, N)
        else:
            summ_embs = text_to_emb(summary, we, N)
    if level in ['sentences', 'documents']:
        summ_embs = we.encode(summary, tokenize=True)

    wms_total = 0
    for ref in references:
        if level == 'words':
            if embedding_type == "elmo":
                ref_embs = text_to_elmo(ref['text'], we, N)
            elif embedding_type == "bert":
                ref_embs = text_to_bert(ref['text'], we, N)
            else:
                ref_embs = text_to_emb(ref['text'], we, N)           
        if level in ['sentences', 'documents']:
            ref_embs = we.encode(ref['text'], tokenize=True)                    
        wms_total += wmd.wms(summ_embs, ref_embs)
        wms_total += wmd.wms(ref_embs, summ_embs)
    return wms_total / (2. * len(references))

def HD_N_AVG(summary, references, external_info):
    N = external_info['N']
    we = external_info['we']
    level = external_info['level']
    
    if level == 'words':
        summ_embs = text_to_emb(summary, we, N)
    if level in ['sentences', 'documents']:
        summ_embs = we.encode(summary, tokenize=True)
    
    hd_total = 0
    for ref in references:
        if level == 'words':
            ref_embs = text_to_emb(ref['text'], we, N)           
        if level in ['sentences', 'documents']:
            ref_embs = we.encode(ref['text'], tokenize=True)        
        hd_total += hausdorff(summ_embs, ref_embs)
        hd_total += hausdorff(ref_embs, summ_embs)
    return hd_total / (2. * len(references))

#def WMS_N_MAX(summary, references, external_info):
#    N = external_info['N']
#    we = external_info['we']
#    summ_embs = text_to_emb(summary, we, N)
#
#    wms_max = 0
#    for ref in references:
#        ref_embs = text_to_emb(ref['text'], we, N)
#        wms_l = wmd.wms(summ_embs, ref_embs)
#        wms_l += wmd.wms(ref_embs, summ_embs)
#        if wms_l > wms_max:
#            wms_max = wms_l
#    return wms_max / 2.

import scipy.stats as stats
#from tqdm import tqdm

import multiprocessing
from functools import partial

def micro_averaging(annot, target, eval_metric, references, external_info=None):
    if len(annot['text']) > 0 :
        target_scores = float(annot[target])
        prediction_scores = eval_metric(annot['text'], references, external_info)
        return [target_scores, prediction_scores]

def multi_micro_averaging(pool, dataset, target, eval_metric, external_info=None):
    correlations = []
    for _ in dataset:
        k, v = _
        references = v['references']
        _micro_averaging = partial(micro_averaging, target=target, eval_metric=eval_metric, references = references, external_info=external_info)
        scores = np.array(filter(None, pool.map(_micro_averaging, v['annotations'])), dtype=np.float32)
        target_scores, prediction_scores = scores[:,0], scores[:,1]                     
        correlations.append(stats.kendalltau(target_scores, prediction_scores)[0])
    return correlations    


#def micro_averaging(dataset, target, eval_metric, external_info=None):
#    correlations = []
##    for k, v in enum(dataset.items()):
#    k, v = dataset
#    references = v['references']
#    target_scores, prediction_scores = [], []
#    for annot in v['annotations']:            
#        if len(annot['text']) > 0 :
#            target_scores.append(float(annot[target]))
#            prediction_scores.append(eval_metric(annot['text'], references, external_info))
#    correlations.append(stats.kendalltau(target_scores, prediction_scores)[0])
#    return np.array(correlations)
#
#
#def multi_micro_averaging(dataset, target, eval_metric, external_info=None):
#    pool = multiprocessing.Pool(processes=16)
#    _micro_averaging = partial(micro_averaging, target=target, eval_metric=eval_metric, external_info=external_info)
#    correlations = pool.map(_micro_averaging, dataset)
#    pool.terminate()                
#    return correlations
