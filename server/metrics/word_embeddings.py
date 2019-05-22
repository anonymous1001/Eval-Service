#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import gzip
from os.path import join, exists
import io
import torch
from metrics.infersent.models import InferSent

def _convert_to_numpy(vector):
    return np.array([float(x) for x in vector])


#def load_embeddings(filepath):
#    dict_embedding = {}
#    with gzip.open(filepath, 'rb') as f:
#        for line in f:
#            line = line.decode('utf-8').rstrip().split(" ")
#            key = line[0]
#            vector = line[1::]
#            dict_embedding[key.lower()] = _convert_to_numpy(vector)
#    return dict_embedding


def load_embeddings(model_type, filepath, modelpath = None):    
    embedding_model = {}    
    if model_type == 'glove':    
        for line in open(filepath, 'r'):
            tmp = line.strip().split()
            word, vec = tmp[0], map(float, tmp[1:])
            if word not in embedding_model:
                embedding_model[word] = vec
                
    if model_type == 'deps':
        with gzip.open(filepath, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').rstrip().split(" ")
                word = line[0]
                vec = line[1::]
                embedding_model[word.lower()] = _convert_to_numpy(vec)        

    if model_type == 'ft':
        fin = io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        embedding_model = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedding_model[tokens[0]] = map(float, tokens[1:])       
    
    if model_type == 'infersent':
        embedding_model = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}).cuda()
        embedding_model.load_state_dict(torch.load(modelpath))  
        embedding_model.set_w2v_path(filepath)
        embedding_model.build_vocab_k_words(K=100000)
    return embedding_model






