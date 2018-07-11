"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

from learn import models
from constants import *
import datasets
import persistence
import numpy as np

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    Y = len(dicts['ind2c'])
    if args.model == "rnn":
        model = models.VanillaRNN(Y, args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        if args.lmbda is not None:
            assert args.description_dir is not None
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout)
    elif args.model == "conv_attn_plus_GRAM":
        filter_size = int(args.filter_size)
        assert args.annotation_type is not None
        assert args.concepts_file is not None #must provide extracted concepts if using this model
        assert args.recombine_option is not None #make sure have specified how to construct the embeddings
        if args.recombine_option == 'weight_matrix':
            assert args.concept_word_dict is not None
        if args.description_dir is None:
            print("YOU DIDN'T SPECIFY A PRETRAINED CODE EMBEDDINGS FILE!")
        #TODO: add more asserts here
        model = models.ConvAttnPoolPlusGram(Y, args.embed_file, args.code_embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts, args.recombine_option,
                                    embed_size=args.embed_size, hidden_sim_size=args.hidden_sim_size, dropout=args.dropout)

        #convert the codes if necessary
        if args.annotation_type != "ICD9":
            raise Exception("You must first process the data to include ICD9 codes!")

        #TODO: ADD CONVERSION FOR ICD9 HERE

    if args.test_model:

        sd = torch.load(args.test_model)

        if args.model == "conv_attn_plus_GRAM":
            assert list(sd.items())[1][1].size(0) == model.concept_embed.weight.size(0)

        model.load_state_dict(sd)


    if args.gpu:
        model.cuda()

    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    #TODO: update this**
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers, 
                  args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "lmbda", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

