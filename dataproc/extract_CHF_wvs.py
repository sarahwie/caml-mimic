"""
    Use the vocabulary to load a matrix of pre-trained word vectors
    **CHF VOCAB SPECIFIC**
"""
import csv
import os
import gensim.models
from tqdm import tqdm

from constants import *
import datasets

import numpy as np
import pickle

def gensim_to_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab, key=int))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)

def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index2word[0])) ))
    words = [PAD_CHAR]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    vocab = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
            vocab.append(line[0])
        #UNK embedding, gaussian randomly initialized 
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    vocab = vocab[1:] #we don't care about pad token
    W = np.array(W)
    w2ind = {k:i for i,k in enumerate(vocab,1)}
    return W, w2ind

def load_concept_embeddings(embed_file, embed_size, ind2concept, concept2ind):

    #TODO: HOW TO INITIALIZE A FEW CONCEPT EMBEDDINGS WE'RE MISSING WITH THIS METHOD:
     #[209, 209-209.99, 249, 338-338.99, 339.0, 339.1, 349.3, 358.3, 365.7, 447.7, 535.7, 610-612.99, 625.7, 649.7, 707.2, 99999, E001-E030.9]

    #unlike for word embeds, this embed_file is a dictionary of 'concept':np.array(embedding). We need to create the stacked matrix based on whether or not the concept is in our concept vocab. or not
    W = np.zeros((len(ind2concept)+2, embed_size))

    embed = pickle.load(open(embed_file,'rb'))

    i = 0
    for key, el in concept2ind.items():
        if key in embed:

            #get the embedding & normalize
            vec = embed[key]
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            assert el != 0 #should not be replacing pad
            W[el,:] = vec

        else:
            i += 1
            #TODO: need to initialize randomly here...for now leave as zeroes

    print("adding unk embedding") #gaussian randomly init.
    vec = np.random.randn(embed_size)
    vec = vec / float(np.linalg.norm(vec) + 1e-6)
    W[len(concept2ind)+1,:] = vec 

    print("missed concept embeddings:", i)

    print("concept embeds init:", W.shape)
    return W

