"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys
import os

from constants import *

class Batch:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self, desc_embed):
        self.docs = []
        self.concepts = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []

    def add_instance(self, inpt, ind2c, c2ind, w2ind, dv_dict, concept2ind, ind2concept, num_labels, concepts=False):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """

        #TODO: UPDATE THIS METHOD TO ALSO INCLUDE CONCEPTS**
        #TODO: ALSO PAD THE CONCEPT MATRIX

        if concepts:
            row, concept = inpt
        else:
            row = inpt

        labels = set()
        hadm_id = int(row[1])
        #assert row[1] == concept[0] #assert same id #TODO
        extr_concepts = concept.split(';') #concept[1].split('')
        text = row[2]
        length = int(row[4])
        cur_code_set = set()
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []
        #get codes as a multi-hot vector
        for l in row[3].split(';'):
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return
        if self.desc_embed:
            for code in cur_code_set:
                l = ind2c[code]
                if l in dv_dict.keys():
                    #need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[l][:])
                else:
                    desc_vecs.append([len(w2ind)+1])
        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]

        if concepts:
            #TODO: MAP HERE TO WORDS

            con = [int(concept2ind[c] if c in concept2ind else len(concept2ind)+1 for c in BLANK)]

        #build instance
        self.docs.append(text)
        if concepts:
            self.concepts.append(con)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        #reset length
        self.length = min(self.max_length, length)

    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    #TODO: write pad_concepts method

    def to_ret(self):
        return np.array(self.docs), np.array(self.concepts), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
               np.array(self.descs)
    #TODO: EMPTY CONCEPTS LST PASSED IF NOT A THING FOR THAT PARTICULAR MODEL

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def data_generator_GRAM(filename, concepts_file, dicts, batch_size, num_labels, desc_embed=False, version='mimic3'):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
            version: which (MIMIC) dataset
        Yields:
            np arrays with data for training loop.
    """
    #TODO: HERE, YIELD A concepts matrix as well as its lookup
    ind2w, w2ind, ind2c, c2ind, dv_dict, concept2ind, ind2concept = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv'], dicts['concept2ind'], dicts['ind2concept']
    with open(filename, 'r') as infile, open(concepts_file, 'r') as concepts_file:
            c = csv.reader(concept_file)
            r = csv.reader(infile)
            #header
            next(r)
            next(c)
            cur_inst = Batch(desc_embed)
            for concept, row in zip(c,r):
                #find the next `batch_size` instances
                if len(cur_inst.docs) == batch_size:
                    cur_inst.pad_docs()
                    yield cur_inst.to_ret()
                    #clear
                    cur_inst = Batch(desc_embed)
                cur_inst.add_instance((row,concept), ind2c, c2ind, w2ind, dv_dict, concept2ind, ind2concept, num_labels, concepts=True)
            cur_inst.pad_docs()
            yield cur_inst.to_ret()

#TODO: THE OLD VERSION
def data_generator(filename, dicts, batch_size, num_labels, desc_embed=False, version='mimic3'):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
            version: which (MIMIC) dataset
        Yields:
            np arrays with data for training loop.
    """

    #TODO: HERE, yield a concepts matrix as well with the same type of lookups**
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_inst = Batch(desc_embed)
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                #clear
                cur_inst = Batch(desc_embed)
            cur_inst.add_instance(row, ind2c, c2ind, w2ind, dv_dict, None, None, num_labels, concepts=False)
        cur_inst.pad_docs()
        yield cur_inst.to_ret()

def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    #hack because the vocabs were created differently for these models
    if args.public_model and args.Y == 'full' and args.version == "mimic3" and args.model == 'conv_attn':
        ind2w = {i:w for i,w in enumerate(sorted(vocab))}
    else:
        ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

#TODO: load concepts codes as well
def load_lookups(args, desc_embed=False):
    """
        Inputs:
            args: Input arguments
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(args, args.vocab)

    #get code and description lookups
    if args.Y == 'full':
        ind2c, desc_dict = load_full_codes(args.data_path, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        desc_dict = load_code_descriptions()
    c2ind = {c:i for i,c in ind2c.items()}

    ind2concept = load_concepts()
    concept2ind = {c:i for i,c in ind2concept.items()}

    #get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(args.Y, version=args.version)
    else:
        dv_dict = None

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict, 'ind2concept': ind2concept, 'concept2ind':concept2ind}
    return dicts

def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    #get description lookup
    #desc_dict = load_code_descriptions(version=version)
    desc_dict = None
    #build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c, desc_dict

#TODO: UPDATE THIS METHOD BASED ON PROPOSED STRUCTURE OF CONCEPTS_FILE**
def load_concepts():

    codes = set()
    for split in ['train', 'dev', 'test']:
        with open(os.path.join(concept_write_dir,'%s_concepts.csv' % split)) as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                codes.add(row)
    ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(version='mimic3'):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for i,row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def load_description_vectors(Y, version='mimic3'):
    #load description one-hot vectors from file
    dv_dict = {}
    if version == 'mimic2':
        data_dir = MIMIC_2_DIR
    else:
        data_dir = MIMIC_3_DIR
    with open("%s/description_vectors.vocab" % (data_dir), 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict
