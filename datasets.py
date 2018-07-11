"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys
import os
import re
import pickle

from constants import *

class Batch:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self, desc_embed):
        self.docs = []
        self.docs_masks = []
        self.word_concept_mask = []
        self.concepts = []
        self.parents = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.batched_concepts_mask = []
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []

    def add_instance(self, inpt, ind2c, c2ind, w2ind, dv_dict, concept2ind, ind2concept, child2parents, concept_word, num_labels, GRAM):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """

        row, concept_dict = inpt

        labels = set()
        hadm_id = int(row[1])
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
        words = text.strip().split()
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in words]

        if GRAM: #get concepts (and parents) arrays from file, dictionary respectively, convert to index

            joint_id = row[0] + '_' + row[1]
            #TODO: MULTI-LABEL FOR ONE-POSITION CASE**
            #just a standard dictionary- this ensures that every patient has been mapped
            con = [int(concept2ind[w]) if w in concept2ind else len(concept2ind)+1 if w != 0 else 0 for w in concept_dict[joint_id]] #TODO: CHECK PADDING HERE*
            assert len(con) == len(concept_dict[joint_id])
            assert len(con) == len(text)

            #truncate concepts
            if len(con) > self.max_length:
                con = con[:self.max_length]

            #TODO: GET PARENTS (len 6*)**
            #TODO: WHY ARE THERE CODES WHICH WE HAVE TO EMBED AS UNK IN TRAINING????** see in ConvAttnPoolPlusGram init() in models.py where have extended len(concept_embeddings matrix) by 1!
            parents = [child2parents[w] if w in child2parents else [w, rootCode] for w in concept_dict[joint_id] if w != 0] #this is the list of parent codes for each concept
            #NOTE: the above: we want to do this regardless of whether or not we have an index of that child in our training set-constructed vocab, so that we can still learn the embeds @ test time over the parents
            #convert to indices
            parent_inx = [[int(concept2ind[it]) if it in concept2ind else len(concept2ind)+1 for it in el] for el in parents] 
            #now, we embed the parents based on whether or not they existed in our model**
            #TODO: check the 3 above dicts: con, parents, and parent_inx**

            #pad with 0's so each has len. 6 (inc. child itself)
            #TODO: WHAT IS THE ROLE OF THE ROOT CODE HERE-- A 'SHARED' EMBEDDING OF SOME SORT?
            pars = [xi+[0]*(6-len(xi)) for xi in parent_inx]

            #****TODO: ASSERT ORDER HELD-- LOWEST LEVEL, UP****
            concept_len = [1]*len([el for el in con if el != 0])
                #if no concepts, an empty list

            #append to arraylist we're keeping
            self.concepts.append(con)
            self.parents.append(pars)
            self.batched_concepts_mask.append(concept_len)

        #truncate long documents
        if len(text) > self.max_length: #TODO: UNDO FOR SH CASE**
            text = text[:self.max_length]

        if GRAM:
            word_mask = [el for el,c in zip(text,con) if c != 0]
            self.docs_masks.append(word_mask)

            children_concepts = [child for child in con if child != 0] #this is the list of children concept codes

            weights_matrix_mask = [concept_word[el] if el in concept_word else len(concept_word)+1 for el in zip(word_mask, children_concepts)]

            self.word_concept_mask.append(weights_matrix_mask)

        #build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        #reset length
        self.length = min(self.max_length, length)


    def pad_docs(self, GRAM):
        #pad all docs (and concepts) to have self.length

        if GRAM:
            padded_docs = []
            padded_concepts = []
            for doc, con in zip(self.docs, self.concepts):
                if len(doc) < self.length:
                    doc.extend([0] * (self.length - len(doc))) #TODO: fix padding**
                    con.extend([0] * (self.length - len(con))) #TODO: fix padding**
                    assert len(doc) == len(con) 
                padded_docs.append(doc)
                assert len(con) == len(doc)
                padded_concepts.append(con)
            self.docs = padded_docs
            self.concepts = padded_concepts

            #also pad parent concepts:
            max_concepts_in_batch = max(len(l) for l in self.parents)
            #TODO: HAVE PADDED HERE AS WELL FOR INC. OF BOTH CHILD AND ROOT CODE**
            [xi.extend([[0,0,0,0,0,0]] * (max_concepts_in_batch-len(xi))) for xi in self.parents]

            #extend batched_concept_mask as well
            [xi.extend([0] * (max_concepts_in_batch-len(xi))) for xi in self.batched_concepts_mask]

            #pad word concepts mask as well
            [xi.extend([0] * (max_concepts_in_batch-len(xi))) for xi in self.docs_masks]

            #and word-concept indices mask as well
            [xi.extend([0] * (max_concepts_in_batch-len(xi))) for xi in self.word_concept_mask]

        else: 
            padded_docs = []
            for doc in self.docs:
                if len(doc) < self.length:
                    doc.extend([0] * (self.length - len(doc))) #TODO: fix padding**
                padded_docs.append(doc)
            self.docs = padded_docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.concepts), np.array(self.parents), np.array(self.labels), np.array(self.batched_concepts_mask), np.array(self.docs_masks), np.array(self.word_concept_mask), np.array(self.hadm_ids), self.code_set, np.array(self.descs)
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

def data_generator(filename, concepts_file, dicts, batch_size, num_labels, GRAM, desc_embed=False, version='mimic3'):
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
    ind2w, w2ind, ind2c, c2ind, dv_dict, concept2ind, ind2concept, child2parents, concept_word = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv'], dicts['concept2ind'], dicts['ind2concept'], dicts['child2parents'], dicts['concept_word']

    if GRAM:
        #load concepts matrix
        concept_dict = pickle.load(open(concepts_file, 'rb'))
    else:
        concept_dict = None

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_inst = Batch(desc_embed)
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs(GRAM)
                yield cur_inst.to_ret()
                #clear
                cur_inst = Batch(desc_embed)
            cur_inst.add_instance((row, concept_dict), ind2c, c2ind, w2ind, dv_dict, concept2ind, ind2concept, child2parents, concept_word, num_labels, GRAM)
        cur_inst.pad_docs(GRAM)
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
    #TODO: WHAT IS THIS...
    if args.public_model and args.Y == 'full' and args.version == "mimic3" and args.model == 'conv_attn':
        ind2w = {i:w for i,w in enumerate(sorted(vocab))}
    else:
        #START FROM ONES
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
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup, concept lookups
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
        desc_dict = load_code_descriptions(args.model)
    c2ind = {c:i for i,c in ind2c.items()}

    if args.model == 'conv_attn_plus_GRAM':
        ind2concept = load_concepts(args)
        concept2ind = {c:i for i,c in ind2concept.items()} 
        child2parents = pickle.load(open(args.parents_file, 'rb'))
        concept_word = pickle.load(open(args.concept_word_dict, 'rb'))

        #remap based on concept2ind and w2ind:
        new_dict = set()
        for key, value in concept_word.items():
            if key[0] in w2ind and key[1] in concept2ind:
                new_dict.add((int(w2ind[key[0]]), int(concept2ind[key[1]])))
            elif key[0] in w2ind:
                new_dict.add((int(w2ind[key[0]]), len(concept2ind)+1))
            elif key[1] in concept2ind:
                new_dict.add((len(w2ind)+1, int(concept2ind[key[1]])))
            else:
                #both unk
                new_dict.add((len(w2ind)+1, len(concept2ind)+1))

        #enumerate again
        concept_word = {c:i for i,c in enumerate(sorted(new_dict), 1)}

    else:
        ind2concept = None
        concept2ind = None
        child2parents = None
        concept_word = None

    #get description one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors(args.description_dir, version=args.version)
    else:
        dv_dict = None

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict, 'ind2concept': ind2concept, 'concept2ind':concept2ind, 'child2parents':child2parents, 'concept_word':concept_word}
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
            try:
                #TODO: WHY DOES JAMES LOAD FULL CODES (FROM ALL 3 DATASETS) HERE?**
                with open(train_path.replace('train', split), 'r') as f:
                    lr = csv.reader(f)
                    next(lr)
                    for row in lr:
                        for code in row[3].split(';'):
                            codes.add(code)
            except StopIteration:
                print("FILE IS EMPTY")
                pass
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c, desc_dict

#TODO: UPDATE THIS METHOD BASED ON PROPOSED STRUCTURE OF CONCEPTS_FILE**
def load_concepts(args):

    codes = set()
    with open(args.concept_vocab) as f:
        content = f.readlines()
    codes.update([x.strip() for x in content]) #add in the new values to the set
    #START FROM ONE
    ind2concept = {i:c for i,c in enumerate(sorted(codes), 1)}
    #print(len(ind2c))
    #print(ind2c)
    return ind2concept

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

def diag_process(code):
    if code.startswith('E'):
        if len(code) > 4: code = code[:4] + '.' + code[4:]
    else:
        if len(code) > 3: code = code[:3] + '.' + code[3:]
    return code

def procedure_process(code):
    code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(model, version='mimic3'):
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

    if model == "conv_attn_plus_GRAM":
        #also load the higher-level CCS grouped codes**

        with open(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_dx_tool_2015.csv'), 'r') as infd:
            reader = csv.reader(infd)
            next(reader)
            for tokens in reader:

                cat1 = tokens[1][1:-1].strip()
                cat2 = tokens[3][1:-1].strip()
                cat3 = tokens[5][1:-1].strip()
                cat4 = tokens[7][1:-1].strip()

                desc1 = tokens[2].strip()
                desc2 = tokens[4].strip()
                desc3 = tokens[6].strip()
                desc4 = tokens[8].strip()

                if cat1 != '' and cat1 not in desc_dict.keys():
                    desc_dict[cat1] = desc1
                if cat2 != '' and cat2 not in desc_dict.keys():
                    desc_dict[cat2] = desc2
                if cat3 != '' and cat3 not in desc_dict.keys():
                    desc_dict[cat3] = desc3
                if cat4 != '' and cat4 not in desc_dict.keys():
                    desc_dict[cat4] = desc4

        #do the same things for the procedure codes: Note only have max. 3 levels vs. diagnoses' 4
        with open(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_pr_tool_2015.csv'), 'r') as infd:
            reader = csv.reader(infd)
            next(reader)
            for tokens in reader:
                
                cat1 = tokens[1][1:-1].strip()
                cat2 = tokens[3][1:-1].strip()
                cat3 = tokens[5][1:-1].strip()

                desc1 = tokens[2].strip()
                desc2 = tokens[4].strip()
                desc3 = tokens[6].strip()

                if cat1 != '' and cat1 not in desc_dict.keys():
                    desc_dict[cat1] = desc1
                if cat2 != '' and cat2 not in desc_dict.keys():
                    desc_dict[cat2] = desc2
                if cat3 != '' and cat3 not in desc_dict.keys():
                    desc_dict[cat3] = desc3

    return desc_dict

def load_description_vectors(description_dir, version='mimic3'):
    #load description one-hot vectors from file
    dv_dict = {}
    if version == 'mimic2':
        data_dir = MIMIC_2_DIR
    else:
        data_dir = MIMIC_3_DIR
    with open(description_dir, 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict
