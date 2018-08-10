# coding=utf-8
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from memory_profiler import profile
import sys
#import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import datetime
import pickle
import numpy as np
import argparse
from scipy import sparse

# inits for proc. tools
wordnet_lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(decode_error='ignore')

class data_generator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile)
            next(reader) #skip header
            for row in tqdm(reader, total=2246586):
                yield process_sentence(row[2])

def process_sentence(inpt):
    return " ".join([wordnet_lemmatizer.lemmatize(word.decode('utf-8', errors='ignore')) for word in inpt.split(" ")])

@profile
def build_vocab(inpt):

    data = data_generator(inpt)

    a = datetime.datetime.now().replace(microsecond=0)
    vectorizer.fit(data)
    b = datetime.datetime.now().replace(microsecond=0)

    print("Time to fit vocab:", str(b-a))
    print("(LEMMATIZED)VOCAB SIZE:", len(vectorizer.vocabulary_))

    outfile = '/path/vocab.p'

    #save vocab. to file
    a = datetime.datetime.now().replace(microsecond=0)
    pickle.dump(vectorizer.vocabulary_, open(outfile, 'wb'))
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to pickle dump:", str(b-a))

@profile
def reduce_vocab_size(vocab_file, inpt):
    vectorizer.vocabulary_ = pickle.load(open(vocab_file, 'rb'))
    print("Old Vocab Size:", len(vectorizer.vocabulary_))

    sentences = data_generator(inpt)

    a = datetime.datetime.now().replace(microsecond=0)
    sparse_matrix = vectorizer.transform(sentences)
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to fit counts to training data:", str(b - a))

    #save
    a = datetime.datetime.now().replace(microsecond=0)
    sparse.save_npz('/path/transformed_docs.npz', sparse_matrix)
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to save transformed docs as sparse_csr:", str(b - a))
    del sentences

    a = datetime.datetime.now().replace(microsecond=0)
    #this line converts those word indices to 1 merely if a word occurred in a given document!
    #summing across the x-axis converts this to a count of the num. documents in which a word appears.
    vals = sparse_matrix.astype(bool).astype(int).sum(axis=0)
    print("TODO: check that 1L by xL here:", vals.shape)
    vals = set(np.where(vals >= 3)[1]) #these are the vocab indices which are applicable
    del sparse_matrix
    new_vocab = set([key for key,value in vectorizer.vocabulary_.items() if value in vals]) #set of words which apply
    #remap to a proper index
    new_vocab = {key:i for i,key in enumerate(new_vocab, 1)}
    del vals
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to subset vocab:", str(b - a))

    print("New Vocab Size:", len(new_vocab))

    a = datetime.datetime.now().replace(microsecond=0)
    pickle.dump(new_vocab, open(
        '/path/NEW_vocab.p', 'wb'))
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to save new vocab as pickle:", str(b-a))

def transform_inputs(vocab_file, inpt, outpt):

    new_vocab = pickle.load(open(vocab_file, 'rb'))
    print("VOCAB SIZE (should be condensed):", len(new_vocab))

    with open(inpt) as csvfile, open(outpt, 'wb') as out:
        reader = csv.reader(csvfile)
        writer = csv.writer(out)
        writer.writerow(next(reader))  # skip header
        for row in tqdm(reader, total=2246586):
            #reconvert the text
            txt = process_sentence(row[2])
            new_txt = ";".join([str(new_vocab[word]) if word in new_vocab else str(len(new_vocab)+1) for word in txt.split()])

            if len(new_txt) != 0:
                writer.writerow([row[0],row[1],new_txt,row[3],row[4]])

    #TODO: run for test and dev case as well**

def main(args):

    if args.method == 'build_vocab':
        build_vocab(args.input_file)

    elif args.method == 'reduce_vocab_size':
        vocab_file = '/path/vocab.p'
        reduce_vocab_size(vocab_file, args.input_file)

    elif args.method == 'transform_inputs':
        assert args.vocab is not None
        assert args.outfile is not None
        transform_inputs(args.vocab, args.input_file, args.outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str) #, options=['build_vocab','reduce_vocab_size','transform_inputs'])
    parser.add_argument('input_file', type=str)
    parser.add_argument('--vocab', type=str, required=False, dest='vocab')
    parser.add_argument('--output', type=str, required=False, dest='outfile')
    args = parser.parse_args()
    main(args)

#input_file = '/path/notes_train.csv'
#new_vocab_file = '/path/NEW_vocab.p'
