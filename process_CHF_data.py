import csv
import pandas as pd
import os
import multiprocessing
import datetime
from tqdm import tqdm
from dataproc import word_embeddings
from build_vocab import build_vocab

def work(filename):
        with open(filename, 'r') as f:
                new_file = filename.split(os.sep)[-1].replace('.', '_NEW.')
                with open(os.path.join('/data/swiegreffe6/sutter_nlp/', new_file), 'w') as out:
                        print(out)
                        reader = csv.reader(f)
                        writer = csv.writer(out)
                        for line in tqdm(reader):
                                writer.writerow([line[0], line[1], line[2].replace(';',' '), line[3], line[4]])

def reconvert_deid_text_strings():

        #test read
        filedir = '/project/sutter/Data/2018_0801_NLP/'
        f = [os.path.join(filedir,filename) for filename in os.listdir(filedir) if 'deidentified_notes' in filename]
        print("Files to process:", len(f))

        cpus = int(len(f))
        print("cpus", cpus)

        a = datetime.datetime.now().replace(microsecond=0)

        for i in range(cpus):
                #start 18 processes
                p = multiprocessing.Process(target=work, args=[f.pop()]) #we pop a directory and initialize a process on it
                p.start()

        b = datetime.datetime.now().replace(microsecond=0)
        print("Time to process", str(b-a))

def reorder(filename):

        print("Parsing %s" % filename)
        df = pd.read_csv(filename)
        df['length_two'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
        df = df.sort_values(['length_two'])
	df.to_csv(filename.replace('NEW','ORDERED'), index=False)

def reorder_by_length():

        #test read
        filedir = '/data/swiegreffe6/sutter_nlp/'
        f = [os.path.join(filedir,filename) for filename in os.listdir(filedir) if 'deidentified' in filename]
        print("Files to process:", len(f))
        print(f)

        cpus = int(len(f))
        print("cpus", cpus)

        a = datetime.datetime.now().replace(microsecond=0)

        for i in range(cpus):
                #start process for each file
                p = multiprocessing.Process(target=reorder, args=[f.pop()]) #we pop a directory and initialize a process on it
                p.start()

        b = datetime.datetime.now().replace(microsecond=0)
        print("Time to process", str(b-a))

def train_word_embeddings():
	a = datetime.datetime.now().replace(microsecond=0)

	#TODO: ONLY TRAIN on the training data that is the pre-subsetted vocab**
	w2v_file = word_embeddings.word_embeddings('full', '/data/swiegreffe6/sutter_nlp/deidentified_notes_train_subset_vocab_ORDERED.csv', 100, 0, 5, 4)	

        b = datetime.datetime.now().replace(microsecond=0)
        print("Time to process", str(b-a))

def rebuild_vocab():

        vocab_min = 3
        MIMIC_3_DIR = '/data/swiegreffe6/sutter_nlp'
        tr = os.path.join(MIMIC_3_DIR, 'deidentified_notes_train_subset_vocab_ORDERED.csv')
        vname = '%s/vocab.csv' % MIMIC_3_DIR
        build_vocab.build_vocab(vocab_min, tr, vname)

if __name__ == '__main__':
        #reconvert_deid_text_strings()
        #reorder_by_length()
	#train_word_embeddings()
        rebuild_vocab()