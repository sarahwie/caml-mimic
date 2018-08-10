'''This file contains tools by Sarah for coding the model extension'''
import csv
from tqdm import tqdm
import pickle
import datetime
import pandas as pd
import re
import argparse
import multiprocessing
import os

def concept_iterator(args, notes):
    #get distinct note_ids set
    notes_id_lst = (notes["SUBJECT_ID"].map(str) + '_' + notes["note_id"].map(str)).unique()
    with tqdm(total=notes.shape[0]) as pbar:
        with open("/PATH/concepts_all_%s.csv" % args.input_dir, "r") as fi:
            reader = csv.reader(fi)
            header = next(reader) #store/add header as first line of the file
            arr = pd.DataFrame(columns=header)
            i = 0
            for line in reader:
                if i == 0: #first row
                    curr_id = line[1] #store id
                    arr.loc[len(arr)] = line #append to dataframe
                else:
                    if line[1] != curr_id:
                        if curr_id in notes_id_lst:
                            #new instance- get text and yield older
                            new_row = notes.loc[notes.note_id == int(curr_id.split('_')[1])]
                            yield arr, new_row
                            pbar.update(1)
                        #else, don't yield, just replace
                        #reset df and curr_id
                        arr = pd.DataFrame(columns=header)
                        curr_id = line[1]
                    else: #else- still same patient_id, append to df
                        arr.loc[len(arr)] = line
                i += 1
            #yield last arr (if we want it)
            if curr_id in notes_id_lst:
                new_row = notes.loc[notes.note_id == int(curr_id.split('_')[1])]
                yield arr, new_row
                pbar.update(1)

def listener_icd(icd_file, q):
    while True:
        writer = csv.writer(open(icd_file, "ab"))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def listener_rxnorm(rxnorm_file, q):
    while True:
        writer = csv.writer(open(rxnorm_file, "ab"))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def listener_snomed(snomed_file, q):
    while True:
        writer = csv.writer(open(snomed_file, 'ab'))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def work(inpt, snomed, icd, rxnorm):

    subset, new_row = inpt

    #BUILD MATRIX HERE

    #get text for constructing concept arr:
    new_text = new_row.TEXT.iloc[0]
    patient_id = new_row.SUBJECT_ID.iloc[0]
    deid_note_id = new_row.note_id_deid.iloc[0]
    words = new_text.split()

    starting_inxs = [0] + [m.start() + 1 for m in re.finditer(' ', new_text)]
    ending_inxs = [m.end() - 1 for m in re.finditer(' ', new_text)] + [len(new_text)]

    #build matrices for different codetypes
    snomed_mat, missed_snomed = build(['SNOMEDCT_US'], subset, words, starting_inxs, ending_inxs, new_text)
    rxnorm_mat, missed_rxnorm = build(['RXNORM'], subset, words, starting_inxs, ending_inxs, new_text)
    icd_mat, missed_icd = build(['ICD9CM', 'ICD10CM'], subset, words, starting_inxs, ending_inxs, new_text)

    #put matrices on queue
    snomed.put([patient_id, deid_note_id, snomed_mat, len(words)])
    rxnorm.put([patient_id, deid_note_id, rxnorm_mat, len(words)])
    icd.put([patient_id, deid_note_id, icd_mat, len(words)])

def build(labeltype, sub, words, starting_inxs, ending_inxs, new_text):

    concept_arr = [set() for i in range(len(words))]
    subset = sub.loc[sub.codingScheme.isin(labeltype)]

    #USE SUBSETS TO CONSTRUCT MATRICES**
    missed = 0
    for _, row in subset.iterrows():
            for m in re.finditer(row['word_phrase'].lower(), new_text):
                #assert new_text[m.start():m.end()] == old_text[row['begin_inx']:row['end_inx']].lower()
                assert new_text[m.start():m.end()] == row['word_phrase'].lower()
                # write code to position in array
                if m.start() not in starting_inxs or m.end() not in ending_inxs:
                    # print("MISMATCH")
                    # print(m.start(), m.end(), new_text[m.start():m.end()])
                    missed += 1
                else:
                    start = starting_inxs.index(m.start())
                    end = ending_inxs.index(m.end())
                    for el in range(start, end+1):
                        concept_arr[el].add(row['code'])

    #convert to string
    c = ' '.join([';'.join(el) if el else 'None' for el in concept_arr])
    return c, missed

def main(args):

    a = datetime.datetime.now().replace(microsecond=0)

    #specify num. cpus
    if args.num_cpus == 'all':
        cpus = multiprocessing.cpu_count()
    else:
        cpus = int(args.num_cpus)

    #load note_id de-identifier:
    notes_deid = pd.read_csv('/path/pn_note_de_id_v1.txt', sep='\t')

    #load the parsed notes:
    print("loading in notes files...")
    notes_test = pd.read_csv('/path/notes_test.csv')
    notes_dev = pd.read_csv('/path/notes_dev.csv')
    notes_train = pd.read_csv('/path/notes_train.csv')
    notes = pd.concat((notes_test, notes_dev, notes_train))
    notes = pd.merge(notes, notes_deid, left_on='HADM_ID', right_on='note_id_deid')
    del notes_test
    del notes_dev
    del notes_train
    del notes_deid #free memory

    print("INPUT RECORDS SHAPE:", notes.shape) 
    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to load notes:", str(b - a))

    a = datetime.datetime.now().replace(microsecond=0)
    #initialize the concept iterator
    el = concept_iterator(args, notes)

    #setup writing queues for each codetype--------------------------------------------------------
    manager = multiprocessing.Manager()
    snomed = manager.Queue()
    icd = manager.Queue()
    rxnorm = manager.Queue()

    #specify files to pass in (need this as an arg to process for some reason)----------------------
    snomed_file = "/path/concept_alignment_matrices/snomed_matrices_%s_MULTIPROC.csv" % args.input_dir
    rxnorm_file = "/path/concept_alignment_matrices/rxnorm_matrices_%s_MULTIPROC.csv" % args.input_dir
    icd_file = "/path/concept_alignment_matrices/icd_matrices_%s_MULTIPROC.csv" % args.input_dir

    #start write processes----------------------------------------------------------------
    writer_process_icd = multiprocessing.Process(target=listener_icd, args=(icd_file, icd))
    writer_process_icd.start()

    writer_process_rxnorm = multiprocessing.Process(target=listener_rxnorm, args=(rxnorm_file, rxnorm))
    writer_process_rxnorm.start()

    writer_process_snomed = multiprocessing.Process(target=listener_snomed, args=(snomed_file, snomed))
    writer_process_snomed.start()

    #write headers to queue----------------------------------------------------------------
    header_df = ['SUBJECT_ID', 'HADM_ID', 'matrix', 'length']
    snomed.put(header_df)
    icd.put(header_df)
    rxnorm.put(header_df)

    #SPAWN PROCESSES FROM EACH PATIENT DIR**------------------------------------------------
    threads = []

    elNext = next(el, None)
    while threads or elNext is not None:
        if (len(threads) < cpus) and elNext is not None:
            p = multiprocessing.Process(target=work, args=[elNext, snomed, icd, rxnorm])
            p.start()
            threads.append(p)
            elNext = next(el, None)
        else:
            for thread in threads:
                if not thread.is_alive():
                    threads.remove(thread)

    #finish write
    snomed.put('kill')
    icd.put('kill')
    rxnorm.put('kill')
    writer_process_rxnorm.join()
    writer_process_snomed.join()
    writer_process_icd.join()

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to process:", str(b - a))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', choices=['dir_001','dir_002', 'TEST'])
    parser.add_argument('num_cpus', choices=['1','2','3','4','5','6','7','8','9','10','11','12','all'])
    args = parser.parse_args()
    main(args)
