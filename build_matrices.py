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

global concepts_arr

def concept_iterator(notes, sub):
    #get distinct note_ids set
    for inx, el in tqdm(notes.iterrows(), total=notes.shape[0]):
         #get extr concepts subset
        yield el #note, concepts subset

# def listener_icd(icd_file, q):
#     while True:
#         writer = csv.writer(open(icd_file, "ab"))
#         line = q.get()
#         if isinstance(line, list):
#             writer.writerow(line)
#         elif line == 'kill':
#             return

# def listener_rxnorm(rxnorm_file, q):
#     while True:
#         writer = csv.writer(open(rxnorm_file, "ab"))
#         line = q.get()
#         if isinstance(line, list):
#             writer.writerow(line)
#         elif line == 'kill':
#             return

def listener_snomed(snomed_file, q):
    while True:
        writer = csv.writer(open(snomed_file, 'ab+'))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def listener_snomed_concept_arr(snomed_file, q):
    global concepts_arr
    concepts_arr = {}
    while True:
        line = q.get()
        if isinstance(line, list):
            concepts_arr[line[0]] = line[1]
        elif line == 'kill':
            pickle.dump(concepts_arr, open(snomed_file,'wb'))
            return

def listener_snomed_missed(snomed_file, q):
    while True:
        writer = csv.writer(open(snomed_file, 'ab+'))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def listener_snomed_ratios(snomed_file, q):
    while True:
        writer = csv.writer(open(snomed_file, 'ab+'))
        line = q.get()
        if isinstance(line, list):
            writer.writerow(line)
        elif line == 'kill':
            return

def work(inpt, snomed, snomed_concept_arr, snomed_missed, snomed_cnts):

    df_CONCEPTS, new_row = inpt
    new_id = str(new_row["SUBJECT_ID"]) + '_' + str(new_row["HADM_ID"])
    subset = df_CONCEPTS.loc[df_CONCEPTS.patient_id.map(str) == new_id]

    #BUILD MATRIX HERE

    #get text for constructing concept arr:
    new_text = new_row.TEXT
    patient_id = str(new_row.SUBJECT_ID)
    deid_note_id = str(new_row.HADM_ID)
    full_id = patient_id + '_' + deid_note_id
    words = new_text.split()    

    starting_inxs = [0] + [m.start() + 1 for m in re.finditer(' ', new_text)]
    ending_inxs = [m.end() - 1 for m in re.finditer(' ', new_text)] + [len(new_text)]

    #build matrices for different codetypes
    snomed_mat, missed_snomed, found_snomed, concept_arr = build(['SNOMEDCT_US'], subset, words, starting_inxs, ending_inxs, new_text)
    #rxnorm_mat, missed_rxnorm = build(['RXNORM'], subset, words, starting_inxs, ending_inxs, new_text)
    #icd_mat, missed_icd = build(['ICD9CM', 'ICD10CM'], subset, words, starting_inxs, ending_inxs, new_text)

    #put matrices on queue
    snomed.put([patient_id, deid_note_id, snomed_mat, len(words)])
    snomed_concept_arr.put([full_id, concept_arr])
    if missed_snomed:
        snomed_missed.put([missed_snomed])
    snomed_cnts.put([len(missed_snomed)/float(found_snomed)])
    # rxnorm.put([patient_id, deid_note_id, rxnorm_mat, len(words)])
    # icd.put([patient_id, deid_note_id, icd_mat, len(words)])

def build(labeltype, sub, words, starting_inxs, ending_inxs, new_text):

    concept_arr = [set() for i in range(len(words))]
    subset = sub.loc[sub.codingScheme.isin(labeltype)]

    #USE SUBSETS TO CONSTRUCT MATRICES**
    missed = set()
    for _, row in subset.iterrows():
        # write code to position in array
        if row['begin_inx'] not in starting_inxs or row['end_inx'] not in ending_inxs:
            missed.add(row['word_phrase'], new_text[row['begin_inx']:row['end_inx']])
        else:
            assert new_text[row['begin_inx']:row['end_inx']] == row['word_phrase']
            start = starting_inxs.index(row['begin_inx'])
            end = ending_inxs.index(row['end_inx'])
            for el in range(start, end+1):
                concept_arr[el].add(row['code'])

    #convert to string
    c = ' '.join([';'.join(el) if el else 'None' for el in concept_arr])
    found = len([el for el in concept_arr if el])
    return c, missed, found, concept_arr

def main(args):

    a = datetime.datetime.now().replace(microsecond=0)

    #specify num. cpus
    if args.num_cpus == 'all':
        cpus = multiprocessing.cpu_count()
    else:
        cpus = int(args.num_cpus)

    #load the parsed notes:
    print("loading in notes files...")
    print("READING NOTES FROM", args.input_dir.replace('train', args.split))
    notes = pd.read_csv(open(args.input_dir.replace('train', args.split), 'r'))
    print("Shape of notes file:", notes.shape)

    #get concepts here
    print("READING CONCEPTS FROM", args.concepts_dir.replace('train', args.split))
    df_CONCEPTS = pd.read_csv(args.concepts_dir)
    print("Shape of concepts file:", df_CONCEPTS.shape)

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to load notes & concepts:", str(b - a))

    a = datetime.datetime.now().replace(microsecond=0)
    #initialize the concept iterator
    el = concept_iterator(notes, df_CONCEPTS)

    #setup writing queues for each codetype--------------------------------------------------------
    manager = multiprocessing.Manager()
    snomed = manager.Queue()
    snomed_concept_arr = manager.Queue()
    snomed_missed = manager.Queue()
    snomed_cnts = manager.Queue()
    #icd = manager.Queue()
    #rxnorm = manager.Queue()

    #specify files to pass in (need this as an arg to process for some reason)----------------------
    snomed_file = os.path.join(args.output_dir, 'train_patient_concepts_matrix_SNOMED.csv').replace('train', args.split)\
    concepts_file = os.path.join(args.output_dir, 'train_patient_concepts_matrix_SNOMED.p').replace('train', args.split)
    missed_file = os.path.join(args.output_dir, 'missed_concepts_train.csv').replace('train', args.split)
    ratios_file = os.path.join(args.output_dir, 'train_ratios_missed.txt').replace('train', args.split)
    #rxnorm_file = args.output_dir.replace('train', args.split).replace('SNOMED', 'RXNORM')
    #icd_file = args.output_dir.replace('train', args.split).replace('SNOMED', 'ICD')
    assert not os.path.isfile(snomed_file)
    assert not os.path.isfile(concepts_file)
    assert not os.path.isfile(missed_file)
    assert not os.path.isfile(ratios_file) 

    #start write processes----------------------------------------------------------------
    #writer_process_icd = multiprocessing.Process(target=listener_icd, args=(icd_file, icd))
    #writer_process_icd.start()

    #writer_process_rxnorm = multiprocessing.Process(target=listener_rxnorm, args=(rxnorm_file, rxnorm))
    #writer_process_rxnorm.start()

    writer_process_snomed = multiprocessing.Process(target=listener_snomed, args=(snomed_file, snomed))
    writer_process_concept_arr = multiprocessing.Process(target=listener_snomed_concept_arr, args=(concepts_file, snomed_concept_arr))
    writer_process_missed = multiprocessing.Process(target=listener_snomed_missed, args=(missed_file, snomed_missed))
    writer_process_ratios = multiprocessing.Process(target=listener_snomed_ratios, args=(ratios_file, snomed_cnts))

    writer_process_snomed.start()
    writer_process_concept_arr.start()
    writer_process_missed.start()
    writer_process_ratios.start()


    #write headers to queue----------------------------------------------------------------
    header_df = ['SUBJECT_ID', 'HADM_ID', 'matrix', 'length']
    snomed.put(header_df)
    #icd.put(header_df)
    #rxnorm.put(header_df)
    icd = None
    rxnorm = None

    #SPAWN PROCESSES FROM EACH PATIENT DIR**------------------------------------------------
    threads = []

    elNext = next(el, None)
    while threads or elNext is not None:
        if (len(threads) < cpus) and elNext is not None:
            p = multiprocessing.Process(target=work, args=[elNext, snomed, snomed_concept_arr, snomed_missed, snomed_cnts])
            p.start()
            threads.append(p)
            elNext = next(el, None)
        else:
            for thread in threads:
                if not thread.is_alive():
                    threads.remove(thread)

    #finish write
    snomed.put('kill')
    snomed_concept_arr.put('kill')
    snomed_missed.put('kill')
    snomed_cnts.put('kill')
    #icd.put('kill')
    #rxnorm.put('kill')
    #writer_process_rxnorm.join()
    writer_process_snomed.join()
    writer_process_concept_arr.join()
    writer_process_missed.join()
    writer_process_ratios.join()
    #writer_process_icd.join()

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to process:", str(b - a))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('concepts_dir')
    parser.add_argument('output_dir')
    parser.add_argument('split')
    parser.add_argument('num_cpus')
    args = parser.parse_args()
    main(args)

#python build_matrices.py /data/swiegreffe6/NEW_MIMIC/mimic3/train_full.csv /data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_train_ALL.csv /data/swiegreffe6/NEW_MIMIC/extracted_concepts/recomputed/ train 22


