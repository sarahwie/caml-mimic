import argparse
import datetime
import multiprocessing
import os
import pickle
import pandas as pd

import testing_concept_extraction
from parse_xmi_real import parse_xmi

'''This script gets *all* extracted concepts for *all* annotated files. We can later subset this training matrix down based on what we want, 
when we call extension_tools.get_concept_text_alignment() specifically**

TODO: see about the size of the dataframe and whether can write out or not'''

def work(cmd, already_found, directory_counts, q):
    df_local = parse_xmi(cmd, already_found, directory_counts)
    if not df_local.empty:  # re-align header for good measure
        df_local = df_local[['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type', 'codingScheme', 'code', 'preferredText',
         'word_phrase']]
        print(cmd + 'ALL_PARSED')
    #else:
    #    print("**EMPTY DIRECTORY**:", cmd)
    q.put(df_local)
    return df_local

def listener(file, q):
    while True:
        line = q.get()
        if isinstance(line, pd.DataFrame):
            line.to_csv(file, mode='a', header=False, encoding='utf-8')
        elif line == 'kill':
            return

def main(args):

    #normalize paths
    input_dir = os.path.normpath(args.input_dir)
    output_dir = os.path.normpath(args.output_dir)

    #first write header to file
    a = datetime.datetime.now().replace(microsecond=0)

    if args.num_cpus == 'all':
        cpus = multiprocessing.cpu_count()
    else:
        cpus = int(args.num_cpus)

    patient_dirs = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]

    threads = []
    len_stas = len(patient_dirs)
    print("+++ Number of directories to process: %s" % (len_stas))

    if input_dir.split(os.sep)[-1] == '':
        identifier = input_dir.split(os.sep)[-2]
    else:
        identifier = input_dir.split(os.sep)[-1]
    file = os.path.join(output_dir, 'concepts_all_%s.csv' % identifier)

    already_found = testing_concept_extraction.get_already_processed_ids(args)
    print(len(already_found))
    print(next(iter(already_found.items())))
        #pickle.load(open('I://concepts_%s_ALREADY_FOUND.p' % identifier, 'rb'))
    directory_counts = testing_concept_extraction.get_directory_counts(args)
    print(len(directory_counts))
    print(next(iter(directory_counts.items())))
        #pickle.load(open('I://directory_counts_%s_ALREADY_FOUND.p' % identifier,'rb'))

    #setup manager with write access to file
    manager = multiprocessing.Manager()
    q = manager.Queue()
    #don't rewrite header
    # header_df = pd.DataFrame(columns=['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type', 'codingScheme', 'code', 'preferredText', 'word_phrase'])
    # header_df.loc[len(header_df)] = ['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type', 'codingScheme', 'code', 'preferredText', 'word_phrase']
    # q.put(header_df)

    #start write process
    writer_process = multiprocessing.Process(target=listener, args=(file, q))
    #writer_process.daemon = 1
    writer_process.start()

    # now spawn processes from each patient dir*
    while threads or patient_dirs:

        if (len(threads) < cpus) and patient_dirs:
            p = multiprocessing.Process(target=work, args=[patient_dirs.pop(), already_found, directory_counts, q])
            #p.daemon = 1
            p.start()
            threads.append(p)
        else:
            for thread in threads:
                if not thread.is_alive():
                    threads.remove(thread)

    #finish write
    q.put('kill')
    writer_process.join()

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to process:", str(b - a))

    #TODO: subset down to train, test and dev concepts before writing out.
    #TODO: also make sure to use this opportunity ^ to remove concepts which shouldn't have been included.
    #TODO: at the end of running this script on 2 diff. machines, will have to join the concepts back together into one file for each split**
    #TODO: remap note ids, only keep those records which we actually have in our final de-identified set**


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_cpus', type=str)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--no-stats', action='store_false', required=False, dest='no_stats',
                        help='whether or not to collect and print stats about number of concepts extracted')
    args = parser.parse_args()
    main(args)
