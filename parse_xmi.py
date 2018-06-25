from xml.dom import minidom as md
from xml.dom.minidom import Node
import os
#from constants import *
from collections import namedtuple, defaultdict
import pandas as pd
import numpy as np
import argparse
import datetime
from tqdm import tqdm
import extension_tools
import csv
import pickle

def main(args):

    parse_xmi(args)

    #build_concept_vocab()

def parse_xmi(args):

    print("Parsing files from %s..." % args.input_dir)
    a = datetime.datetime.now().replace(microsecond=0)

    all_files = os.listdir(args.input_dir)
    xmi_files = [e for e in all_files if '.xm' in e]
    print("Num files to process:", len(xmi_files))

    path = os.path.normpath(args.input_dir)
    num = path.split(os.sep)[-1] #this should be the dir number
    split = path.split(os.sep)[-2].replace('_out', '')
    identifier = split + '_' + num


    df_meta = pd.DataFrame()
    codes = {}
    patient_concepts_matrix = {}

    if args.no_stats:
        # #first iterate through to get sets of concepts
        # print("GETTING CONCEPTS FOR STATS: ")
        # for file in tqdm(xmi_files, total=len(xmi_files)):
        #     xmldoc = md.parse(os.path.join(args.input_dir,file))

        #     conc_names = set()
        #     for element in xmldoc.getElementsByTagName("xmi:XMI"):
        #         for el in element.childNodes:
        #             if el.nodeType == Node.ELEMENT_NODE:
        #                 if 'Mention' in el.tagName:
        #                     conc_names.add(el.tagName.split(":")[1])

        # #then initialize DF/store stats/extracted concepts in pandas df:
        # column_ordering = [el for el in iter(conc_names)]

        column_ordering = ['SignSymptomMention', 'ProcedureMention', 'MedicationMention','AnatomicalSiteMention','EntityMention' 'DiseaseDisorderMention']
        df = pd.DataFrame(columns = ['patient_id'] + column_ordering)

    num_codes = []
    print("EXTRACTING CONCEPTS:")
    for file in tqdm(xmi_files, total=len(xmi_files)):

        #get patient-note id
        pat_note_id = file.split(".")[0]

        xmldoc = md.parse(os.path.join(args.input_dir,file))

        #initialize local data structures
        concepts = defaultdict(list)
        lookups = pd.DataFrame(columns=['lookup_id', 'codingScheme', 'code', 'preferredText'])
        df_local = pd.DataFrame(columns=['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type'])

        #only pass through the text once*
        cnt = 1
        for element in xmldoc.getElementsByTagName("xmi:XMI"):
            for el in element.childNodes:
                if el.nodeType == Node.ELEMENT_NODE:
                    if 'Mention' in el.tagName:
                        concepts[el.tagName.split(":")[1]].append(el.attributes['ontologyConceptArr'].value)
                        #store all the tags for now
                        df_local.loc[len(df_local)] = [file.split(".")[0],el.attributes['ontologyConceptArr'].value.split(' ')[-1],int(el.attributes['begin'].value),int(el.attributes['end'].value),el.tagName.split(":")[1]]
                    #TODO: HERE, ONLY EXTRACTING ICD9 CODES**
                    elif 'refsem:UmlsConcept' in el.tagName and 'ICD9' in el.attributes['codingScheme'].value:
                        try: 
                            cnt += 1
                            lookups.loc[len(lookups)] = [el.attributes['xmi:id'].value,el.attributes['codingScheme'].value, el.attributes['code'].value, el.attributes['preferredText'].value]
                        except KeyError:
                            pass
                    elif 'cas:Sofa' in el.tagName:
                        text = el.attributes['sofaString'].value

        #TODO: there seem to be extra codes in the file's UmlsConcept mappings that don't actually show up in the annotation

        #write stats to global DFs:
        if args.no_stats:
            df.loc[len(df)] = [pat_note_id] + [len(concepts[term]) for term in column_ordering]

        #merge the lookups info with the
        df_local = df_local.merge(lookups, how='inner', left_on='lookup_id', right_on='lookup_id')

        if not df_local.empty:
            df_local['code'] = df_local['code'].astype(str)

        #TODO: UPDATE HERE**
        for i, row in df_local.iterrows():
            if row['code'] in codes.keys():
                codes[row['code']] += 1
            else:
                codes[row['code']] = 1

        num_codes.append(len(df_local))

        #add in text info
        if not df_local.empty:
            fn = lambda x: text[int(x['begin_inx']):int(x['end_inx'])]
            df_local['word_phrase'] = df_local.apply(fn, axis=1)

        #go ahead and create concept matrix:
        concept_arr = extension_tools.get_concept_matrix(df_local, text)

        df_meta = df_meta.append(df_local)

        #write list to a larger dictionary for pickling
        patient_concepts_matrix[pat_note_id] = concept_arr

        #once have parsed once, pass through again to match actual text term & its location to SNOMED value
        # TODO: write out individual CUIs and their text terms to file*

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to process files from %s:" % args.input_dir)
    print(str(b - a))

    print("Writing out...")
    a = datetime.datetime.now().replace(microsecond=0)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.no_stats:
        df.to_csv(os.path.join(args.output_dir, 'concept_extraction_stats_%s.csv' % identifier))

    df_meta.to_csv(os.path.join(args.output_dir, 'concepts_%s_ICD9.csv' % identifier))
    #TODO: SUBSET
    with open(os.path.join(args.output_dir, 'concepts_vocab_%s_ICD9.csv' % identifier), 'w') as fp:
        writer = csv.writer(fp)
        for item in codes.keys():
            writer.writerow([item, codes[item]])

    #dump dictionary of concepts matrices
    print("Number of keys in dictionary:", len(patient_concepts_matrix.keys()))
    pickle.dump(patient_concepts_matrix, open(os.path.join(args.output_dir, 'patient_concepts_matrix_%s_ICD9.p' % identifier), 'wb'))

    print("Stats on number of ICD9 codes per file:", pd.Series(num_codes).describe())

    b = datetime.datetime.now().replace(microsecond=0)
    print("Time to write out:", str(b - a))

    if args.no_stats:
        print("Stats on extracted concepts:")
        print(df['patient_id'].describe())
        print(df['LabMention'].describe())
        print(df['SignSymptomMention'].describe())
        print(df['DiseaseDisorderMention'].describe())
        print(df['AnatomicalSiteMention'].describe())
        print(df['MedicationMention'].describe())
        print(df['ProcedureMention'].describe())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run the XMI concept extractor on cTAKES output and generate concepts input for MIMIC datafiles')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--no-stats', action='store_false', required=False, dest='no_stats', help='whether or not to collect and print stats about number of concepts extracted')
    args = parser.parse_args()
    main(args)