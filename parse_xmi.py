from xml.dom import minidom as md
from xml.dom.minidom import Node
import os
from constants import *
from collections import namedtuple, defaultdict
import pandas as pd
import numpy as np
import argparse
import datetime
from tqdm import tqdm

def main(args):

    parse_xmi(args)

    #build_concept_vocab()

def parse_xmi(args):

    for split in ['train', 'test', 'dev']:
        print("Parsing %s files..." % split)
        a = datetime.datetime.now().replace(microsecond=0)

        if args.test:
            all_files = os.listdir(test_dir)

        else:
            all_files = os.listdir(xmi_dir.replace('train', split))

        xmi_files = filter(lambda x: '.xm' in x, all_files)

        if args.test:
            print(xmi_files)

        #store stats/extracted concepts in pandas df:
        if args.no_stats:
            df = pd.DataFrame(columns = ['patient_id','LabMention','SignSymptomMention',
                                     'DiseaseDisorderMention', 'AnatomicalSiteMention','MedicationMention','ProcedureMention'])

        df_meta = pd.DataFrame()
        codes = set()

        for file in tqdm(xmi_files, total=len(xmi_files)):
            if args.test:
                xmldoc = md.parse(os.path.join(test_dir, file))
            else:
                xmldoc = md.parse(os.path.join(xmi_dir.replace('train', split),file))

            #initialize local data structures
            concepts = defaultdict(list)
            lookups = pd.DataFrame(columns=['lookup_id', 'codingScheme', 'code', 'preferredText'])
            df_local = pd.DataFrame(columns=['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type'])

            #only pass through the text once*
            for element in xmldoc.getElementsByTagName("xmi:XMI"):
                for el in element.childNodes:
                    if el.nodeType == Node.ELEMENT_NODE:
                        if 'Mention' in el.tagName:
                            #TODO: append something else*
                            concepts[el.tagName.split(":")[1]].append(el.attributes['ontologyConceptArr'].value)
                            #store all the tags for now, will match on the text and SNOMED ontology later...
                            df_local.loc[len(df_local)] = [file.split(".")[0],el.attributes['ontologyConceptArr'].value[-4:],int(el.attributes['begin'].value),int(el.attributes['end'].value),el.tagName.split(":")[1]]
                        elif 'refsem:UmlsConcept' in el.tagName:
                            lookups.loc[len(lookups)] = [el.attributes['xmi:id'].value,el.attributes['codingScheme'].value, el.attributes['code'].value, el.attributes['preferredText'].value]
                            codes.add(el.attributes['code'].value)
                        elif 'cas:Sofa' in el.tagName:
                            text = el.attributes['sofaString'].value

            #TODO: there seem to be extra codes in the file's UmlsConcept mappings that don't actually show up in the annotation

            #write stats to global DFs:
            if args.no_stats:
                df.loc[len(df)] = [file.split(".")[0], len(concepts['LabMention']),len(concepts['SignSymptomMention']),len(concepts['DiseaseDisorderMention']),
                               len(concepts['AnatomicalSiteMention']),len(concepts['MedicationMention']),len(concepts['ProcedureMention'])]

            #merge the lookups info with the
            df_local = df_local.merge(lookups, how='left', left_on='lookup_id', right_on='lookup_id')
            df_meta = df_meta.append(df_local)

            fn = lambda x: text[x['begin_inx']:x['end_inx']]
            df_meta['word_phrase'] = df_meta.apply(fn, axis=1)

            #once have parsed once, pass through again to match actual text term & its location to SNOMED value
            # TODO: write out individual CUIs and their text terms to file*

        b = datetime.datetime.now().replace(microsecond=0)
        print("Time to process %s files:" % split)
        print(str(b - a))

        print("Writing out...")
        a = datetime.datetime.now().replace(microsecond=0)
        if args.no_stats and args.test:
            df.to_csv(os.path.join(test_dir,'concept_extraction_stats.csv'))
        elif args.no_stats:
            df.to_csv(os.path.join(concept_write_dir, '%s_concept_extraction_stats.csv' % split))

        if args.test:
            df_meta.to_csv(os.path.join(test_dir,'train_concepts.csv'))
            with open(os.path.join(test_dir, '%s_meta_concepts.csv' % split), 'wb') as fp:
                for item in codes:
                    fp.write("%s\n" % item)
        else:
            df_meta.to_csv(os.path.join(concept_write_dir, '%s_concepts.csv' % split))
            #TODO
            with open(os.path.join(concept_write_dir, '%s_meta_concepts.csv' % split), 'wb') as fp:
                for item in codes:
                    fp.write("%s\n" % item)

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
    parser.add_argument('--no-stats', action='store_false', required=False, dest='no_stats', help='whether or not to collect and print stats about number of concepts extracted')
    parser.add_argument('--test', action='store_true', required=False, dest='test')
    args = parser.parse_args()
    main(args)