from xml.dom import minidom as md
from xml.dom.minidom import Node
import os
from constants import *
from collections import namedtuple, defaultdict
import pandas as pd
import numpy as np
import argparse


def main(args):

    parse_xmi()

    build_concept_vocab()

    for split in ['train', 'test', 'dev']:

        if args.test:
            all_files = os.listdir(test_dir)

        else:
            all_files = os.listdir(xmi_dir.replace('train', split))

        xmi_files = filter(lambda x: '.xm' in x, all_files)

        if args.test:
            print(xmi_files)

        #store stats/extracted concepts in pandas df:
        if args.stats:
            df = pd.DataFrame(columns = ['patient_id','LabMention','SignSymptomMention',
                                     'DiseaseDisorderMention', 'AnatomicalSiteMention','MedicationMention','ProcedureMention'])

        df_meta = pd.DataFrame()

        for file in xmi_files:
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
                for a in element.childNodes:
                    if a.nodeType == Node.ELEMENT_NODE:
                        if 'Mention' in a.tagName:
                            #TODO: append something else*
                            concepts[a.tagName.split(":")[1]].append(a.attributes['ontologyConceptArr'].value)
                            #store all the tags for now, will match on the text and SNOMED ontology later...
                            df_local.loc[len(df_local)] = [file.split(".")[0],a.attributes['ontologyConceptArr'].value[-4:],int(a.attributes['begin'].value),int(a.attributes['end'].value),a.tagName.split(":")[1]]
                        elif 'refsem:UmlsConcept' in a.tagName:
                            lookups.loc[len(lookups)] = [a.attributes['xmi:id'].value,a.attributes['codingScheme'].value, a.attributes['code'].value, a.attributes['preferredText'].value]

                        elif 'cas:Sofa' in a.tagName:
                            text = a.attributes['sofaString'].value

            #TODO: there seem to be extra codes in the file's UmlsConcept mappings that don't actually show up in the annotation

            #write stats to global DFs:
            df.loc[len(df)] = [file.split(".")[0], len(concepts['LabMention']),len(concepts['SignSymptomMention']),len(concepts['DiseaseDisorderMention']),
                               len(concepts['AnatomicalSiteMention']),len(concepts['MedicationMention']),len(concepts['ProcedureMention'])]

            #merge the lookups info with the
            df_local = df_local.merge(lookups, how='left', left_on='lookup_id', right_on='lookup_id')
            df_meta = df_meta.append(df_local)

            fn = lambda x: text[x['begin_inx']:x['end_inx']]
            df_meta['word_phrase'] = df_meta.apply(fn, axis=1)

            #once have parsed once, pass through again to match actual text term & its location to SNOMED value
            # TODO: write out individual CUIs and their text terms to file*

        df.to_csv(os.path.join(second_dir,'concept_extraction_stats.csv'))
        df_meta.to_csv(os.path.join(second_dir,'EXTRACTED_CONCEPTS.csv'))


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
    parser.add_argument('--stats', type=bool, required=True, default=False, dest='stats', help='whether or not to collect and print stats about number of concepts extracted')
    parser.add_argument('--test', type=bool, required=False, default=False, dest='test')
    args = parser.parse_args()
    main(args)