from xml.dom import minidom as md
from xml.dom.minidom import Node
import os
from collections import defaultdict
import pandas as pd
import datetime
from tqdm import tqdm

def parse_xmi(input_dir, already_found, directory_counts, no_stats=False):

    df_meta = pd.DataFrame()

    if input_dir.split(os.sep)[-1] not in directory_counts:
        xmi_files = [os.path.join(input_dir, e) for e in os.listdir(input_dir) if '.xm' in e]
    elif len(os.listdir(input_dir)) != len(directory_counts[input_dir.split(os.sep)[-1]]):
        xmi_files = [os.path.join(input_dir, e) for e in os.listdir(input_dir) if '.xm' in e if input_dir.split(os.sep)[-1] + '_' + e.split(os.sep)[-1].split(".")[0] not in already_found]
    else:
        xmi_files = []

    if len(xmi_files) != 0:
        for file in tqdm(xmi_files, total=len(xmi_files)):
            # get patient-note id
            note_id = file.split(os.sep)[-1].split(".")[0]
            pat_id = input_dir.split(os.sep)[-1]

            xmldoc = md.parse(os.path.join(input_dir, file))

            # initialize local data structures
            concepts = defaultdict(list)
            lookups = pd.DataFrame(columns=['lookup_id', 'codingScheme', 'code', 'preferredText'])
            df_local = pd.DataFrame(columns=['patient_id', 'lookup_id', 'begin_inx', 'end_inx', 'mention_type'])

            # only pass through the text once*
            cnt = 1
            for element in xmldoc.getElementsByTagName("xmi:XMI"):
                for el in element.childNodes:
                    if el.nodeType == Node.ELEMENT_NODE:
                        if 'Mention' in el.tagName:
                            concepts[el.tagName.split(":")[1]].append(el.attributes['ontologyConceptArr'].value)
                            # store all the tags for now
                            df_local.loc[len(df_local)] = [pat_id + '_' + note_id,
                                                           el.attributes['ontologyConceptArr'].value.split(' ')[-1],
                                                           int(el.attributes['begin'].value),
                                                           int(el.attributes['end'].value), el.tagName.split(":")[1]]
                        # WE ARE EXTRACTING ALL CONCEPTS HERE**
                        elif 'refsem:UmlsConcept' in el.tagName:
                            try:
                                cnt += 1
                                lookups.loc[len(lookups)] = [el.attributes['xmi:id'].value,
                                                             el.attributes['codingScheme'].value,
                                                             el.attributes['code'].value,
                                                             el.attributes['preferredText'].value]
                            except KeyError:
                                pass
                        elif 'cas:Sofa' in el.tagName:
                            text = el.attributes['sofaString'].value

            # merge the lookups info with the
            df_local = df_local.merge(lookups, how='inner', left_on='lookup_id', right_on='lookup_id')

            if not df_local.empty:
                df_local['code'] = df_local['code'].str.encode('utf-8')

            # add in text info
            if not df_local.empty:
                fn = lambda x: text[int(x['begin_inx']):int(x['end_inx'])]
                df_local['word_phrase'] = df_local.apply(fn, axis=1)

            df_meta = df_meta.append(df_local)

    return df_meta
