'''This file contains tools by Sarah for coding the model extension'''
from constants import *
import os
import csv
from tqdm import tqdm
import pickle

def map_icd_to_SNOMED():

	with open(os.path.join(DATA_DIR, 'ICD9CM_SNOMEDCT_map_201712/ICD9CM_SNOMED_MAP_1TOM_201712.txt'), 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		i = 0
		next(reader)
		icdToSnomed = {}
		for line in tqdm(reader):
			if line[7] != 'NULL':
				if line[0] in icdToSnomed.keys():
					icdToSnomed[line[0]].append(line[7])
				else:
					icdToSnomed[line[0]] = [line[7]]
			i += 1

	#do the same with the one-to-one mapping file
	with open(os.path.join(DATA_DIR, 'ICD9CM_SNOMEDCT_map_201712/ICD9CM_SNOMED_MAP_1TO1_201712.txt'), 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		i = 0
		next(reader)
		for line in tqdm(reader):
			if line[7] != 'NULL':
				if line[0] in icdToSnomed.keys():
					icdToSnomed[line[0]].append(line[7])
				else:
					icdToSnomed[line[0]] = [line[7]]
			i += 1

	print(len(icdToSnomed))

	#convert SNOMED to ICD
	snomedToICD = {}
	for k, v in icdToSnomed.items():
		for el in v:
			if el not in snomedToICD.keys():
				snomedToICD[el] = [k]
			else:
				snomedToICD[el].append(k)


	print(len(snomedToICD))

	#pickle.dump(snomedToICD, open(os.path.join(DATA_DIR, 'snomedToICD.pkl'), 'wb'))

def map_snomed_to_icd():

	with open(os.path.join(DATA_DIR, 'SnomedCT_UStoICD10CM_20180301T120000Z/tls_Icd10cmHumanReadableMap_US1000124_20180301.tsv'), 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		snomedToICD = {}
		for line in tqdm(reader):
			if line[5] in snomedToICD.keys():
				snomedToICD[line[5]].append(line[11])
			else:
				snomedToICD[line[5]] = [line[11]]

	print(len(snomedToICD))

	pickle.dump(snomedToICD, open(os.path.join(DATA_DIR, 'UPDATEDsnomedToICD10.pkl'), 'wb'))

def get_SNOMED_to_ICD_stats():

	#snomedToICD = pickle.load(open(os.path.join(DATA_DIR, 'snomedToICD.pkl'), 'rb'))
	snomedToICD = pickle.load(open(os.path.join(DATA_DIR, 'UPDATEDsnomedToICD10.pkl'), 'rb'))
	print(type(snomedToICD))

	text_file = open(os.path.join(DATA_DIR, 'dev_meta_concepts.csv'), 'r')
	lines = text_file.read().splitlines()
	text_file.close()

	print("found:", len([el for el in lines if el in snomedToICD]))
	print("out of", len(lines))

	#only about ~18% (2608/14704) found with the 12/2017 ICD9 to SNOMED mapping file**
	# 5351/14704 found with updated SNOMED to ICD9 codefile**

def map_extr_concepts_to_icd():


if __name__ == '__main__':
	#map_icd_to_SNOMED() #TODO: CONSIDER BOTH PROCS AND DIAGS**
	#map_snomed_to_icd()
	#get_SNOMED_to_ICD_stats()
	map_extr_concepts_to_icd()



