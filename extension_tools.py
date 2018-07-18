'''This file contains tools by Sarah for coding the model extension'''
from constants import *
import os
import csv
from tqdm import tqdm
import pickle
import datetime
import pandas as pd
from ast import literal_eval
import re
from collections import defaultdict
import datasets
from dataproc import extract_wvs
import numpy as np
import argparse

'''Overview of Methods: 

	map_icd_to_SNOMED(): 
	map_snomed_to_icd(): initial methods for manually mapping ICD codes to SNOMED (& vice versa), before setup NLM automatic mapping during parsing.
	get_SNOMED_to_ICD_stats(): 
	map_extr_concepts_to_icd()

	restructure_concepts_for_batched_input(): can't remember...

	get_concept_text_alignment(inpt_file, split, outpt_file): this one is used to create the patient matrices based on the extracted concepts for each file/split**
	get_concept_matrix(sub, text): same method, but can be called from another file (called before from parse_xmi.py)**

	get_parent_trees(): this is used to build the child-to-parent dictionary based on the ICD9-to-CCS groupings (might need this again later)
	update_vocab(dirs_map, old_vocab, out_dir, load=False): 
	compute_pairs_vocab_dict(filename, concepts_file): 
	build_concept_embeddings_matrix(description_dir, embed_file): builds the concept embeddings matrix based on 
	
'''


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

#TODO: DO THIS AS A PREPROC STEP!
def map_extr_concepts_to_icd():

	snomedToICD = pickle.load(open(os.path.join(DATA_DIR, 'snomedToICD.pkl'), 'rb'))
	#TODO: USING ICD9 FOR NOW**
	print("Length SNOMED key-mapping dictionary:", len(snomedToICD))

	for split in ['train', 'test', 'dev']:
		#go find the extracted concepts files:
		print("Converting %s files..." % split)
		s = datetime.datetime.now().replace(microsecond=0)

		#get metadata files
		with open(os.path.join(concept_write_dir, '%s_meta_concepts_SNOMED.csv' % split), 'r') as f:
			with open(os.path.join(concept_write_dir, '%s_meta_concepts.txt' % split), 'w') as g:
				lines = f.read().splitlines()
				subset = [snomedToICD[el] for el in lines if el in snomedToICD.keys()]
				print("number of codes in the dictionary:", len(subset))
				print("out of:", len(lines))
				for item in subset:
					g.write("%s\n" % item)

		#add to actual file--
			#load as pandas dataframe
		a = datetime.datetime.now().replace(microsecond=0)

		try: 
			df = pd.read_csv(os.path.join(concept_write_dir, '%s_concepts_SNOMED.csv' % split))
			df['code'] = df['code'].apply(str)

			b = datetime.datetime.now().replace(microsecond=0)
			print("Time to read pandas dataframe:", str(b-a))

			print(df.codingScheme.unique())
			print(df.shape)
			df = df.loc[df.codingScheme == 'SNOMEDCT_US']
			print(df.shape)
			df = df.loc[df.code.isin(snomedToICD.keys())]
			print(df.shape)

			fn = lambda x: snomedToICD[x]
			df['ICD9_code'] = df['code'].apply(fn)

			#rewrite out to file:
			a = datetime.datetime.now().replace(microsecond=0)

			df.to_csv(os.path.join(concept_write_dir, '%s_concepts.csv' % split))

			b = datetime.datetime.now().replace(microsecond=0)
			print("Time to read pandas dataframe:", str(b-a))

			e = datetime.datetime.now().replace(microsecond=0)
			print("Time to process %s files:" % split, str(e - s))

		except pd.io.common.EmptyDataError:
			open(os.path.join(concept_write_dir, '%s_concepts.csv' % split), 'w')

def restructure_concepts_for_batched_input():

	#TODO: ALIGN HERE FOR NON-CLEANED & PRE-PARSED TEXT*
 
	#for split in ['train', 'test', 'dev']: #TODO: PUT BACK**
	for split in ['train']:

		#go find the extracted concepts files:
		print("Merging %s file on patient..." % split)
		s = datetime.datetime.now().replace(microsecond=0)

		#load as pandas dataframe
		a = datetime.datetime.now().replace(microsecond=0)

		#try: 
		df = pd.read_csv(os.path.join(concept_write_dir, '%s_concepts.csv' % split))

		df = df[['begin_inx','ICD9_code','end_inx','patient_id','word_phrase']]
		# df.begin_inx = df.begin_inx.apply(str)
		# df.end_inx = df.end_inx.apply(str)
		# df.patient_id = df.patient_id.apply(str)
		# df.ICD9_code = df.ICD9_code.apply(str)

		# b = datetime.datetime.now().replace(microsecond=0)
		# print("Time to read pandas dataframe:", str(b-a))

		# a = datetime.datetime.now().replace(microsecond=0)
		
		# df = df.groupby('patient_id', as_index=False).agg({'begin_inx': lambda x: ';'.join(list(x)), 'end_inx': lambda x: ';'.join(list(x)), 'word_phrase': lambda x: ';'.join(list(x)), 'ICD9_code': lambda x: ';'.join(list(x))})

		# b = datetime.datetime.now().replace(microsecond=0)
		# print("Time to group:", str(b-a))

		# #rewrite out to file:
		# a = datetime.datetime.now().replace(microsecond=0)

		# df.to_csv(os.path.join(concept_write_dir, '%s_concepts_JOINED.csv' % split))

		# b = datetime.datetime.now().replace(microsecond=0)
		# print("Time to read pandas dataframe:", str(b-a))

		# e = datetime.datetime.now().replace(microsecond=0)
		# print("Time to process %s files:" % split, str(e - s))


		# except pd.io.common.EmptyDataError:
		# 	open(os.path.join(concept_write_dir, '%s_concepts.csv' % split), 'w')

		# #instead, write out to file for each one
		for pat in df.patient_id.unique().tolist():
			df_sub = df.loc[df.patient_id == pat]

			#create file for this patient
			f = open(os.path.join(concept_write_dir, 'patient_extracted_concepts/%s.txt' % pat), 'w') #TODO: FOR SH, THIS MUST BE PATIENT+NOTE*
			writer = csv.writer(f)
			for inx, row in df_sub.iterrows():
				writer.writerow(row)
			f.close()

def get_concept_text_alignment(inpt_file, split, outpt_file, lst_terms, dir_name, vocab=False):

	if vocab == True:
		concept_vocab = {}

	a = datetime.datetime.now().replace(microsecond=0)

	#get concepts here
	df_CONCEPTS = pd.read_csv(inpt_file)
	print("Shape:", df_CONCEPTS.shape)
	#subset down to those terms we want
	print(list(df_CONCEPTS.codingScheme.unique()))
	print(lst_terms)
	df_CONCEPTS = df_CONCEPTS.loc[df_CONCEPTS.codingScheme.isin(lst_terms)]
	print("Shape after subset:", df_CONCEPTS.shape)
	print(df_CONCEPTS.head())

	patient_concepts_matrix = {}
	missed_concepts = 0
	multi_words = 0
	unequal_text = 0
	total = 0
	overlaps = 0
	missed_patients = 0

	with open(os.path.join(MIMIC_3_DIR, '%s_full.csv' % split), 'r') as f:
		reader = csv.reader(f)
		next(reader)
		for line in reader:
			#new patient
			text = line[2]
			pat_note_id = line[0] + '_' + line[1]

			#get concepts
			sub = df_CONCEPTS.loc[df_CONCEPTS.patient_id == pat_note_id]

			#SUBSET DOWN TO CODES WE WANT**
			#for now, just ICD9 and RXNORM

			if sub.shape[0] == 0:
				print("NO ROWS FOR:", pat_note_id)
				missed_patients += 1

			words = text.strip().split()
			concept_arr = [0] * len(words)

			starting_inxs = [0] + [m.start()+1 for m in re.finditer(' ', text)]
			ending_inxs = [m.end()-1 for m in re.finditer(' ', text)] + [len(text)]

			#so now, these indices can be used to mark word positions in sep. text, aka text[starting_inxs[0]:ending_inxs[0]] == words[0]

			if len(words) != len(starting_inxs) or len(words) != len(ending_inxs):
				raise Exception("not the same length!")

			#POPULATE
			pat_codes = set()
			for _, row in sub.iterrows():

				#get the beginning and end index positions, the ICD9 code, etc.
				#row['begin_inx'] #these indices are the same as python indexing! (nice)**
				#row['end_inx']
				#row['word_phrase']
				#TODO: MAPPING HERE***

				if row['begin_inx'] in starting_inxs and row['end_inx'] in ending_inxs:

					#write code to position in array
					word_pos = starting_inxs.index(row['begin_inx'])
					# if concept_arr[word_pos] != 0:
					# 	if isinstance(concept_arr[word_pos], list):
					# 		concept_arr[word_pos].append(row['code'])
					# 	else: 
					# 		concept_arr[word_pos] = [concept_arr[word_pos]] + [row['code']]
					# 	overlaps += 1
					# else:
					# 	concept_arr[word_pos] = row['code']

					#TODO:**for now, just make single code version**
					concept_arr[word_pos] = row['code']
					pat_codes.add(row['code'])

					#check for overlap
					end_pos = ending_inxs.index(row['end_inx'])
					if word_pos != end_pos:
						multi_words += 1
						# if concept_arr[end_pos] != 0: #if we have an overlap, append to list: TODO CHECK THIS
						# 	if isinstance(concept_arr[end_pos], list):
						# 		concept_arr[end_pos].append(row['code'])
						# 	else: 
						# 		concept_arr[end_pos] = [concept_arr[end_pos]] + [row['code']]
						# 	overlaps += 1
						# else:
						# 	concept_arr[end_pos] = row['code']

						#TODO:**for now, just make single code version**
						concept_arr[end_pos] = row['code']
						pat_codes.add(row['code'])

					if row['word_phrase'] != ' '.join(words[word_pos:end_pos+1]): #check text equal
						print(row['word_phrase'])
						print(' '.join(words[word_pos:end_pos+1]))
						unequal_text += 1

				else:
					print("ALIGNMENT ISSUE-MISSED CONCEPT")
					missed_concepts += 1

			total += 1

			#add to array
			patient_concepts_matrix[pat_note_id] = concept_arr

			if vocab:
				#add each el of pat_codes to vocab
				for el in pat_codes:
					if el in concept_vocab:
						concept_vocab[el] += 1
					else:
						concept_vocab[el] = 1


	print("number of missed concepts:", missed_concepts)
	print("number of missed patients:", missed_patients)
	print("number of multi-word phrases:", multi_words)
	print("number of unequal texts:", unequal_text)
	print("num overlaps: ", overlaps)
	print("total:", total)
	pickle.dump(patient_concepts_matrix, open(outpt_file, 'wb'))

	b = datetime.datetime.now().replace(microsecond=0)
	print("Time to process %s files:" % split, str(b-a))

	#TESTING-------------------------------------------
	# print(patient_concepts_matrix['84392_129675'])
	# print("SHOULD HAVE LENGTH 1644:")
	# print(len(patient_concepts_matrix['84392_129675']))
	# print("SHOULD HAVE 56 NON-ZERO CONCEPTS")
	#--------------------------------------------------

	if vocab:
		#subset down the vocab and dump, after processing all training examples:
		new_vocab = set()
		for inx, cnt in concept_vocab.items():
			if cnt >= 3:
				new_vocab.add(inx)

		print("Old vocab size:", len(concept_vocab))
		print("New vocab size:", len(new_vocab))

		with open('/data/swiegreffe6/NEW_MIMIC/extracted_concepts/' + dir_name + '/concept_vocab_%s_children.txt' % dir_name, 'w') as new:
			for line in iter(new_vocab):
				new.write("%s\n" % line)

def get_concept_matrix(sub, text):

	words = text.strip().split()
	concept_arr = [0] * len(words)

	starting_inxs = [0] + [m.start()+1 for m in re.finditer(' ', text)]
	ending_inxs = [m.end()-1 for m in re.finditer(' ', text)] + [len(text)]

	#so now, these indices can be used to mark word positions in sep. text, aka text[starting_inxs[0]:ending_inxs[0]] == words[0]

	if len(words) != len(starting_inxs) or len(words) != len(ending_inxs):
		raise Exception("not the same length!")

	#POPULATE
	for _, row in sub.iterrows():

		#get the beginning and end index positions, the ICD9_code, etc.
		#row['begin_inx'] #these indices are the same as python indexing! (nice)**
		#row['end_inx']
		#row['word_phrase']
		#TODO: MAPPING HERE***

		if row['begin_inx'] in starting_inxs and row['end_inx'] in ending_inxs:

			#write code to position in array
			word_pos = starting_inxs.index(row['begin_inx'])
			# if concept_arr[word_pos] != 0:
			# 	if isinstance(concept_arr[word_pos], list):
			# 		concept_arr[word_pos].append(row['ICD9_code'])
			# 	else: 
			# 		concept_arr[word_pos] = [concept_arr[word_pos]] + [row['ICD9_code']]
			# 	overlaps += 1
			# else:
			# 	concept_arr[word_pos] = row['ICD9_code']

			#TODO:**for now, just make single code version**
			concept_arr[word_pos] = row['code']

			#check for overlap
			end_pos = ending_inxs.index(row['end_inx'])
			if word_pos != end_pos:
				# if concept_arr[end_pos] != 0: #if we have an overlap, append to list: TODO CHECK THIS
				# 	if isinstance(concept_arr[end_pos], list):
				# 		concept_arr[end_pos].append(row['ICD9_code'])
				# 	else: 
				# 		concept_arr[end_pos] = [concept_arr[end_pos]] + [row['ICD9_code']]
				# 	overlaps += 1
				# else:
				# 	concept_arr[end_pos] = row['ICD9_code']

				#TODO:**for now, just make single code version**
				concept_arr[end_pos] = row['code']

		else:
			print("ALIGNMENT ISSUE-MISSED CONCEPT")

	return concept_arr


def get_parent_trees():
	'''This method takes the input codeset and calculates the parent trees from them, adding the parent codes to the full set as well.
	We also create a dictionary mapping from child to parents.'''

	'''code heavily influenced by Edward Choi's build_trees script from the original GRAM implementation,
	found here: https://github.com/mp2893/gram/blob/master/build_trees.py'''

	def diag_process(code):
		if code.startswith('E'):
			if len(code) > 4: code = code[:4] + '.' + code[4:]
		else:
			if len(code) > 3: code = code[:3] + '.' + code[3:]
		return code

	def procedure_process(code):
		code = code[:2] + '.' + code[2:]
		return code

	# df_mapping_diagnoses = pd.read_csv(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_dx_tool_2015.csv'))
	# df_mapping_procedures = pd.read_csv(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_pr_tool_2015.csv'))

	# #fix ICD9 format
	# fn = lambda x: diag_process(x[1:-1].strip())
	# fn2 = lambda x: procedure_process(x[1:-1].strip())

	# df_mapping_diagnoses['\'ICD-9-CM CODE\''] = df_mapping_diagnoses['\'ICD-9-CM CODE\''].apply(fn)

	# #do the same for procedures:
	# df_mapping_procedures['\'ICD-9-CM CODE\''] = df_mapping_procedures['\'ICD-9-CM CODE\''].apply(fn2)

	# #write out for testing/posterity
	# with open(os.path.join(concept_write_dir, 'DIAG_CODES.txt'), 'w') as the_file:
	# 	for i, row in df_mapping_diagnoses.iterrows():
	# 		the_file.write(row['\'ICD-9-CM CODE\'']+ '\n')

	# with open(os.path.join(concept_write_dir, 'PROC_CODES.txt'), 'w') as the_file:
	# 	for i, row in df_mapping_procedures.iterrows():
	# 		the_file.write(row['\'ICD-9-CM CODE\'']+ '\n')

	# codes = set()
	# #make a list of codes and see how many can map
	# for i, row in df_mapping_diagnoses.iterrows():
	# 	codes.add(row['\'ICD-9-CM CODE\''])

	# for i, row in df_mapping_procedures.iterrows():
	# 	codes.add(row['\'ICD-9-CM CODE\''])


	# with open(os.path.join(concept_write_dir, '%s_meta_concepts.txt' % split), 'r') as f:
	# 	lines = f.read().splitlines()
	# 	#these are the child codes
	# 	for line in lines:
	# 		if line not in codes:
	# 			print(line) #Just one not present**

	#get a mapping for parents: taken from Ed's code
#------------------------------------------------------------------------------------------

	a = datetime.datetime.now().replace(microsecond=0)

	dirs_map = defaultdict(list)#TODO: RETURN SELF + ROOTCODE**)

	#TODO: INCLUDE A ROOT CODE?? For now, if doesn't have any parents in CCS, just use that embedding**
	#TODO: HERE, MAKE SURE DUPLICATE ALL CHANGES FOR BOTH PROCS AND DIAGS FILES :)

	infd = open(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_dx_tool_2015.csv'), 'r')
	_ = infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		icd9 = tokens[0][1:-1].strip()
		cat1 = tokens[1][1:-1].strip()
		cat2 = tokens[3][1:-1].strip()
		cat3 = tokens[5][1:-1].strip()
		cat4 = tokens[7][1:-1].strip()

		icdCode = diag_process(icd9)

		if len(cat4) > 0:
			dirs_map[icdCode] = [icdCode, cat4, cat3, cat2, cat1, rootCode]
		elif len(cat3) > 0:
			dirs_map[icdCode] = [icdCode, cat3, cat2, cat1, rootCode]
		elif len(cat2) > 0:
			dirs_map[icdCode] = [icdCode, cat2, cat1, rootCode]
		elif len(cat1) > 0:
			dirs_map[icdCode] = [icdCode, cat1, rootCode]

	infd.close()

	#do the same things for the procedure codes: Note only have max. 3 levels vs. diagnoses' 4
	infd = open(os.path.join(DATA_DIR, 'Multi_Level_CCS_2015/ccs_multi_pr_tool_2015.csv'), 'r')
	_ = infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		icd9 = tokens[0][1:-1].strip()
		cat1 = tokens[1][1:-1].strip()
		cat2 = tokens[3][1:-1].strip()
		cat3 = tokens[5][1:-1].strip()

		icdCode = procedure_process(icd9)

		if len(cat3) > 0:
			dirs_map[icdCode] = [icdCode, cat3, cat2, cat1, rootCode]
		elif len(cat2) > 0:
			dirs_map[icdCode] = [icdCode, cat2, cat1, rootCode]
		elif len(cat1) > 0:
			dirs_map[icdCode] = [icdCode, cat1, rootCode]

	infd.close()

	print(len(dirs_map))

	#TODO: we want to make sure we have a mapping for each code in our original vocabulary-- this is coded into the main model code by the defaultdict

	print(dirs_map)
	pickle.dump(dirs_map, open(os.path.join(concept_write_dir, 'code_parents.p'), 'wb'))

	update_vocab(dirs_map, os.path.join(concept_write_dir, 'train_meta_concepts.txt'), concept_write_dir)


def update_vocab(dirs_map, old_vocab, out_dir, load=False):

	if load:
		d = pickle.load(open(dirs_map, 'rb'))
	else:
		d= dirs_map

	codes = set()
	old_codes = set()
	#TODO: UPDATE
	with open(old_vocab, 'r') as old:
		for line in old:
			line = line.strip()
			codes.add(line)
			old_codes.add(line)
			for el in d[line]:
				codes.add(el)

	with open(os.path.join(out_dir, 'concept_vocab.txt'), 'w') as new:
		for line in iter(codes):
			new.write("%s\n" % line)

	print("Length child codes:", len(old_codes))
	print("Number of parent + child codes:", len(codes))

def compute_pairs_vocab_dict(filename, concepts_file):

	#get 4513 unique from the full MIMIC3 train_full file, 3466 once subset to >1 occurrence.

	concept_dict = pickle.load(open(concepts_file, 'rb'))

	concept_pairs = {}
	with open(filename, 'r') as infile:
		r = csv.reader(infile)
		#header
		next(r)
		for row in tqdm(r):
			text = row[2]
			joint_id = row[0] + '_' + row[1]
			words = text.strip().split()

			#cut to max length
			con = concept_dict[joint_id]

			#truncate concepts
			if len(con) > MAX_LENGTH: #TODO: UNDO FOR SH CASE**
				con = con[:MAX_LENGTH]

			#truncate long documents
			if len(words) > MAX_LENGTH: #TODO: UNDO FOR SH CASE**
				words = words[:MAX_LENGTH]

			assert len(words) == len(con)

			#add pairs and counts
			for w, c in zip(words, con):
				if (w,c) in concept_pairs and w != 0 and c!=0:
					concept_pairs[(w,c)] += 1
				elif w != 0 and c != 0:
					concept_pairs[(w,c)] = 1

	print("Length concept-word pairs before reduction:", len(concept_pairs))

	#subset down concept pairs to only those with a >1 occurrence:
	#TODO: measuring based on total concept occurrences, not distinct document occs*
	concept_pairs_vocab = set([k for k, v in concept_pairs.items() if v > 1])
	print("Length after:", len(concept_pairs_vocab))

	#create dictionary
	concept_word_dict = {c:i for i,c in enumerate(sorted(concept_pairs_vocab), 1)}

	#write out
	pickle.dump(concept_word_dict, open('/Users/SWiegreffe/Desktop/mimicdata/new_files/concept_word_dict.p','wb'))

def build_concept_embeddings_matrix(description_dir, embed_file):

	dv_dict = datasets.load_description_vectors(description_dir)
	concept_embeds = {}

	#extract word embeddings from file but DON'T NORMALIZE THEM:
	W = []
	vocab = []
	with open(embed_file) as ef:
	    for line in ef:
	        line = line.rstrip().split()
	        vec = np.array(line[1:]).astype(np.float)
	        W.append(vec)
	        vocab.append(line[0])
	vocab = vocab[1:]
	W = np.array(W)
	print("Embedding shape:", W.shape)
	w2ind = {k:i for i,k in enumerate(vocab,1)}	
	print(len(w2ind))

	#now build the concept embeddings
	i = 0
	for key, val_lst in dv_dict.items():
		#remove UNK tokens
		new_vals = [e for e in val_lst if e != W.shape[0]]
		if len(new_vals) > 0:
			embed = np.zeros(len(W[0,:]))
			for el in new_vals:
				#get word embedding
				embed += W[el]
			embed = embed / float(len(new_vals)) #sum up all the embeddings which are not UNK tokens
			concept_embeds[key] = embed
		else: 
			#if have no embeddings for any of the words in the code description,
			#randomly gaussian initialize
			concept_embeds[key] = np.random.randn(len(W[0,:]))
			i += 1

	print("Initial dictionary size", len(dv_dict))
	print("New dictionary size:", len(concept_embeds))
	print("Uninitializable embeds:", i)
	#save dict
	pickle.dump(concept_embeds, open(os.path.join(MIMIC_3_DIR, 'concepts.embed'), 'wb'), protocol=2)


def main(args):

	if args.method == 'get_concept_text_alignment':
		assert args.input_dir is not None
		assert args.output_dir is not None
		assert args.split is not None
		assert args.lst_terms is not None
		assert args.dir_name is not None

		get_concept_text_alignment(args.input_dir, args.split, args.output_dir, args.lst_terms, args.dir_name, args.vocab)

	elif args.method == 'update_vocab':
		assert args.dirs_map is not None
		assert args.vocab_file is not None
		assert args.output_dir is not None

		update_vocab(args.dirs_map, args.vocab_file, args.output_dir, args.load)

	#TODO: continue other methods here!

if __name__ == '__main__':

	'''Example/past commands: 

		map_icd_to_SNOMED() #TODO: CONSIDER BOTH PROCS AND DIAGS**
		map_snomed_to_icd()
		get_SNOMED_to_ICD_stats()
		map_extr_concepts_to_icd()
		restructure_concepts_for_batched_input()
		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_train_ALL.csv', 'train', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM/train_patient_concepts_matrix_ICD9_RXNORM.p', ['ICD9CM', 'RXNORM'], 'ICD9_RXNORM', vocab=True)
		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_train_ALL.csv', 'train', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM_SNOMED/train_patient_concepts_matrix_ICD9_RXNORM_SNOMED.p', ['ICD9CM', 'RXNORM', 'SNOMEDCT_US'], 'ICD9_RXNORM_SNOMED', vocab=True)
	
		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_dev_ALL.csv', 'dev', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM/dev_patient_concepts_matrix_ICD9_RXNORM.p', ['ICD9', 'RXNORM'], 'ICD9_RXNORM', vocab=False)
		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_test_ALL.csv', 'test', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM/test_patient_concepts_matrix_ICD9_RXNORM.p', ['ICD9', 'RXNORM'], 'ICD9_RXNORM', vocab=False)

		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_dev_ALL.csv', 'dev', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM_SNOMED/dev_patient_concepts_matrix_ICD9_RXNORM_SNOMED.p', ['ICD9', 'RXNORM', 'SNOMEDCT_US'], 'ICD9_RXNORM_SNOMED', vocab=False)
		get_concept_text_alignment('/data/swiegreffe6/NEW_MIMIC/patient_notes/concepts_test_ALL.csv', 'test', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM_SNOMED/test_patient_concepts_matrix_ICD9_RXNORM_SNOMED.p', ['ICD9', 'RXNORM', 'SNOMEDCT_US'], 'ICD9_RXNORM_SNOMED', vocab=False)
	
		get_parent_trees('train')

		update_vocab('/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9/code_parents.p', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM/concept_vocab_ICD9_RXNORM_children.txt', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM/', load=True)
		update_vocab('/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9/code_parents.p', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM_SNOMED/concept_vocab_ICD9_RXNORM_SNOMED_children.txt', '/data/swiegreffe6/NEW_MIMIC/extracted_concepts/ICD9_RXNORM_SNOMED/', load=True)

		compute_pairs_vocab_dict(filename='/Users/SWiegreffe/Desktop/mimicdata/new_files/train_full_SUBSET.csv', concepts_file='/Users/SWiegreffe/Desktop/mimicdata/new_files/train_patient_concepts_matrix.p')
		build_concept_embeddings_matrix('/Users/SWiegreffe/Desktop/mimicdata/mimic3/description_vectors.vocab', '/Users/SWiegreffe/Desktop/mimicdata/processed_full.embed')
	
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('method', type=str, choices=['get_concept_text_alignment','update_vocab','compute_pairs_vocab_dict','build_concept_embeddings_matrix'])
	parser.add_argument('--split', type=str, required=False, help='split')
	parser.add_argument("--input-dir", type=str, required=False, dest='input_dir')
	parser.add_argument("--output-dir", type=str, required=False, dest='output_dir')
	parser.add_argument("--codes-list", nargs='+', required=False, dest='lst_terms')
	parser.add_argument("--dir-name", type=str, required=False, dest='dir_name')
	parser.add_argument("--construct-vocab", action='store_true', required=False, dest='vocab')

	parser.add_argument("--vocab-file", type=str, required=False, dest='vocab_file')
	parser.add_argument("--dirs-map", type=str, required=False, dest='dirs_map')
	parser.add_argument("--load-dirs-map", action='store_true', required=False, dest='load')
	args = parser.parse_args()
	main(args)


