import os
import datetime
import argparse
import pandas as pd

'''This method takes the respective extracted/created files from the manual parallelized process of parse_xmi.py
In future, probably a good to python multiprocess this/queue file write output'''

def main(args):

    recombine_all_files(args)
    remerge_dictionary()
    concepts_vocab_test_dir_005_all_except_ICD9.csv

def recombine_all_files(args):

	input_dir = '/data/swiegreffe6/NEW_MIMIC/extracted_concepts'
	out_dir = '/data/swiegreffe6/NEW_MIMIC/patient_notes'

    print("Parsing files from %s..." % input_dir)

 #    output_dir = os.path.join(input_dir, args.name)
	# if not os.path.exists(output_dir):
	# 	os.mkdir(output)

    for split in ['dev', 'test', 'train']:

    	a = datetime.datetime.now().replace(microsecond=0)

    	print("Parsing split %s" % split)

	    child_files = [os.path.abspath(e) for e in os.listdir(input_dir) if 'concepts_%s_dir_' % split in e]
	    print("Files to process:", len(child_files))
	    print(child_files)

	    summed_len = 0
		for inx, file in enumerate(child_files):
			if inx == 0:
				#first file
				df = pd.read_csv(open(file, 'rb'))
				summed_len += df.shape[0]
				cols = list(df.columns)
			else:
				new_df = pd.read_csv(open(file, 'rb'))
				summed_len += new_df.shape[0]
				new_df = new_df[[cols]]
				df.append(new_df)

			#load in as pandas df
			#reorder columns
			#merge

		print(df.shape)
		assert df.shape[0] == summed_len

		outfile_name = 'concepts_' + split + '_' + args.name + '.csv'
		df.to_csv(os.path.join(out_dir, outfile_name))

	    b = datetime.datetime.now().replace(microsecond=0)
	    print("Time to parse files:", str(b - a))


'''This method is due to us manually parallelizing the creation of the training data concept vocab.
We now need a method to merge them back together'''
def remerge_dictionary():

	concepts = set()
	with open('/data/mimicdata/mimic3/patient_notes/extracted_concepts/concepts_vocab_train_ICD9.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader) #no header
		for line in reader:
			concepts.add(line[0]) #TODO: here, could instead join all the rows w/ the same concept id and then use the value in line[1] post-join to cut by occ. threshold

	print(len(concepts)) #TODO: probably worth a check here that actually aligns with other file**

	#write back out to file
	with open('/data/mimicdata/mimic3/patient_notes/extracted_concepts/concept_vocab_children.txt', 'w') as new:
		for line in iter(concepts):
			new.write("%s\n" % line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='file extension to add for naming purposes')
    args = parser.parse_args()
    main(args)

