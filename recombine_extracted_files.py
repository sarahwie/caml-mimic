import os
import datetime
import argparse
import pandas as pd

'''This method takes the respective extracted/created files from the manual parallelized process of parse_xmi.py
In future, probably a good to python multiprocess this/queue file write output'''

def main(args):

	input_dir = '/data/swiegreffe6/NEW_MIMIC/extracted_concepts'
	out_dir = '/data/swiegreffe6/NEW_MIMIC/patient_notes'

	recombine_all_files(args, input_dir, out_dir)
	#remerge_dictionary(args, input_dir, out_dir)

def recombine_all_files(args, input_dir, out_dir):

	print("Parsing files from %s..." % input_dir)

	split = args.split

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



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('name', type=str, help='file extension to add for naming purposes')
	parser.add_argument('split', type=str, help='split')
	args = parser.parse_args()
	main(args)

