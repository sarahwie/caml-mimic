import os
import pickle

if __name__ == "__main__":

	for split in ['train', 'test', 'dev']:

		substring = 'patient_concepts_matrix_%s_dir' % split
		path = '/data/mimicdata/mimic3/patient_notes/extracted_concepts/'

		files = [f for f in os.listdir(path) if substring in f]

		print(files)
		print(len(files))

		result = {}
		lengths = []
    	for f in files:
    		#load dictionary
    		dictionary = pickle.load(open(path + f, 'rb'))

    		lengths.append(len(dictionary))
        	result.update(dictionary)

        print("TOTAL LENGTH", len(result))
        print("DOES THIS MATCH TOTAL? : ", sum(result))

        new_name = "%s_patient_concepts_matrix.p" % split
		pickle.dump(result, open(path + new_name, 'wb'))