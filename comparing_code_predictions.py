import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan, suppress=True)

def main():

	get_code_rarities()


def get_code_rarities():

	true_codes = []
	hadm_ids = []
	new_hadm_ids = []
	pred_codes = []

	#first calculate the top k
	counts = Counter()

	#define a vocab of the codes we have as true labels
	vocab = set()

	#get counts from training set, get true occurrences from test set
	train_file = '/Users/SWiegreffe/Desktop/mimicdata/test_full.csv' #TODO: UPDATE**
	train_full = pd.read_csv(train_file) #TODO: CHANGE PATH to train file
	print("train file:", train_file)

	test_real = pd.read_csv('/Users/SWiegreffe/Desktop/mimicdata/test_full.csv') #TODO: CHANGE PATH

	for row in train_full.itertuples():
		for label in set(str(row[4]).split(';')): #remove duplicates*
			counts[label] += 1

	for row in test_real.itertuples():
		true_labels = []
		for label in set(str(row[4]).split(';')):
			true_labels.append(label)
			vocab.add(label)
		true_codes.append(true_labels)
		hadm_id = str(row[2])
		hadm_ids.append(hadm_id)

	with open('/Users/SWiegreffe/Desktop/mimicdata/preds_test.psv', 'r') as f: 	#TODO: CHANGE PATH #dev = f.read().splitlines()
		for line in f:
			els = line.strip().split('|')
			new_hadm_ids.append(els[0])
			pred_codes.append(list(set(els[1:])))

	assert hadm_ids == new_hadm_ids #assert IDs align
	assert len(true_codes) == len(hadm_ids) == len(pred_codes) #and have retrieved all records

	print("Number of distinct codes in the true test labels:", len(vocab))

	#convert to one-hot numpy encoding
	ind2c = {i:c for i,c in enumerate(vocab)}
	c2ind = {c:i for i,c in ind2c.items()}

	assert len(ind2c) == len(c2ind) == len(vocab)

	#remap to one-hot vectors
	true_one_hots = np.zeros((len(true_codes), len(vocab)))
	pred_one_hots = np.zeros(true_one_hots.shape)

	i = 0
	for a in true_codes:
		for el in a:
			true_one_hots[i,c2ind[el]] = 1
		assert len(np.where(true_one_hots[i] != 0)[0]) == len(true_codes[i])
		i += 1

	i = 0
	for b in pred_codes:
		for el in b: 
			if el in c2ind: #**We are only looking at true predicted codes, not false positives**
				pred_one_hots[i, c2ind[el]] = 1
		#assert len(np.where(pred_one_hots[i] != 0)[0]) == len(pred_codes[i]) #does not hold here b/c some keys are missing**
		i += 1

#-------------TESTING
	# print(true_codes[0])
	# print(pred_codes[0])
	# print(np.where(true_one_hots[0] != 0))
	# print(np.where(pred_one_hots[0] != 0))

	# for val in np.where(true_one_hots[0] != 0)[0]:
	# 	print(ind2c[val])

	# print("predicted codes:")
	# print(c2ind['34.04'])
	# print(c2ind['427.5'])
	# print(c2ind['96.71'])

	# print("true codes:")
	# print(c2ind['860.4'])
	# print(c2ind['868.03'])
	# print(c2ind['E957.1'])
	# print(c2ind['854.05'])

	#get the intersection of the numpy arrays
	true_positives = np.logical_and(pred_one_hots, true_one_hots).sum(axis=0) #for each code, the number of times a label occurred and was correctly predicted

	all_positives = true_one_hots.sum(axis=0)

	pred_accs = (true_positives/all_positives.astype(float))*100	#np.shape of (4075,)

	cnts = np.zeros(len(vocab))
	for i in range(len(vocab)):
		cnts[i] = counts[ind2c[i]] #get count

	#assert np.array_equal(cnts, all_positives) #for test only**

	#now, plot!
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = pred_accs
	y = cnts
	ax.scatter(x, y)
	plt.show()

if __name__ == "__main__":
	main()

