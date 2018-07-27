import matplotlib
matplotlib.use('Agg')

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
# np.set_printoptions(threshold=np.nan, suppress=True)

def main():

	get_code_rarities()


def get_code_rarities():

	true_codes = []
	hadm_ids = []
	new_hadm_ids_caml = []
	new_hadm_ids_meca = []
	pred_codes_caml = []
	pred_codes_meca = []

	#first calculate the top k
	counts = Counter()

	#define a vocab of the codes we have as true labels
	vocab = set()

	#get counts from training set, get true occurrences from test set
	train_file = '/Users/SWiegreffe/Desktop/mimicdata/test_full.csv'
	#train_file = '/data/swiegreffe6/NEW_MIMIC/mimic3/train_full.csv'
	train_full = pd.read_csv(train_file) #TODO: CHANGE PATH to train file
	print("train file:", train_file)

	test_real = pd.read_csv('/Users/SWiegreffe/Desktop/mimicdata/test_full.csv')
	#test_real = pd.read_csv('/data/swiegreffe6/NEW_MIMIC/mimic3/test_full.csv')

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

	with open('/Users/SWiegreffe/Desktop/mimicdata/preds_test.psv', 'r') as f: 
	#with open('/data/swiegreffe6/caml_models/conv_attn_Jul_11_02:37/preds_test.psv', 'r') as f:
		for line in f:
			els = line.strip().split('|')
			new_hadm_ids_caml.append(els[0])
			pred_codes_caml.append(list(set(els[1:])))

	with open('/Users/SWiegreffe/Desktop/mimicdata/preds_test.psv', 'r') as f:
	# with open('/data/swiegreffe6/NEW_MIMIC/saved_models/keep/conv_attn_plus_GRAM_Jul_12_19:06/preds_test.psv', 'r') as f:
		for line in f:
			els = line.strip().split('|')
			new_hadm_ids_meca.append(els[0])
			pred_codes_meca.append(list(set(els[1:])))

	assert hadm_ids == new_hadm_ids_caml == new_hadm_ids_meca #assert IDs align
	assert len(true_codes) == len(hadm_ids) == len(pred_codes_caml) == len(pred_codes_meca) #and have retrieved all records

	print("Number of distinct codes in the true test labels:", len(vocab))

	#convert to one-hot numpy encoding
	ind2c = {i:c for i,c in enumerate(vocab)}
	c2ind = {c:i for i,c in ind2c.items()}

	assert len(ind2c) == len(c2ind) == len(vocab)

	#remap to one-hot vectors
	true_one_hots = np.zeros((len(true_codes), len(vocab)))
	pred_one_hots_caml = np.zeros(true_one_hots.shape)
	pred_one_hots_meca = np.zeros(true_one_hots.shape)

	i = 0
	for a in true_codes:
		for el in a:
			true_one_hots[i,c2ind[el]] = 1
		assert len(np.where(true_one_hots[i] != 0)[0]) == len(true_codes[i])
		i += 1

	i = 0
	for b in pred_codes_caml:
		for el in b: 
			if el in c2ind: #**We are only looking at true predicted codes, not false positives**
				pred_one_hots_caml[i, c2ind[el]] = 1
		#assert len(np.where(pred_one_hots[i] != 0)[0]) == len(pred_codes[i]) #does not hold here b/c some keys are missing**
		i += 1

	i = 0
	for c in pred_codes_meca:
		for el in c: 
			if el in c2ind: #**We are only looking at true predicted codes, not false positives**
				pred_one_hots_meca[i, c2ind[el]] = 1
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

#---------------------------- VALUES

	#get the intersection of the numpy arrays
	true_positives_caml = np.logical_and(pred_one_hots_caml, true_one_hots).sum(axis=0) #for each code, the number of times a label occurred and was correctly predicted
	false_positives_caml = (pred_one_hots_caml > true_one_hots).astype(int).sum(axis=0)
	true_negatives_caml = np.logical_and(np.logical_not(pred_one_hots_caml), np.logical_not(true_one_hots)).sum(axis=0)
	false_negatives_caml = (true_one_hots > pred_one_hots_caml).astype(int).sum(axis=0)

	predicted_condition_positive_caml = pred_one_hots_caml.sum(axis=0)
	assert np.all(predicted_condition_positive_caml == true_positives_caml + false_positives_caml)
	predicted_condition_negative_caml = np.logical_not(pred_one_hots_caml).sum(axis=0)
	assert np.all(predicted_condition_negative_caml == true_negatives_caml + false_negatives_caml)

	true_positives_meca = np.logical_and(pred_one_hots_meca, true_one_hots).sum(axis=0)
	false_positives_meca = (pred_one_hots_meca > true_one_hots).astype(int).sum(axis=0)
	true_negatives_meca = np.logical_and(np.logical_not(pred_one_hots_meca), np.logical_not(true_one_hots)).sum(axis=0)
	false_negatives_meca = (true_one_hots > pred_one_hots_meca).astype(int).sum(axis=0)

	predicted_condition_positive_meca = pred_one_hots_meca.sum(axis=0)
	assert np.all(predicted_condition_positive_meca == true_positives_meca + false_positives_meca)
	predicted_condition_negative_meca = np.logical_not(pred_one_hots_meca).sum(axis=0)
	assert np.all(predicted_condition_negative_meca == true_negatives_meca + false_negatives_meca)

	condition_positive = true_one_hots.sum(axis=0)
	assert np.all(condition_positive == true_positives_caml + false_negatives_caml)
	assert np.all(condition_positive == true_positives_meca + false_negatives_meca)
	condition_negative = np.logical_not(true_one_hots).sum(axis=0)
	assert np.all(condition_negative == true_negatives_caml + false_positives_caml)
	assert np.all(condition_negative == true_negatives_meca + false_positives_meca)

	all_cases = true_one_hots.shape[0]

	assert np.all(np.array([all_cases]*true_one_hots.shape[1]) == condition_positive + condition_negative)

#---------------------------- METRICS

	tpr_caml = (true_positives_caml/condition_positive.astype(float))*100	#true positive rate (or recall). np.shape of (4075,)
	precision_caml = (true_positives_caml/predicted_condition_positive_caml.astype(float))*100
	fpr_caml = (false_positives_caml/condition_negative.astype(float))*100 #false positive rate
	acc_caml = ((true_positives_caml + true_negatives_caml)/float(all_cases))*100	
	#f1_caml = 

	tpr_meca = (true_positives_meca/condition_positive.astype(float))*100	#true positive rate (or recall). np.shape of (4075,)
	precision_meca = (true_positives_meca/predicted_condition_positive_meca.astype(float))*100
	fpr_meca = (false_positives_meca/condition_negative.astype(float))*100 #false positive rate
	acc_meca = ((true_positives_meca + true_negatives_meca)/float(all_cases))*100	

#---------------------------- COUNTS OF LABEL OCCURRENCES FROM TRAINING DATA

	cnts = np.zeros(len(vocab))
	for i in range(len(vocab)):
		cnts[i] = counts[ind2c[i]]

	#assert np.array_equal(cnts, all_positives) #for test only**

#---------------------------- PLOTS

	#Plot One- TPR/Recall
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = cnts
	ax.scatter(x, tpr_caml, c='b', label='CAML')
	ax.scatter(x, tpr_meca, c='r', label='MECA')
	fig.suptitle('Code Occurrences in Training Data vs Recall on Test')
	plt.xlabel('Number of Occurrences of Code in Training Data')
	plt.ylabel('True Positive Rate/Recall on Test Data')
	plt.legend(loc='upper right')
	plt.savefig('/nethome/swiegreffe6/fig_recall.png')
	#plt.show()

	#Plot Two- FPR
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = cnts
	ax.scatter(x, fpr_caml, c='b', label='CAML')
	ax.scatter(x, fpr_meca, c='r', label='MECA')
	fig.suptitle('Code Occurrences in Training Data vs False Positives Rate on Test')
	plt.xlabel('Number of Occurrences of Code in Training Data')
	plt.ylabel('False Positive Rate on Test Data')
	plt.legend(loc='upper right')
	plt.savefig('/nethome/swiegreffe6/fig_fpr.png')
	#plt.show()

	#Plot Three- Precision
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = cnts
	ax.scatter(x, precision_caml, c='b', label='CAML')
	ax.scatter(x, precision_meca, c='r', label='MECA')
	fig.suptitle('Code Occurrences in Training Data vs Precision on Test')
	plt.xlabel('Number of Occurrences of Code in Training Data')
	plt.ylabel('Precision on Test Data')
	plt.legend(loc='upper right')
	plt.savefig('/nethome/swiegreffe6/fig_precision.png')
	#plt.show()

	#Plot Four- Accuracy
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = cnts
	ax.scatter(x, acc_caml, c='b', label='CAML')
	ax.scatter(x, acc_meca, c='r', label='MECA')
	fig.suptitle('Code Occurrences in Training Data vs Accuracy on Test')
	plt.xlabel('Number of Occurrences of Code in Training Data')
	plt.ylabel('Accuracy on Test')
	plt.legend(loc='upper right')
	plt.savefig('/nethome/swiegreffe6/fig_acc.png')
	#plt.show()

if __name__ == "__main__":
	main()

