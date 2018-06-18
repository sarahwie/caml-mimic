#this script runs the CTakes parser on a sample of MIMIC data, that has been formatted via James' script to be in the correct format for CAML'''
import csv
import os

main_data_dir = '/data/mimicdata/mimic3'
write_dir = '/data/mimicdata/mimic3/patient_notes/'

def main():

	print("processing TRAIN")
	with open(os.path.join(main_data_dir,'train_full.csv'), 'rb') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			f = open(os.path.join(write_dir, 'train', str(line[0])+'_'+str(line[1])+'.txt'), 'w')
			f.write(line[2])
			f.close()

	print("processing TEST")
	with open(os.path.join(main_data_dir,'test_full.csv'), 'rb') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			f = open(os.path.join(write_dir, 'test', str(line[0])+'_'+str(line[1])+'.txt'), 'w')
			f.write(line[2])
			f.close()

	print("processing DEV")
	with open(os.path.join(main_data_dir,'dev_full.csv'), 'rb') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			f = open(os.path.join(write_dir, 'dev', str(line[0])+'_'+str(line[1])+'.txt'), 'w')
			f.write(line[2])
			f.close()

if __name__ == '__main__':
	main()
