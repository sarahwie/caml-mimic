Pipeline:

first, run parse_xmi.py for all 3 directories*
incorporate into James' preproc pipeline*

TODOS: update datasets.data_generator to make sure reading in the (expanded) concepts file as well

TODOs:
	- extract concepts from MIMIC XMI
	- create 'concepts' input file, pass in
	- create dictionary of concepts with indices
	- 

TODOS CAML EXTENSION: 
	- keep embedding dropout @ 0 for now during training**

ARGS-- 

data_path = ../../mimicdata/mimic3/train_full.csv
concepts_file = ../../mimicdata/mimic3/train_concepts.csv
vocab = ../../mimicdata/mimic3/vocab.csv
Y (size of label space) = full
model = conv_attn
n_epochs = 200
--filter-size 10
--num-filter-maps 50
--dropout 0.2
--patience 10
--lr 0.0001
--criterion prec_at_8
--embed-file ../../mimicdata/mimic3/processed_full.embed
--gpu