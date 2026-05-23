
#!/bin/bash

stage="Prepare"
dataset="cdr"
transformer="allenai/scibert_scivocab_uncased" # "roberta-large"
max_seq_length=1024


python3 prepare_cdr.py --stage=${stage} --dataset=${dataset} --transformer=${transformer} --max_seq_length=${max_seq_length}
