import argparse
import os
import json
import random
import torch
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)

    parser.add_argument('--save_path', type=str, help="file path to save model.", default='best.pt')
    parser.add_argument('--log_path', type=str, default='log.txt')
    parser.add_argument('--seed', type=int, default=2004)
    parser.add_argument('--tqdm', action='store_true')

    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=1, help="Gradient accumulation.") 
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help="Warmup.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--new_lr", type=float, default=1e-4)
    parser.add_argument("--pretrained_lr", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)


    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--transformer', type=str, default='bert-base-cased')
    parser.add_argument('--seq_process_type', type=str, default='sd', help='choose type of sequence prcess, sd: sliding windows, o: original (as SAIS)')

    parser.add_argument('--type_dim', type=int, default=20)
    parser.add_argument('--graph_layers', type=int, default=2, help='min = 2')

    parser.add_argument('--use_psd', type=str, default='True')
    parser.add_argument('--lower_temp', type=float, default=2.0)
    parser.add_argument('--upper_temp', type=float, default=20.0)
    parser.add_argument('--loss_tradeoff', type=float, default=1.0)

    parser.add_argument('--use_sc', type=str, default='True')
    parser.add_argument('--sc_temp', type=float, default=1.0)
    parser.add_argument('--sc_weight', type=float, default=1.0)

    args = parser.parse_args()
    
    return args


class Config:

    def __init__(self, args=None):
        args = parse_args() if args is None else args
        self.__dict__.update(vars(args))

        self.use_psd = True if self.use_psd == 'True' else False
        self.use_sc = True if self.use_sc == 'True' else False

        # process other configurations.
        self.set_seed()

        self.small_negative = -1e10
        self.small_positive = 1e-10

        self.marker_entity = '*'
        
        self.dir_curr = os.getcwd()
        self.dir_data = os.path.join(self.dir_curr, '../data')
        self.dir_dataset = os.path.join(self.dir_data, self.dataset)
        self.dir_dataset_ori = os.path.join(self.dir_dataset, 'original')
        self.dir_dataset_pro = os.path.join(self.dir_dataset, 'processed')


        self.train_set = 'train'
        self.dev_set = 'dev'
        self.test_set = 'test'

        self.file_corpus_train = os.path.join(self.dir_dataset_ori, f'{self.train_set}.json')
        self.file_corpus_dev = os.path.join(self.dir_dataset_ori, f'{self.dev_set}.json')
        self.file_corpus_test = os.path.join(self.dir_dataset_ori, f'{self.test_set}.json')

        self.file_corpuses = {self.train_set: self.file_corpus_train, self.dev_set: self.file_corpus_dev, self.test_set: self.file_corpus_test}

        self.file_processed_train = os.path.join(self.dir_dataset_pro, f'{self.train_set}_processed.pkl')
        self.file_processed_dev = os.path.join(self.dir_dataset_pro, f'{self.dev_set}_processed.pkl')
        self.file_processed_test = os.path.join(self.dir_dataset_pro, f'{self.test_set}_processed.pkl')

        self.files_processed = {self.train_set: self.file_processed_train, self.dev_set: self.file_processed_dev, self.test_set: self.file_processed_test}


        self.data_ner2id = json.load(open(os.path.join(self.dir_dataset_ori, 'ner2id.json'), 'r'))
        self.data_rel2id = json.load(open(os.path.join(self.dir_dataset_ori, 'rel2id.json'), 'r'))
        self.data_id2ner = {v: k for k, v in self.data_ner2id.items()}
        self.data_id2rel = {v: k for k, v in self.data_rel2id.items()}

        self.id_rel_thre = self.data_rel2id['Na']
        self.num_ner = len(self.data_ner2id)
        self.num_rel = len(self.data_rel2id)


        if self.dataset == 'cdr':
            self.logging("Dataset: CDR, use binary F1.")
            self.f1_type = 'binary'
            self.rel = 'CID' # main class to compute F1 binary.
            self.topk = 1 # for AT_pred

            self.data_ner2word = {'CHEM': 'chemical', 'DISE': 'disease'} # not really necessary
        elif self.dataset == 'gda':
            self.logging("Dataset: GDA, use binary F1.")
            self.f1_type = 'binary'
            self.rel = 'GDA' # main class to compute F1 binary.
            self.topk = 1 # for AT_pred

            self.data_ner2word = {'GENE': 'gene', 'DISE': 'disease'} # not really necessary
        elif self.dataset == 'biored':
            self.logging("Dataset: BioRED, use overall F1.")
            self.f1_type = 'overall'
            self.topk = 1 #for AT_pred
        else:
            raise Exception("Define topk, data_ner2word, f1_type")
            #set topk < 0 for multi classes.

            
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def logging(self, text):
        with open(self.log_path, 'a') as file:
            print(time.strftime("%Y %b %d %a, %H:%M:%S: ") + text, file=file, flush=True)
