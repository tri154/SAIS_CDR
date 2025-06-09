import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--seed', type=int, default=2004)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--transformer', type=str, default='bert-base-cased')

    args = parser.parse_args()
    
    return args


class Config:

    def __init__(self, ):
        args = parse_args()
        self.__dict__.update(vars(args))

        # process other configurations.


        self.dir_curr = os.getcwd()
        self.dir_data = os.path.join(self.dir_curr, '../data')
        self.dir_dataset = os.path.join(self.dir_data, self.dataset)
        self.dir_dataset_ori = os.path.join(self.dir_dataset, 'original')
        self.dir_dataset_pro = os.path.join(self.dir_dataset, 'processed')


        self.train_set = 'train_annotated'
        self.dev_set = 'dev'
        self.test_set = 'test'

        self.file_corpus_train = os.path.join(self.dir_dataset_ori, f'{self.train_set}.json')
        self.file_corpus_dev = os.path.join(self.dir_dataset_ori, f'{self.dev_set}.json')
        self.file_corpus_test = os.path.join(self.dir_dataset_ori, f'{self.test_set}.json')

        self.file_corpuses = {self.train_set: self.file_corpus_train, self.dev_set: self.file_corpus_dev, self.test_set: self.file_corpus_test}

        self.data_ner2id = json.load(open(os.path.join(self.dir_dataset_ori, 'ner2id.json'), 'r'))
        self.data_rel2id = json.load(open(os.path.join(self.dir_dataset_ori, 'rel2id.json'), 'r'))
        self.data_id2ner = {v: k for k, v in self.data_ner2id.items()}
        self.data_id2rel = {v: k for k, v in self.data_rel2id.items()}

        self.id_rel_thre = self.data_rel2id['Na']
        self.num_ner = len(self.data_ner2id)
        self.num_rel = len(self.data_rel2id)

        if self.dataset == 'cdr':
            self.data_ner2word = {'CHEM': 'chemical', 'DISE': 'disease'}
            


        



        
