import os 
import json
from dataset import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import dill as pk

class Preprocessing:
    def __init__(self, cfg):
        self.cfg = cfg
        is_done = len(os.listdir(cfg.dir_dataset_pro)) != 1

        if not is_done:
            config = AutoConfig.from_pretrained(cfg.transformer, num_labels=cfg.num_rel)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.transformer) # only load hidden states, that will output (batch_size, max_seq_length, 768) Bert case.
            transformer = AutoModel.from_pretrained(cfg.transformer, config=config)
    
            config.cls_token_id = self.tokenizer.cls_token_id
            config.sep_token_id = self.tokenizer.sep_token_id

            self.prepare()
        self.train_set, self.dev_set, self.test_set = map(lambda x: pk.load(open(self.cfg.file_processed[x], 'rb')), [self.cfg.train_set, self.cfg.dev_set, self.cfg.test_set])

    def prepare(self):
        self.train_set = None
        self.dev_set = None
        self.test_set = None

        for mode, corpus in self.cfg.file_corpuses.items():
            corpus = json.load(open(corpus, 'r'))
            corpus_data = self.__prepare_corpus(corpus) #a list of dict
            pk.dump(corpus_data, open(self.cfg.files_processed[mode], 'wb'), -1)

    def __prepare_corpus(self, corpus):
        corpus_data = [] # list of dict
        for did, doc in enumerate(corpus): #each doc
            doc_data = self.__prepare_doc(doc)
            corpus_data.append(doc_data)
        return corpus_data


    def __prepare_doc(self, doc):
        start_mpos, end_mpos = set(), set()
        for eid, entity in enumerate(doc['vertexSet']):
            for mid, mention in enumerate(entity):
                start_mpos.add((mention['sent_id'], mention['pos'][0]))
                end_mpos.add((mention['sent_id'], mention['pos'][1] - 1))

        doc_tokens = []
        for sid, sent in enumerate(doc['sents']): #each sentence
            sent_tokens = []
            for wid, word in enumerate(sent):
                tokens = self.tokenizer.tokenize(word)
                if (sid, wid) in start_mpos:
                    tokens = [self.cfg.marker_entity] + tokens
                if (sid, wid) in end_mpos:
                    tokens = tokens + [self.cfg.marker_entity]
                sent_tokens += tokens
            sent_tokens = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_tokens = self.tokenize.build_inputs_with_special_tokens(sent_tokens)
            doc_tokens += sent_tokens
        doc_tokens = torch.Tensor(doc_tokens).int()
        doc_title = doc['title']

        doc_data = {'doc_tokens': doc_tokens,
                    'doc_title': doc_title}

            
        return doc_data



                


            


