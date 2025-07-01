import os 
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import dill as pk
from collections import defaultdict

class Preprocessing:
    def __init__(self, cfg):
        self.cfg = cfg
        is_done = len(os.listdir(cfg.dir_dataset_pro)) != 1

        if not is_done:
            config = AutoConfig.from_pretrained(cfg.transformer, num_labels=cfg.num_rel)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.transformer) # only load hidden states, that will output (batch_size, max_seq_length, 768) Bert case.
            transformer = AutoModel.from_pretrained(cfg.transformer, config=config) #NOTE: remove if not use.
    
            config.cls_token_id = self.tokenizer.cls_token_id
            config.sep_token_id = self.tokenizer.sep_token_id

            self.prepare()
        self.train_set, self.dev_set, self.test_set = map(lambda x: pk.load(open(self.cfg.files_processed[x], 'rb')), [self.cfg.train_set, self.cfg.dev_set, self.cfg.test_set])

    def prepare(self):
        # Save a list of per doc data on each corpus.
        # Where each doc store:
        # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
        #             'doc_title': doc_title,
        #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
        #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.
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
        sid_pos2eid = defaultdict(set)
        for eid, entity in enumerate(doc['vertexSet']):
            for mid, mention in enumerate(entity):
                start_mpos.add((mention['sent_id'], mention['pos'][0]))
                sid_pos2eid[(mention['sent_id'], mention['pos'][0])].add(eid)
                end_mpos.add((mention['sent_id'], mention['pos'][1] - 1))

        doc_tokens = []
        doc_start_mpos = defaultdict(set)
        doc_sent_pos = dict()
        for sid, sent in enumerate(doc['sents']): #each sentence
            sent_tokens = []
            for wid, word in enumerate(sent):
                tokens = self.tokenizer.tokenize(word)
                if (sid, wid) in start_mpos:
                    tokens = [self.cfg.marker_entity] + tokens
                    for eid in sid_pos2eid[(sid, wid)]:
                        doc_start_mpos[eid].add(len(doc_tokens) + len(sent_tokens) + 1)
                if (sid, wid) in end_mpos:
                    tokens = tokens + [self.cfg.marker_entity]
                sent_tokens += tokens
            sent_tokens = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_tokens = self.tokenizer.build_inputs_with_special_tokens(sent_tokens)
            doc_sent_pos[sid] = (len(doc_tokens), len(doc_tokens) + len(sent_tokens))
            doc_tokens += sent_tokens
        doc_tokens = torch.Tensor(doc_tokens).int()
        doc_title = doc['title']

        doc_epair_rels = defaultdict(list)
        for rel in doc['labels']:
            h, t, r = rel['h'], rel['t'], rel['r']
            doc_epair_rels[(h, t)].append(r)

        doc_data = {'doc_tokens': doc_tokens,
                    'doc_title': doc_title,
                    'doc_start_mpos': doc_start_mpos,
                    'doc_sent_pos': doc_sent_pos,
                    'doc_epair_rels': doc_epair_rels}

            
        return doc_data

