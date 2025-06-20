import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import math
from itertools import permutations
import torch.nn.functional as F

class Trainer:
    def __init__(self, cfg, model, train_set=None, tester=None):
        self.cfg = cfg
        self.model = model
        self.train_set = train_set
        self.tester = tester

        print()
        # TODO: optimizer

        # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
        #             'doc_title': doc_title,
        #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
        #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.
    def prepare_batch_train(self, batch_size):
        train_set = self.train_set
        num_batch = math.ceil(len(train_set) / batch_size)

        for idx_batch in range(num_batch):
            batch_inputs = train_set[idx_batch * batch_size:(idx_batch + 1) * batch_size]

            batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
            batch_titles = []
            batch_start_mpos = []
            batch_epair_rels = list()

            for doc_input in batch_inputs:
                batch_titles.append(doc_input['doc_title'])
                batch_token_seqs.append(doc_input['doc_tokens'])
                doc_start_mpos = doc_input['doc_start_mpos']
                batch_start_mpos.append(doc_start_mpos)

                doc_seqs_len = doc_input['doc_tokens'].shape[0]
                batch_token_masks.append(torch.ones(doc_seqs_len))
                doc_tokens_types = torch.zeros(doc_seqs_len)

                for sid in range(len(doc_input['doc_sent_pos'])):
                    start, end = doc_input['doc_sent_pos'][sid][0], doc_input['doc_sent_pos'][sid][1]
                    doc_tokens_types[start:end] = sid % 2
                batch_token_types.append(doc_tokens_types)

                batch_epair_rels.append(doc_input['doc_epair_rels'])

            batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long().to(self.cfg.device)
            batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float().to(self.cfg.device)
            batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long().to(self.cfg.device)



            max_m_n_p_b = max([len(mention_pos) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()])  # max mention number in batch.

            # num_entity_per_doc = torch.tensor([len(doc_start_mpos.values()) for doc_start_mpos in batch_start_mpos]) # Keep it on CPU.
            # batch_start_mpos = torch.stack([F.pad(torch.tensor(list(mention_pos)), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()]).to(self.cfg.device)

            temp = [] 
            num_entity_per_doc = []
            for doc_start_mpos in batch_start_mpos:
                num_entity_per_doc.append(len(doc_start_mpos.values()))
                for eid in sorted(doc_start_mpos):
                    mention_pos = doc_start_mpos[eid]
                    temp.append(F.pad(torch.tensor(list(mention_pos)), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1))

            batch_start_mpos = torch.stack(temp) #expecting: [sum entity, max_mention]
            num_entity_per_doc = torch.tensor(num_entity_per_doc) # keep in cpu.

            yield {'batch_titles': np.array(batch_titles),
                    'batch_token_seqs': batch_token_seqs,
                    'batch_token_masks': batch_token_masks,
                    'batch_token_types': batch_token_types,
                    'batch_start_mpos': batch_start_mpos,
                    'batch_epair_rels': batch_epair_rels,
                    'num_entity_per_doc': num_entity_per_doc}
            
    def debug(self):
        for idx_batch, batch_input in enumerate(self.prepare_batch_train(self.cfg.batch_size)):
            self.model(batch_input, is_training=True)
            input("Stop")
                

    def one_epoch_train(self, batch_size):
        self.model.train()
        self.optimizer.zero_grad()

        np.random.shuffle(self.train_set)

        for idx_batch, batch_input in enumerate(self.prepare_batch_train(batch_size)):
            batch_loss = self.model(batch_input, is_training=True)


    def train(self, num_epoches, batch_size, train_set=None):
        if train_set is not None:
            self.train_set = train_set

        best_score, best_epoch = 0, 0
        for idx_epoch in range(num_epoches):

            self.one_epoch_train(batch_size)
            score = self.tester.test_dev(self.model)

            if score >= best_score:
                best_score, best_epoch = score, idx_epoch
                torch.save(self.model.state_dict(), self.save_path)

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        self.tester.test(self.model)
                






        

        
