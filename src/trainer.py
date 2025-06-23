import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import math
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg, model, train_set=None, tester=None):
        self.cfg = cfg
        self.model = model
        self.train_set = train_set
        self.tester = tester

        self.opt, self.sched = self.prepare_optimizer_scheduler()


    # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
    #             'doc_title': doc_title,
    #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
    #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.


    def prepare_optimizer_scheduler(self):
        grouped_params = defaultdict(list)
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                grouped_params['pretrained_lr'].append(param)
            else:
                grouped_params['new_lr'].append(param)

        grouped_lrs = [{'params': grouped_params[group], 'lr': lr} for group, lr in zip(['pretrained_lr', 'new_lr'], [self.cfg.pretrained_lr, self.cfg.new_lr])]
        opt = AdamW(grouped_lrs)

        num_updates = math.ceil(math.ceil(len(self.train_set) / self.cfg.batch_size) / self.cfg.update_freq) * self.cfg.num_epoch
        num_warmups = int(num_updates * self.cfg.warmup_ratio)
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)

        return opt, sched
               
    def prepare_batch(self, batch_size):
        inputs = self.train_set
        num_batch = math.ceil(len(inputs) / batch_size)

        for idx_batch in range(num_batch):
            batch_inputs = inputs[idx_batch * batch_size:(idx_batch + 1) * batch_size]

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
        # for idx_batch, batch_input in enumerate(self.prepare_batch(self.cfg.batch_size)):
        #     self.model(batch_input, is_training=True)
        #     input("Stop")
                
        self.tester.test(self.model, dataset='dev')

    def train_one_epoch(self, batch_size):
        self.model.train()
        self.opt.zero_grad()

        np.random.shuffle(self.train_set)

        num_batch = math.ceil(len(self.train_set) / batch_size)

        for idx_batch, batch_input in enumerate(self.prepare_batch(batch_size)):
            batch_loss = self.model(batch_input, is_training=True)
            (batch_loss / self.cfg.update_freq).backward()

            if idx_batch % self.cfg.update_freq == 0 or idx_batch == num_batch - 1:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()


    def train(self, num_epoches, batch_size, train_set=None):
        if train_set is not None:
            self.train_set = train_set

        best_f1, best_epoch = 0, 0
        for idx_epoch in tqdm(range(num_epoches)):

            self.train_one_epoch(batch_size)
            presicion, recall, f1 = self.tester.test(self.model, dataset='dev')
            print(f"epoch: {idx_epoch}, P={presicion}, R={recall}, F1={f1}.")

            if f1 >= best_f1:
                best_f1, best_epoch = f1, idx_epoch
                torch.save(self.model.state_dict(), self.save_path)

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        precision, recall, f1 = self.tester.test(self.model, dataset='test')
