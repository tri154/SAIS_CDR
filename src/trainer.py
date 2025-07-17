import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import math
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg, model, train_set, tester):
        self.cfg = cfg
        self.model = model
        self.train_set = train_set
        self.tester = tester
        self.cur_epoch = 0

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
        opt = AdamW(grouped_lrs, eps=self.cfg.adam_epsilon)

        num_updates = math.ceil(math.ceil(len(self.train_set) / self.cfg.batch_size) / self.cfg.update_freq) * self.cfg.num_epoch
        num_warmups = int(num_updates * self.cfg.warmup_ratio)
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)

        return opt, sched

    def prepare_batch(self, batch_size):
        inputs = self.train_set
        num_batch = math.ceil(len(inputs) / batch_size)
        device = self.cfg.device

        for idx_batch in range(num_batch):
            indicies = (idx_batch * batch_size, (idx_batch + 1) * batch_size)
            batch_inputs = inputs[indicies[0]:indicies[1]]

            batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
            batch_titles = list()
            batch_start_mpos = list()
            batch_epair_rels = list()
            batch_sent_pos = list()
            num_sent_per_doc = list()
            batch_mpos2sid = list()
            batch_mentions_link = list()
            batch_teacher_logits = list()
  
            for doc_input in batch_inputs:
                batch_titles.append(doc_input['doc_title'])
                batch_token_seqs.append(doc_input['doc_tokens'])
                batch_start_mpos.append(doc_input['doc_start_mpos'])
                batch_sent_pos.append(doc_input['doc_sent_pos'])
                batch_mpos2sid.append(doc_input['doc_mpos2sid'])
                batch_mentions_link.append(doc_input['doc_mentions_link'])
                num_sent_per_doc.append(len(doc_input['doc_sent_pos']))

                if 'teacher_logits' in doc_input:
                    batch_teacher_logits.append(doc_input['teacher_logits'])

                doc_seqs_len = doc_input['doc_tokens'].shape[0]
                batch_token_masks.append(torch.ones(doc_seqs_len))
                doc_tokens_types = torch.zeros(doc_seqs_len)

                for sid in range(len(doc_input['doc_sent_pos'])):
                    start, end = doc_input['doc_sent_pos'][sid][0], doc_input['doc_sent_pos'][sid][1]
                    doc_tokens_types[start:end] = sid % 2
                batch_token_types.append(doc_tokens_types)

                batch_epair_rels.append(doc_input['doc_epair_rels'])

            batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long()
            batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float()
            batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long()

            max_m_n_p_b = max([len(mention_pos) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()])  # max mention number in batch.

            # num_entity_per_doc = torch.tensor([len(doc_start_mpos.values()) for doc_start_mpos in batch_start_mpos]) # Keep it on CPU.
            # batch_start_mpos = torch.stack([F.pad(torch.tensor(list(mention_pos)), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1) for doc_start_mpos in batch_start_mpos for mention_pos in doc_start_mpos.values()]).to(self.cfg.device)

            temp = []
            num_entity_per_doc = []
            num_mention_per_doc = [0 for _ in range(self.cfg.batch_size)]
            num_mention_per_entity = []
            for did, doc_start_mpos in enumerate(batch_start_mpos):
                num_entity_per_doc.append(len(doc_start_mpos.values()))
                for eid in sorted(doc_start_mpos): # Keep entity order.
                    mention_pos = doc_start_mpos[eid]
                    num_mention = len(mention_pos)
                    num_mention_per_entity.append(num_mention)
                    num_mention_per_doc[did] += num_mention
                    temp.append(F.pad(torch.tensor(sorted(list(mention_pos))), pad=(0, max_m_n_p_b - len(mention_pos)), value=-1)) # Keep mention order

            batch_start_mpos = torch.stack(temp) #expecting: [sum entity, max_mention]
            num_entity_per_doc = torch.tensor(num_entity_per_doc)
            num_mention_per_doc = torch.tensor(num_mention_per_doc)
            num_sent_per_doc = torch.tensor(num_sent_per_doc)
            num_mention_per_entity = torch.tensor(num_mention_per_entity)
            batch_mpos2sid = torch.cat(batch_mpos2sid, dim=0)
            batch_teacher_logits = torch.cat(batch_teacher_logits, dim=0) if len(batch_teacher_logits) != 0 else None

            num_mentlink_per_doc = torch.tensor([ts.shape[-1] for ts in batch_mentions_link])
            batch_mentions_link = torch.cat(batch_mentions_link, dim=-1)

            yield { 'indices': indicies,
                    'batch_titles': np.array(batch_titles),
                    'batch_epair_rels': batch_epair_rels,
                    'batch_sent_pos': batch_sent_pos,
                    'num_sent_per_doc': num_sent_per_doc.cpu(), 
                    'num_entity_per_doc': num_entity_per_doc.cpu(),
                    'num_mention_per_doc': num_mention_per_doc.cpu(),
                    'num_mentlink_per_doc': num_mentlink_per_doc.cpu(),
                    'num_mention_per_entity': num_mention_per_entity.to(device),
                    'batch_token_seqs': batch_token_seqs.to(device),
                    'batch_token_masks': batch_token_masks.to(device),
                    'batch_token_types': batch_token_types.to(device),
                    'batch_start_mpos': batch_start_mpos.to(device),
                    'batch_mpos2sid': batch_mpos2sid.to(device),
                    'batch_mentions_link': batch_mentions_link.to(device),
                    'batch_teacher_logits': batch_teacher_logits.to(device) if batch_teacher_logits is not None else None,
                    }

    def debug(self):
        for  batch_input in self.prepare_batch(self.cfg.batch_size):
            loss = self.model(batch_input, is_training=True)
            print(loss)
            input("Stop")

    def PSD_add_logits(self, batch_logits, indicies):
        for did, doc_idx in enumerate(range(indicies[0], indicies[1])):
            self.train_set[doc_idx]['teacher_logits'] = batch_logits[did]

    def train_one_epoch(self, current_epoch, batch_size, no_tqdm=False):
        self.model.train()
        self.opt.zero_grad()

        np.random.shuffle(self.train_set)

        num_batch = math.ceil(len(self.train_set) / batch_size)


        for idx_batch, batch_input in enumerate(tqdm(self.prepare_batch(batch_size), total=num_batch, disable=no_tqdm)):
            batch_loss, batch_logits = self.model(batch_input, current_epoch=current_epoch, is_training=True)
            self.PSD_add_logits(batch_logits, batch_input['indices'])
            (batch_loss / self.cfg.update_freq).backward()

            if idx_batch % self.cfg.update_freq == 0 or idx_batch == num_batch - 1:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()


    def train(self, num_epoches, batch_size, train_set=None, no_tqdm=False):
        if train_set is not None:
            self.train_set = train_set

        self.best_f1 = 0
        for idx_epoch in range(num_epoches):
            print(f'epoch {idx_epoch}/{num_epoches} ' + '=' * 100)
            self.train_one_epoch(idx_epoch, batch_size, no_tqdm=no_tqdm)
            presicion, recall, f1 = self.tester.test(self.model, dataset='dev')
            print(f"epoch: {idx_epoch}, P={presicion}, R={recall}, F1={f1}.")

            if f1 >= self.best_f1:
                self.best_f1 = f1
                torch.save(self.model.state_dict(), self.cfg.save_path)
            self.cur_epoch += 1

        self.model.load_state_dict(torch.load(self.cfg.save_path, map_location=self.cfg.device))
        precision, recall, self.f1 = self.tester.test(self.model, dataset='test')
        print(f"Test result: P={precision}, R={recall}, F1={f1}")
        return self.best_f1
