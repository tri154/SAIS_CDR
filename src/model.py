import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from itertools import permutations


class Transformer(nn.Module):
    
    def __init__(self, cfg):
        super(Transformer, self).__init__()

        self.cfg = cfg
        config = AutoConfig.from_pretrained(cfg.transformer, num_labels=cfg.num_rel)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.transformer)
        self.transformer = AutoModel.from_pretrained(cfg.transformer, config=config)

        config.cls_token_id = self.tokenizer.cls_token_id
        config.sep_token_id = self.tokenizer.sep_token_id
    
        self.max_num_tokens = 512
        
        self.start_token_len, self.end_token_len = 1, 1
        self.start_token_ids = torch.Tensor([self.transformer.config.cls_token_id]).to(cfg.device)
        self.end_token_ids = torch.Tensor([self.transformer.config.sep_token_id]).to(cfg.device)
        
        
    def forward(self, batch_token_seqs, batch_token_masks, batch_token_types):
        
        if 'roberta' in self.transformer.config._name_or_path:
            batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, batch_num_tokens = batch_token_seqs.size()
        
        if batch_num_tokens <= self.max_num_tokens:
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            
        else:
            new_token_seqs, new_token_masks, new_token_types, new_token_segs = [], [], [], []
            real_num_tokens = batch_token_masks.sum(1).int().tolist()
            for doc_id, real_num_token in enumerate(real_num_tokens): # for each docs.
                if real_num_token <= self.max_num_tokens:
                    new_token_seqs.append(batch_token_seqs[doc_id, :self.max_num_tokens])
                    new_token_masks.append(batch_token_masks[doc_id, :self.max_num_tokens])
                    new_token_types.append(batch_token_types[doc_id, :self.max_num_tokens])
                    new_token_segs.append(1)
                else:
                    new_token_seq1 = torch.cat([batch_token_seqs[doc_id, :self.max_num_tokens - self.end_token_len], self.end_token_ids], dim=-1)
                    new_token_mask1 = batch_token_masks[doc_id, :self.max_num_tokens]
                    new_token_type1 = torch.cat([batch_token_types[doc_id, :self.max_num_tokens - self.end_token_len], batch_token_types[doc_id, self.max_num_tokens - self.end_token_len - 1].repeat(self.end_token_len)], dim=-1)
                    
                    new_token_seq2 = torch.cat([self.start_token_ids, batch_token_seqs[doc_id, real_num_token - self.max_num_tokens + self.start_token_len : real_num_token]], dim=-1)                    
                    new_token_mask2 = batch_token_masks[doc_id, real_num_token - self.max_num_tokens : real_num_token]
                    new_token_type2 = torch.cat([batch_token_types[doc_id, real_num_token - self.max_num_tokens + self.start_token_len].repeat(self.start_token_len), batch_token_types[doc_id, real_num_token - self.max_num_tokens + self.start_token_len : real_num_token]], dim=-1)
                    
                    new_token_seqs.extend([new_token_seq1, new_token_seq2])
                    new_token_masks.extend([new_token_mask1, new_token_mask2])
                    new_token_types.extend([new_token_type1, new_token_type2])
                    new_token_segs.append(2) # drop middle part if the token length exceeds 2 * 512, or may be the maximum is smaller than that.
                    
            batch_token_seqs, batch_token_masks, batch_token_types = torch.stack(new_token_seqs, dim=0).long(), torch.stack(new_token_masks, dim=0).float(), torch.stack(new_token_types, dim=0).long()
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            
            seg_id, new_token_embs, new_token_atts = 0, [], []
            for (new_token_seq, real_num_token) in zip(new_token_segs, real_num_tokens):
                if new_token_seq == 1:
                    new_token_emb = F.pad(batch_token_embs[seg_id], (0, 0, 0, batch_num_tokens - self.max_num_tokens))
                    new_token_att = F.pad(batch_token_atts[seg_id], (0, batch_num_tokens - self.max_num_tokens, 0, batch_num_tokens - self.max_num_tokens))
                    new_token_embs.append(new_token_emb)
                    new_token_atts.append(new_token_att)
                    
                elif new_token_seq == 2:
                    valid_num_token1 = self.max_num_tokens - self.end_token_len
                    new_token_emb1 = F.pad(batch_token_embs[seg_id][:valid_num_token1], (0, 0, 0, batch_num_tokens - valid_num_token1))
                    new_token_mask1 = F.pad(batch_token_masks[seg_id][:valid_num_token1], (0, batch_num_tokens - valid_num_token1))
                    new_token_att1 = F.pad(batch_token_atts[seg_id][:, :valid_num_token1, :valid_num_token1], (0, batch_num_tokens - valid_num_token1, 0, batch_num_tokens - valid_num_token1))
                    
                    valid_num_token2 = real_num_token - self.max_num_tokens
                    new_token_emb2 = F.pad(batch_token_embs[seg_id + 1][self.start_token_len:], (0, 0, valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    new_token_mask2 = F.pad(batch_token_masks[seg_id + 1][self.start_token_len:], (valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    new_token_att2 = F.pad(batch_token_atts[seg_id + 1][:, self.start_token_len:, self.start_token_len:], (valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token, valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    
                    new_token_mask = new_token_mask1 + new_token_mask2 + self.cfg.small_positive
                    new_token_emb = (new_token_emb1 + new_token_emb2) / new_token_mask.unsqueeze(-1)
                    new_token_att = (new_token_att1 + new_token_att2)
                    new_token_att /= (new_token_att.sum(-1, keepdim=True) + self.cfg.small_positive)
                    new_token_embs.append(new_token_emb)
                    new_token_atts.append(new_token_att)
                    
                seg_id += new_token_seq
            batch_token_embs, batch_token_atts = torch.stack(new_token_embs, dim=0), torch.stack(new_token_atts, dim=0)
            
        return batch_token_embs, batch_token_atts
    
    
    
class Model(nn.Module):
    
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        if cfg.transformer == 'bert-base-cased':
            self.hidden_dim = 768
        #NOTE: change if transformer changes.
        self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_t = nn.Linear(self.hidden_dim, self.hidden_dim)
    
        self.RE_predictor_module = nn.Linear(self.hidden_dim * self.cfg.bilinear_block_size, self.cfg.num_rel)

        # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
        #             'doc_title': doc_title,
        #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
        #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.


    def compute_entity_embs(self, batch_token_embs, batch_start_mpos, num_entity_per_doc):
        # batch_start_mpos: (sum_entity, max_mention_n) pad -1
        batch_did = torch.arange(len(batch_token_embs)).repeat_interleave(num_entity_per_doc).unsqueeze(-1)
        batch_token_embs = F.pad(batch_token_embs, (0, 0, 0, 1), value=self.cfg.small_negative)
        batch_entity_embs = batch_token_embs[batch_did, batch_start_mpos].logsumexp(dim=-2)

        return batch_entity_embs 
        

    def forward(self, batch_input):
        batch_token_seqs = batch_input['batch_token_seqs']
        batch_token_masks = batch_input['batch_token_masks']
        batch_token_types = batch_input['batch_token_types']
        batch_start_mpos = batch_input['batch_start_mpos']
        batch_epair_rels = batch_input['batch_epair_rels']
        num_entity_per_doc = batch_input['num_entity_per_doc']

        batch_token_embs, batch_token_atts = self.transformer(batch_token_seqs, batch_token_masks, batch_token_types)
        batch_entity_embs = self.compute_entity_embs(batch_token_embs, batch_start_mpos, num_entity_per_doc)


        start_entity_pos = torch.cumsum(torch.cat([torch.tensor([0]), num_entity_per_doc]), dim=0)

        head_entity_pairs = list()
        tail_entity_pairs = list()
        batch_labels = list()

        for did in range(len(start_entity_pos) - 1):
            doc_epair_rels = batch_epair_rels[did]
            offset = int(start_entity_pos[did])
            for eid_h, eid_t in permutations(np.arange(offset, int(start_entity_pos[did + 1])), 2):
                pair_labels = torch.zeros(self.cfg.num_rel)
                for r in doc_epair_rels[(eid_h - offset, eid_t - offset)]:
                    pair_labels[self.cfg.data_rel2id[r]] = 1
                batch_labels.append(pair_labels)
                head_entity_pairs.append(eid_h)
                tail_entity_pairs.append(eid_t)

        head_entity_pairs = torch.tensor(head_entity_pairs).to(self.cfg.device)
        tail_entity_pairs = torch.tensor(tail_entity_pairs).to(self.cfg.device)

        head_entity_embs = batch_entity_embs[head_entity_pairs]
        tail_entity_embs = batch_entity_embs[tail_entity_pairs]

        head_entity_rep = torch.tanh(self.W_h(head_entity_embs))
        tail_entity_rep = torch.tanh(self.W_t(tail_entity_embs))

        head_entity_rep = head_entity_rep.view(-1, self.hidden_dim // self.cfg.bilinear_block_size, self.cfg.bilinear_block_size)
        tail_entity_rep = tail_entity_rep.view(-1, self.hidden_dim // self.cfg.bilinear_block_size, self.cfg.bilinear_block_size)
        batch_RE_reps = (head_entity_rep.unsqueeze(3) * tail_entity_rep.unsqueeze(2)).view(-1, self.hidden_dim * self.cfg.bilinear_block_size)
        batch_RE_reps = self.RE_predictor_module(batch_RE_reps)

        print(batch_RE_reps.shape)
        # TODO: don't have lablels to compute loss.

       

        # print(head_entity_embs.shape) # (50, 768)
        # print(tail_entity_embs.shape) # (50, 768)
        # print(batch_token_atts.shape) # (4, 12, 370, 370) doc, n_head, max_length
        # print(batch_token_embs.shape) # (b, l, 768)
        # print(batch_token_seqs.shape) b, N_max, 768

        
