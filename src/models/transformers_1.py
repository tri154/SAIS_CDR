import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
        self.start_token_ids = torch.Tensor([self.tokenizer.cls_token_id]).to(cfg.device)
        self.end_token_ids = torch.Tensor([self.tokenizer.sep_token_id]).to(cfg.device)
        self.pad_token_ids = self.tokenizer.pad_token_id

    def forward_o(self, batch_token_seqs, batch_token_masks, batch_token_types):
        # o: original implement from SAIS/DocUnet/Atlop.
        # Make sure tokenized document sqeuence is smaller than 512 * 2.

        if 'roberta' in self.transformer.config._name_or_path:
            batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, batch_num_tokens = batch_token_seqs.size()
        assert batch_num_tokens <= self.max_num_tokens * 2, "Error, see method note."

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

            seg_id = 0
            new_token_embs = list()
            new_token_atts = list()
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
    


    def forward_sd_rewrite(self, batch_token_seqs, batch_token_masks, batch_token_types, stride=128):
        if 'roberta' in self.transformer.config._name_or_path:
            batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, max_doc_length = batch_token_seqs.shape

        if max_doc_length <= self.max_num_tokens:
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            return batch_token_embs, batch_token_atts

        num_token_per_doc = batch_token_masks.sum(1).int().tolist()

        token_seqs = list()
        token_masks = list()
        token_types = list()
        num_seg_per_doc = list()
        valids = list()

        
        for did, num_token in enumerate(num_token_per_doc):
            print(num_token)
            if num_token <= self.max_num_tokens:
                token_seqs.append(batch_token_seqs[did, :self.max_num_tokens])
                token_masks.append(batch_token_seqs[did, :self.max_num_tokens])
                token_types.append(batch_token_seqs[did, :self.max_num_tokens])
                num_seg_per_doc.append(1)
                valids.append((-1, -1)) # not contribute
                continue

            start = 0
            end = self.max_num_tokens - self.end_token_len
            num_seg = 1

            sequence = torch.cat([batch_token_seqs[did, start:end],
                                 self.end_token_ids], dim=-1)
            mask = batch_token_masks[did, start:end + self.end_token_len]
            type = torch.cat([batch_token_types[did, start:end],
                             batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)

            token_seqs.append(sequence)
            token_masks.append(mask)
            token_types.append(type)
            valids.append((start, end))

            while True:
                start = end - stride
                end = start + self.max_num_tokens - self.end_token_len - self.start_token_len
                num_seg += 1
                if end >= num_token:
                    end = min(end, num_token)

                    sequence = torch.cat([self.start_token_ids,
                                         batch_token_seqs[did, start:end]], dim=-1)
                    mask = batch_token_masks[did, start - self.start_token_len:end]
                    type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                                     batch_token_types[did, start:end]], dim=-1)

                    pad_len = self.max_num_tokens - sequence.shape[-1]
                    sequence = F.pad(sequence, (0, pad_len), value=self.pad_token_ids)
                    mask = F.pad(mask, (0, pad_len), value=0.0)
                    type = F.pad(type, (0, pad_len), value=0)

                    token_seqs.append(sequence)
                    token_masks.append(mask)
                    token_types.append(type)
                    valids.append((start, end))
                    num_seg_per_doc.append(num_seg)
                    break

                sequence = torch.cat([self.start_token_ids,
                                     batch_token_seqs[did, start:end],
                                     self.end_token_ids], dim=-1)
                mask = batch_token_masks[did, start - self.start_token_len:end + self.end_token_len]
                type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                                 batch_token_types[did, start:end],
                                 batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)
                
                token_seqs.append(sequence)
                token_masks.append(mask)
                token_types.append(type)
                valids.append((start, end))

        batch_token_seqs = torch.stack(token_seqs).long()
        batch_token_masks = torch.stack(token_masks).float()
        batch_token_types = torch.stack(token_types).long()

        # print(batch_token_seqs.shape) #torch.Size([4, 512])
        # print(batch_token_masks.shape) #torch.Size([4, 512])
        # print(batch_token_types.shape) #torch.Size([4, 512])
        # print(num_seg_per_doc) #torch.Size([4])
        # print(valids)
        # input("HERE")

        batch_output = self.transformer(input_ids=batch_token_seqs,
                                        attention_mask=batch_token_masks,
                                        token_type_ids=batch_token_types,
                                        output_attentions=True)
        token_embs, token_atts = batch_output[0], batch_output[-1][-1]

        batch_token_embs = list()
        batch_token_atts = list()
        seg_id = 0
        for num_seg in num_seg_per_doc:
            if num_seg == 1:
                emb = F.pad(token_embs[seg_id], (0, 0, 0, max_doc_length - self.max_num_tokens))
                att = F.pad(token_atts[seg_id], (0, max_doc_length - self.max_num_tokens, 0, max_doc_length - self.max_num_tokens))
                batch_token_embs.append(emb)
                batch_token_atts.append(att)
            else:
                t_embs = list()
                t_atts = list()
                t_masks = list()
                for i in range(num_seg):
                    valid = valids[seg_id + i]
                    num_valid = valid[1] - valid[0]
                    if i == 0: #valid = 511
                        sl = (0, num_valid)
                    elif i == num_seg - 1: #valid = ??
                        sl = (self.start_token_len, self.start_token_len + num_valid)
                    else: #valid = 512
                        sl = (self.start_token_len, self.start_token_len + num_valid)
                
                    emb = F.pad(token_embs[seg_id + i, sl[0]:sl[1]],
                                pad=(0, 0, valid[0], max_doc_length - valid[1]))
                    att = F.pad(token_atts[seg_id + i, :, sl[0]:sl[1], sl[0]:sl[1]],
                                pad=(valid[0], max_doc_length - valid[1], valid[0], max_doc_length - valid[1]))
                    mask = F.pad(batch_token_masks[seg_id + i, sl[0]:sl[1]],
                                 pad=(valid[0], max_doc_length - valid[1]))

                    t_embs.append(emb)
                    t_atts.append(att)
                    t_masks.append(mask)
                t_embs = torch.stack(t_embs, dim=0)
                t_atts = torch.stack(t_atts, dim=0)
                t_masks = torch.stack(t_masks, dim=0)
                doc_token_embs = t_embs.sum(0) / (t_masks.sum(dim=0).unsqueeze(-1) + self.cfg.small_positive)
                doc_token_atts = t_atts.sum(0)
                doc_token_atts = doc_token_atts / (doc_token_atts.sum(-1, keepdim=True) + self.cfg.small_positive)
                batch_token_embs.append(doc_token_embs)
                batch_token_atts.append(doc_token_atts)
            seg_id += num_seg

        batch_token_embs = torch.stack(batch_token_embs)
        batch_token_atts = torch.stack(batch_token_atts)
        print(batch_token_embs.shape)
        print(batch_token_atts.shape)
        input("HERE")
        return batch_token_embs, batch_token_atts

                        


    def forward_sd(self, batch_token_seqs, batch_token_masks, batch_token_types, stride=128):
        # sd: sliding window.
        # the logic hasn't been tested carefully, be careful using another stride

        if 'roberta' in self.transformer.config._name_or_path:
            batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, batch_num_tokens = batch_token_seqs.size()

        if batch_num_tokens <= self.max_num_tokens:
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            return batch_token_embs, batch_token_atts

        new_token_seqs, new_token_masks, new_token_types, new_token_segs = [], [], [], []
        real_num_tokens = batch_token_masks.sum(1).int().tolist()
        # ========================
        # real_num_tokens = [1300] # DEBUG
        for did, real_num_token in enumerate(real_num_tokens): # for each docs.
            print(real_num_token)
            if real_num_token <= self.max_num_tokens:
                new_token_seqs.append(batch_token_seqs[did, :self.max_num_tokens])
                new_token_masks.append(batch_token_masks[did, :self.max_num_tokens])
                new_token_types.append(batch_token_types[did, :self.max_num_tokens])
                new_token_segs.append(1)
            else:
                # real_num_token > self.max_num_tokens
                start = 0
                end = self.max_num_tokens - self.end_token_len 
                debug = list()
                new_token_seq = torch.cat([batch_token_seqs[did, start:end], self.end_token_ids], dim=-1)
                new_token_mask = batch_token_masks[did, start:end + self.end_token_len]
                new_token_type = torch.cat([batch_token_types[did, start:end],
                                            batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)
                new_token_seqs.append(new_token_seq)
                new_token_masks.append(new_token_mask)
                new_token_types.append(new_token_type)
                debug.append((start, end))
                while True:
                    start = end - stride # 383
                    end = start + self.max_num_tokens - self.end_token_len - self.start_token_len
                    if end >= real_num_token: # 511
                        end = min(end, real_num_token)
                        new_token_seq = torch.cat([self.start_token_ids, batch_token_seqs[did, start:end]], dim=-1)
                        new_token_mask = batch_token_masks[did, start - self.start_token_len:end]
                        new_token_type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                                                    batch_token_types[did, start:end]], dim=-1)
                        pad_len = self.max_num_tokens - new_token_seq.shape[-1]
                        new_token_seq = F.pad(new_token_seq, (0, pad_len), value=self.pad_token_ids)
                        new_token_mask = F.pad(new_token_mask, (0, pad_len), value=0.0)
                        new_token_type = F.pad(new_token_type, (0, pad_len), value=0)

                        debug.append((start, end))
                        new_token_seqs.append(new_token_seq)
                        new_token_masks.append(new_token_mask)
                        new_token_types.append(new_token_type)
                        new_token_segs.append(len(debug))

                        break
                    else:
                        new_token_seq = torch.cat([self.start_token_ids, batch_token_seqs[did, start:end], self.end_token_ids], dim=-1)
                        new_token_mask = batch_token_masks[did, start - self.start_token_len:end + self.end_token_len]
                        new_token_type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                                                    batch_token_types[did, start:end],
                                                    batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)
                        new_token_seqs.append(new_token_seq)
                        new_token_masks.append(new_token_mask)
                        new_token_types.append(new_token_type)
                        debug.append((start, end))
                print(debug)
                #========================================
                
        batch_token_seqs = torch.stack(new_token_seqs, dim=0).long()
        batch_token_masks = torch.stack(new_token_masks, dim=0).float()
        batch_token_types = torch.stack(new_token_types, dim=0).long()
        # print(batch_token_seqs.shape) #torch.Size([4, 512])
        # print(batch_token_masks.shape) #torch.Size([4, 512])
        # print(batch_token_types.shape) #torch.Size([4, 512])
        # print(new_token_segs) #torch.Size([4])
        # input("HERE")

        batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
        batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]

        #===========================================
        seg_id = 0
        new_token_embs = list()
        new_token_atts = list()
        for (new_token_seq, real_num_token) in zip(new_token_segs, real_num_tokens):
            if new_token_seq == 1:
                new_token_emb = F.pad(batch_token_embs[seg_id], (0, 0, 0, batch_num_tokens - self.max_num_tokens))
                new_token_att = F.pad(batch_token_atts[seg_id], (0, batch_num_tokens - self.max_num_tokens, 0, batch_num_tokens - self.max_num_tokens))
                new_token_embs.append(new_token_emb)
                new_token_atts.append(new_token_att)
            else:
                for i in range(new_token_seq):
                    if i == 0:
                        valid_num_token = self.max_num_tokens - self.end_token_len
                        new_token_emb = F.pad(batch_token_embs[seg_id + i, :valid_num_token], (0, 0, batch_num_tokens - valid_num_token))
                        new_token_mask = F.pad(batch_token_masks[seg_id + i, :valid_num_token], (0, batch_num_tokens - valid_num_token))
                        new_token_att = F.pad(batch_token_atts[seg_id + i, :, :valid_num_token, :valid_num_token], (0, batch_num_tokens - valid_num_token, 0, batch_num_tokens - valid_num_token))
                    elif i == new_token_seq - 1:
                        pass
                        # valid_num_token = self.max_num_tokens - 
                    else:
                        pass
            seg_id += new_token_seq

                


        #===========================================

        seg_id, new_token_embs, new_token_atts = 0, [], []
        for (new_token_seq, real_num_token) in zip(new_token_segs, real_num_tokens):
            if new_token_seq == 1:
                new_token_emb = F.pad(batch_token_embs[seg_id], (0, 0, 0, batch_num_tokens - self.max_num_tokens))
                new_token_att = F.pad(batch_token_atts[seg_id], (0, batch_num_tokens - self.max_num_tokens, 0, batch_num_tokens - self.max_num_tokens))
                new_token_embs.append(new_token_emb)
                new_token_atts.append(new_token_att)

            else:
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

    def forward(self, batch_token_seqs, batch_token_masks, batch_token_types, use_original=False):
        if use_original:
            return self.forward_o(batch_token_seqs, batch_token_masks, batch_token_types)
        return self.forward_sd_rewrite(batch_token_seqs, batch_token_masks, batch_token_types)

