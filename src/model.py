import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from itertools import permutations
from loss import Loss
from torch_geometric.nn import RGCNConv
from attentionUnet import AttentionUNet
from similarity import Similarity

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


class RGCN(nn.Module):
    """
    Example useage:
    x = torch.randn(4, 128)
    node_type = torch.tensor([0, 0, 1, 2])
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    edge_type = torch.tensor([0, 0, 1, 1, 2])
    model = RGCN(128, 128, 128, 3, 128)

    out = model(x, node_type, edge_index, edge_type)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_node_type=3, type_dim=128):
        super(RGCN, self).__init__()
        self.activation = nn.ReLU()

        self.node_emb = torch.nn.Embedding(num_node_type, type_dim)

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels + type_dim, in_channels + type_dim, num_relations))
        self.convs.append(RGCNConv(in_channels + type_dim, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, x, node_type, edge_index, edge_type):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        edge_type: Edge type labels [num_edges]
        """
        x = torch.cat([x, self.node_emb(node_type)], dim=-1)
        for conv in self.convs[:-1]:
            x = self.activation(conv(x, edge_index, edge_type))
        x = self.convs[-1](x, edge_index, edge_type) 
        return self.activation(x)

    
class Model(nn.Module):
    
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        if cfg.transformer == 'bert-base-cased':
            self.hidden_dim = 768 #NOTE: change if transformer changes.
        self.RGCN = RGCN(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3, 3, self.cfg.type_dim).to(self.cfg.device)
        self.unet = AttentionUNet(3, 256, down_channel=256)
        self.similarity = Similarity(self.hidden_dim).to(self.cfg.device)

        self.loss = Loss(cfg)
        self.W_h = nn.Linear(self.hidden_dim + 256, self.hidden_dim + 256)
        self.W_t = nn.Linear(self.hidden_dim + 256, self.hidden_dim + 256)
        self.RE_predictor_module = nn.Linear((self.hidden_dim + 256) * self.cfg.bilinear_block_size, self.cfg.num_rel)

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
        
    def compute_mentions_embs(self, batch_token_embs, batch_start_mpos, num_mention_per_doc):
        batch_did = torch.arange(len(batch_token_embs)).repeat_interleave(num_mention_per_doc)
        batch_mentions_pos = batch_start_mpos[batch_start_mpos != -1].flatten()
        return batch_token_embs[batch_did, batch_mentions_pos]
        # out: [all_mentions, 768], order: first_doc, first_mention

    def compute_sent_embs(self, batch_token_embs, num_sent_per_doc, batch_sent_pos):
        batch_sent_embs = list()

        num_sent_cumsum = torch.cumsum(num_sent_per_doc, dim=0)

        for sid, sent in enumerate(batch_sent_pos):
            did = (sid < num_sent_cumsum).nonzero()[0].item()
            batch_sent_embs.append(torch.mean(batch_token_embs[did, sent[0]:sent[1]], dim=0))
            
        return torch.stack(batch_sent_embs).to(self.cfg.device)

    def forward(self, batch_input, is_training=False):
        batch_token_seqs = batch_input['batch_token_seqs']
        batch_token_masks = batch_input['batch_token_masks']
        batch_token_types = batch_input['batch_token_types']
        batch_start_mpos = batch_input['batch_start_mpos']
        batch_epair_rels = batch_input['batch_epair_rels']
        batch_mpos2sentid = batch_input['batch_mpos2sentid']
        num_entity_per_doc = batch_input['num_entity_per_doc']
        num_mention_per_doc = batch_input['num_mention_per_doc']
        batch_sent_pos = batch_input['batch_sent_pos']
        num_sent_per_doc = batch_input['num_sent_per_doc']

        batch_token_embs, batch_token_atts = self.transformer(batch_token_seqs, batch_token_masks, batch_token_types)
        batch_entity_embs = self.compute_entity_embs(batch_token_embs, batch_start_mpos, num_entity_per_doc)
        # batch_mention_embs = self.compute_mentions_embs(batch_token_embs, batch_start_mpos, num_mention_per_doc)
        batch_sent_embs = self.compute_sent_embs(batch_token_embs, num_sent_per_doc, batch_sent_pos)

        batch_did = torch.arange(len(batch_token_embs)).repeat_interleave(num_mention_per_doc)
        index = batch_start_mpos != -1
        batch_mentions_pos = batch_start_mpos[index]
        batch_mention_embs = batch_token_embs[batch_did, batch_mentions_pos]

        num_mention_per_entity = torch.count_nonzero(index, dim=-1)
        start_entity_pos = torch.cumsum(torch.cat([torch.tensor([0]), num_entity_per_doc]), dim=0)
        to_eid = list()
        for did in range(len(start_entity_pos) - 1):
            start = start_entity_pos[did]
            end = start_entity_pos[did + 1]
            num_entity = end - start
            temp = num_mention_per_entity[start_entity_pos[did]: start_entity_pos[did + 1]]
            to_eid.append(torch.arange(num_entity).repeat_interleave(temp))
        to_eid = torch.cat(to_eid)

        node_reps = torch.cat([batch_entity_embs, batch_mention_embs, batch_sent_embs], dim=0).to(self.cfg.device)
        types_to_num = torch.tensor([len(batch_entity_embs), len(batch_mention_embs), len(batch_sent_embs)])
        node_types = torch.arange(3).repeat_interleave(types_to_num).to(self.cfg.device)

        edges = list()
        edges_type = list()
        
        mention_index = torch.arange(types_to_num[1]) + types_to_num[0]

        offsets = start_entity_pos[:-1].repeat_interleave(num_mention_per_doc)
        entity_index = to_eid + offsets

        edges.append(torch.stack([mention_index, entity_index]))
        edges_type.append(torch.tensor([0]).repeat(edges[-1].shape[-1]))


        sent_cumsum = torch.cat([torch.tensor([0]), num_sent_per_doc], dim=-1).cumsum(dim=-1) + len(batch_entity_embs) + len(batch_mention_embs)
        offsets = sent_cumsum[:-1].repeat_interleave(num_mention_per_doc)
        sentence_index = batch_mpos2sentid[:, 1] + offsets

        edges.append(torch.stack([mention_index, sentence_index]))
        edges_type.append(torch.tensor([1]).repeat(edges[-1].shape[-1]))

        head_sent = list()
        tail_sent = list()
        for id in range(len(sent_cumsum) - 1):
            head_sent.append(torch.arange(sent_cumsum[id], sent_cumsum[id + 1] - 1))
            tail_sent.append(torch.arange(sent_cumsum[id] + 1, sent_cumsum[id + 1]))

        head_sent = torch.cat(head_sent, dim=-1)
        tail_sent = torch.cat(tail_sent, dim=-1)

        edges.append(torch.stack([head_sent, tail_sent]))
        edges_type.append(torch.tensor([2]).repeat(edges[-1].shape[-1]))

        edges = torch.cat(edges, dim=1).to(self.cfg.device)
        edges_type = torch.cat(edges_type, dim=0).to(self.cfg.device)

        node_reps = self.RGCN(node_reps, node_types, edges, edges_type)
        batch_entity_embs = node_reps[:len(batch_entity_embs)]
       
        batch_entity_embs = torch.split(batch_entity_embs, num_entity_per_doc.tolist())

        u_inputs = list()
        for doc_entity_embs in batch_entity_embs:
            out = self.similarity(doc_entity_embs)
            u_inputs.append(out)

        u_outs = list()
        for u_input in u_inputs:
            temp = u_input.unsqueeze(0)
            u_outs.append(self.unet(temp).squeeze(0))

        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)

        head_entity_pairs = list()
        tail_entity_pairs = list()
        h_t = list()
        batch_labels = list()

        #_______

        for did in range(self.cfg.batch_size):
            doc_epair_rels = batch_epair_rels[did]
            offset = int(start_entity_pos[did])
            for e_h, e_t in doc_epair_rels:
                h_t.append(u_outs[did][e_h, e_t])
                head_entity_pairs.append(e_h + offset)
                tail_entity_pairs.append(e_t + offset)
                pair_label = torch.zeros(self.cfg.num_rel)
                for r in doc_epair_rels[(e_h, e_t)]:
                    pair_label[self.cfg.data_rel2id[r]] = 1
                batch_labels.append(pair_label)
        #_______

        # for did in range(len(start_entity_pos) - 1):
        #     doc_epair_rels = batch_epair_rels[did]
        #     offset = int(start_entity_pos[did])
        #     for eid_h, eid_t in permutations(np.arange(offset, int(start_entity_pos[did + 1])), 2):
        #         h_t.append(u_outs[did][eid_h - offset, eid_t - offset])
        #         pair_labels = torch.zeros(self.cfg.num_rel)
        #         for r in doc_epair_rels[(eid_h - offset, eid_t - offset)]:
        #             pair_labels[self.cfg.data_rel2id[r]] = 1
        #         batch_labels.append(pair_labels)
        #         head_entity_pairs.append(eid_h)
        #         tail_entity_pairs.append(eid_t)

        h_t = torch.stack(h_t)
        head_entity_pairs = torch.tensor(head_entity_pairs).to(self.cfg.device)
        tail_entity_pairs = torch.tensor(tail_entity_pairs).to(self.cfg.device)
        batch_labels = torch.stack(batch_labels).to(self.cfg.device)

        head_entity_embs = torch.cat([batch_entity_embs[head_entity_pairs], h_t], dim=-1)
        tail_entity_embs = torch.cat([batch_entity_embs[tail_entity_pairs], h_t], dim=-1)



        head_entity_rep = torch.tanh(self.W_h(head_entity_embs))
        tail_entity_rep = torch.tanh(self.W_t(tail_entity_embs))

        head_entity_rep = head_entity_rep.view(-1, (self.hidden_dim + 256) // self.cfg.bilinear_block_size, self.cfg.bilinear_block_size)
        tail_entity_rep = tail_entity_rep.view(-1, (self.hidden_dim + 256) // self.cfg.bilinear_block_size, self.cfg.bilinear_block_size)
        batch_RE_reps = (head_entity_rep.unsqueeze(3) * tail_entity_rep.unsqueeze(2)).view(-1, (self.hidden_dim + 256) * self.cfg.bilinear_block_size)
        batch_RE_reps = self.RE_predictor_module(batch_RE_reps)

        if is_training:
            return self.loss.ATLOP_loss(batch_RE_reps, batch_labels)
        else:
            return self.loss.ATLOP_pred(batch_RE_reps), batch_labels
