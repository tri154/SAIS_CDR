import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import Loss
from torch_geometric.utils import to_undirected
from torch.nn.utils.rnn import pad_sequence
from models.transformers import Transformer
from models.rgcn import RGCN
# from models.cnn import CNN


class Model(nn.Module):

    def __init__(self, cfg, emb_size=512):
        super(Model, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        self.emb_size = emb_size

        if "bert-base-cased" == cfg.transformer:
            self.hidden_dim = 768 #NOTE: change if transformer changes.
        elif "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" == cfg.transformer:
            self.hidden_dim = 768
        
        self.num_node_types = 3

        self.extractor_trans = nn.Linear(self.hidden_dim, emb_size)
        self.rgcn = RGCN(emb_size, emb_size, num_relations=4, num_node_type=3, type_dim=self.cfg.type_dim, num_layers=self.cfg.graph_layers)
        # self.cnn = CNN(emb_size, device=self.cfg.device)
        self.ht_extractor = nn.Linear(emb_size*4, emb_size*2)


        self.MIP_Linear = nn.Sequential(
            nn.Linear(emb_size * 5, emb_size * 2),
            nn.Dropout(0.1)
        )
        
        # self.MIP_Linear1 = nn.Linear(emb_size * 5, emb_size * 4)
        # self.MIP_Linear2 = nn.Linear(emb_size * 4, emb_size * 2)
        self.bilinear = nn.Linear(emb_size * 2, self.cfg.num_rel)

        self.loss = Loss(cfg)

        # doc_data = {'doc_tokens': doc_tokens, # list of token id of the doc. single dimension single dimension.
        #             'doc_title': doc_title,
        #             'doc_start_mpos': doc_start_mpos, # a dict of set. entity_id -> set of start of mentions token.
        #             'doc_sent_pos': doc_sent_pos} # a dict, sent_id -> (start, end) in token.


    def compute_entity_embs(self, batch_token_embs, batch_start_mpos, num_entity_per_doc):
        batch_did = torch.arange(self.cur_batch_size).repeat_interleave(num_entity_per_doc).unsqueeze(-1).to(self.cfg.device)
        # batch_token_embs = F.pad(batch_token_embs, (0, 0, 0, 1), value=self.cfg.small_negative) #NOTE: can optimize.
        batch_entity_embs = batch_token_embs[batch_did, batch_start_mpos].logsumexp(dim=-2)

        return batch_entity_embs

    def compute_mention_embs(self, batch_token_embs, batch_start_mpos, num_mention_per_doc):
        batch_mention_pos = batch_start_mpos[batch_start_mpos != -1].flatten()
        batch_did = torch.arange(self.cur_batch_size).repeat_interleave(num_mention_per_doc).to(self.cfg.device)
        batch_mention_embs = batch_token_embs[batch_did, batch_mention_pos]
        
        return batch_mention_embs

    def compute_sentence_embs(self, batch_token_embs, batch_token_atts, batch_sent_pos):
        batch_sent_embs = list()

        for did in range(self.cur_batch_size):
            doc_token_embs = batch_token_embs[did]
            doc_token_atts = batch_token_atts[did]
            doc_sent_pos = batch_sent_pos[did]

            for sid in sorted(doc_sent_pos): # keep sentence order.
                start, end = doc_sent_pos[sid]
                sent_token_embs = doc_token_embs[start:end]
                sent_token_atts = doc_token_atts[:, start:end, start:end]

                sent_token_atts = sent_token_atts.mean(dim=(1, 0))
                sent_token_atts = sent_token_atts / (sent_token_atts.sum(0) / self.cfg.small_positive)
                batch_sent_embs.append(sent_token_atts @ sent_token_embs)

        batch_sent_embs = torch.stack(batch_sent_embs).to(self.cfg.device)
        return batch_sent_embs

    def compute_node_embs(self,
                          batch_token_embs,
                          batch_token_atts,
                          batch_start_mpos,
                          batch_sent_pos,
                          num_entity_per_doc,
                          num_mention_per_doc,):
        device = self.cfg.device
        batch_entity_embs = self.compute_entity_embs(batch_token_embs, batch_start_mpos, num_entity_per_doc)
        batch_mention_embs = self.compute_mention_embs(batch_token_embs, batch_start_mpos, num_mention_per_doc)
        batch_sent_embs = self.compute_sentence_embs(batch_token_embs, batch_token_atts, batch_sent_pos)

        batch_node_embs = [batch_entity_embs, batch_mention_embs, batch_sent_embs]
        num_per_type = [len(type) for type in batch_node_embs]
        nodes_type = torch.arange(self.num_node_types, device=device).repeat_interleave(torch.tensor([len(nodes) for nodes in batch_node_embs], device=device))
        batch_node_embs = torch.cat(batch_node_embs, dim=0)

        return batch_node_embs, nodes_type, num_per_type


    def get_entity_mention_link(self, num_mention_per_entity, num_per_type):
        device = self.cfg.device
        entity = torch.arange(len(num_mention_per_entity), device=device).repeat_interleave(num_mention_per_entity)
        mention = torch.arange(num_per_type[0], num_per_type[0] + num_per_type[1], device=device)
        entity_mention_links = torch.stack([entity, mention]).to(device)
        return to_undirected(entity_mention_links)

    def get_sentence_sentence_link(self, num_sent_per_doc, num_per_type):
        device = self.cfg.device
        offset = num_per_type[0] + num_per_type[1]
        n_cumsum = torch.cumsum(torch.cat([torch.tensor([0]), num_sent_per_doc], dim=0), dim=0)
        start = []
        for did in range(self.cur_batch_size):
            temp = torch.arange(int(n_cumsum[did]), int(n_cumsum[did + 1]) - 1, device=device)
            start.append(temp)

        start = torch.cat(start, dim=0)
        start = start + offset
        end = start + 1
        sent_sent_links = torch.stack([start, end]).to(device)
        return to_undirected(sent_sent_links)

    def get_ment_sent_link(self, batch_mpos2sid, num_sent_per_doc, num_mention_per_doc, num_per_type):
        device = self.cfg.device
        sent_cumsum = torch.cat([torch.tensor([0]), num_sent_per_doc], dim=-1).cumsum(dim=-1) + num_per_type[0] + num_per_type[1]
        offsets = sent_cumsum[:-1].repeat_interleave(num_mention_per_doc).to(device)
        sentence = batch_mpos2sid[:, 1] + offsets
        mention = torch.arange(num_per_type[0], num_per_type[0] + num_per_type[1], device=device)
        ment_sent_links = torch.stack([mention, sentence]).to(device)
        return to_undirected(ment_sent_links)


    def get_ment_ment_link(self, batch_mentions_link, num_mentlink_per_doc, num_mention_per_doc, num_per_type):
        device = self.cfg.device
        temp = torch.cat([torch.tensor([0]), num_mention_per_doc]).cumsum(dim=0) + num_per_type[0]
        temp = temp[:-1].repeat_interleave(num_mentlink_per_doc).unsqueeze(0).to(device)
        ment_ment_links = batch_mentions_link + temp
        return to_undirected(ment_ment_links)


    def get_relation_map(self, gcn_nodes, num_entity_per_doc):
        device = self.cfg.device
        relation_map = list()
        max_entity_per_doc = max(num_entity_per_doc)
        batch_entity_embs =  torch.split(gcn_nodes[-1][:torch.sum(num_entity_per_doc)], num_entity_per_doc.tolist())
        for did in range(self.cur_batch_size):
            doc_entity_embs = batch_entity_embs[did]
            e_s_map = torch.einsum('ij, jk -> jik', doc_entity_embs, doc_entity_embs.T).to(device)
            offset = max_entity_per_doc - num_entity_per_doc[did]
            if offset > 0:
                e_s_map = F.pad(e_s_map, (0, offset, 0, offset), value=0).to(device)
            relation_map.append(e_s_map)

        relation_map = torch.stack(relation_map).to(device)
        return relation_map

    def get_entity_pairs(self, batch_epair_rels, num_entity_per_doc):
        device = self.cfg.device

        head_entities = list()
        tail_entities = list()
        batch_labels = list()

        for did in range(self.cur_batch_size):
            doc_epair_rels = batch_epair_rels[did]
            for e_h, e_t in doc_epair_rels:
                head_entities.append(e_h)
                tail_entities.append(e_t)
                pair_label = torch.zeros(self.cfg.num_rel)
                for r in doc_epair_rels[(e_h, e_t)]:
                    pair_label[self.cfg.data_rel2id[r]] = 1
                batch_labels.append(pair_label)

        start_entity_pos = torch.cumsum(torch.cat([torch.tensor([0]), num_entity_per_doc]), dim=0)
        num_rel_per_doc = torch.tensor([len(doc_epair_rels) for doc_epair_rels in batch_epair_rels]).cpu()
        offsets = start_entity_pos[:-1].repeat_interleave(num_rel_per_doc).to(device)

        head_entities = torch.tensor(head_entities).to(device)
        tail_entities = torch.tensor(tail_entities).to(device)
        batch_labels = torch.stack(batch_labels).to(device)
        
        return head_entities, tail_entities, batch_labels, offsets, num_rel_per_doc # reuse
        
    
    def forward(self, batch_input, current_epoch=None, is_training=False):
        batch_token_seqs = batch_input['batch_token_seqs']
        batch_token_masks = batch_input['batch_token_masks']
        batch_token_types = batch_input['batch_token_types']
        batch_start_mpos = batch_input['batch_start_mpos']
        batch_epair_rels = batch_input['batch_epair_rels']
        batch_sent_pos = batch_input['batch_sent_pos']
        batch_mpos2sid = batch_input['batch_mpos2sid']
        batch_mentions_link = batch_input['batch_mentions_link']
        batch_teacher_logits = batch_input['batch_teacher_logits']
        num_mentlink_per_doc = batch_input['num_mentlink_per_doc']
        num_entity_per_doc = batch_input['num_entity_per_doc']
        num_mention_per_doc = batch_input['num_mention_per_doc']
        num_mention_per_entity = batch_input['num_mention_per_entity']
        num_sent_per_doc = batch_input['num_sent_per_doc']
        device = self.cfg.device
        self.cur_batch_size = len(batch_token_seqs)

        batch_token_embs, batch_token_atts = self.transformer(batch_token_seqs, batch_token_masks, batch_token_types)
        batch_token_embs = self.extractor_trans(batch_token_embs)

        batch_token_embs = F.pad(batch_token_embs, (0, 0, 0, 1), value=self.cfg.small_negative)

        batch_node_embs, nodes_type, num_per_type = self.compute_node_embs(batch_token_embs,
                                                                          batch_token_atts,
                                                                          batch_start_mpos,
                                                                          batch_sent_pos,
                                                                          num_entity_per_doc,
                                                                          num_mention_per_doc)
    
        #nodes order:
        # doc1_e1, doc1_e2 ... doc1_en, doc2_e1, ...
        # doc1_e1_mention_1, doc1_e1_mention2, ...
        # doc1_sent1, doc1_sent2, ...

        ent_ment_links = self.get_entity_mention_link(num_mention_per_entity, num_per_type) # TODO: move links to preprocessing for performance.
        sent_sent_links = self.get_sentence_sentence_link(num_sent_per_doc, num_per_type)
        ment_sent_links = self.get_ment_sent_link(batch_mpos2sid, num_sent_per_doc, num_mention_per_doc, num_per_type)
        ment_ment_links = self.get_ment_ment_link(batch_mentions_link, num_mentlink_per_doc, num_mention_per_doc, num_per_type)

        edges = [ent_ment_links, sent_sent_links, ment_sent_links, ment_ment_links]
        edges_type = torch.arange(len(edges), device=device).repeat_interleave(torch.tensor([ts.shape[-1] for ts in edges], device=device))
        edges = torch.cat(edges, dim=-1)

        gcn_nodes = self.rgcn(batch_node_embs, nodes_type, edges, edges_type)

        # relation_map = self.get_relation_map(gcn_nodes, num_entity_per_doc)
        # relation_map = self.cnn(relation_map) # 4, 512, n_e_max, n_e_max

        # Compute relation representation, need refactor.

        gcn_nodes = torch.cat([gcn_nodes[0], gcn_nodes[-1]], dim=-1)
        head_entities, tail_entities, batch_labels, offsets, num_rel_per_doc = self.get_entity_pairs(batch_epair_rels, num_entity_per_doc)
        entity_h = gcn_nodes[head_entities + offsets]
        entity_t = gcn_nodes[tail_entities + offsets]
        entity_ht = self.ht_extractor(torch.cat([entity_h, entity_t], dim=-1)) # 14, 1024

        # batch_did = torch.arange(self.cur_batch_size).repeat_interleave(num_rel_per_doc).to(device)
        # relation = relation_map[batch_did, :, head_entities, tail_entities] # 14, 512
        
        batch_token_atts = F.pad(batch_token_atts, ((0, 0, 0, 1)), value=0.0)

        batch_did = torch.arange(self.cur_batch_size).repeat_interleave(num_entity_per_doc).unsqueeze(-1).to(device)
        batch_entity_att = batch_token_atts[batch_did, :, batch_start_mpos] #NOTE: might take lot of memory.
        batch_entity_att = torch.sum(batch_entity_att, dim=1) / (num_mention_per_entity.unsqueeze(-1).unsqueeze(-1) + 1e-5)
        batch_entity_att = batch_entity_att.mean(dim=1) # 16, 370 ,TESTED

        batch_entity_att = torch.split(batch_entity_att, num_entity_per_doc.tolist())
        batch_entity_att = pad_sequence(batch_entity_att, batch_first=True, padding_value = 0.0) # 4, max_num_e, 512
        batch_entity_att = torch.bmm(batch_entity_att, batch_token_embs[:, :-1])  # 4, max_e_num, 512

        batch_did = torch.arange(self.cur_batch_size).repeat_interleave(num_rel_per_doc).unsqueeze(-1).to(device)
        pair_entities = torch.stack([head_entities, tail_entities], dim=-1)
        e_tw = batch_entity_att[batch_did, pair_entities]
        e_tw = e_tw.reshape(len(e_tw), -1) # 14, 1024
        
        # relation_rep = torch.cat([relation, e_tw, entity_ht], dim=-1)
        relation_rep = torch.cat([e_tw, entity_ht], dim=-1)

        sc_loss = 0
        if is_training and self.cfg.use_sc:
            sc_loss = self.loss.SC_loss(relation_rep, batch_labels)

        relation_rep = torch.tanh(self.MIP_Linear(relation_rep))
        # relation_rep = torch.tanh(self.MIP_Linear1(relation_rep))
        # relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))
        logits = self.bilinear(relation_rep)

        if not is_training:
            return self.loss.AT_pred(logits), batch_labels
            
        at_loss = self.loss.AT_loss_original(logits, batch_labels)

        kd_loss = torch.tensor(0.0)
        current_tradeoff = 0.0
        if batch_teacher_logits is not None:
            kd_loss, current_tradeoff = self.loss.PSD_loss(logits, batch_teacher_logits, current_epoch)


        loss = at_loss + current_tradeoff * kd_loss  + self.cfg.sc_weight * sc_loss
        return loss, torch.split(logits.detach().cpu(), num_rel_per_doc.tolist())
        
