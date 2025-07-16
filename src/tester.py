import torch
import math
import torch.nn.functional as F 
import torch.nn.utils.rnn as rnn
import numpy as np

class Tester:
    def __init__(self, cfg, dev_set, test_set):
        self.cfg = cfg
        self.dev_set = dev_set
        self.test_set = test_set

    def prepare_batch(self, batch_size, dataset='dev'):
        inputs = self.test_set if dataset == 'test' else self.dev_set 

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
  
            for doc_input in batch_inputs:
                batch_titles.append(doc_input['doc_title'])
                batch_token_seqs.append(doc_input['doc_tokens'])
                batch_start_mpos.append(doc_input['doc_start_mpos'])
                batch_sent_pos.append(doc_input['doc_sent_pos'])
                batch_mpos2sid.append(doc_input['doc_mpos2sid'])
                batch_mentions_link.append(doc_input['doc_mentions_link'])
                num_sent_per_doc.append(len(doc_input['doc_sent_pos']))


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
                    }

    def cal_f1(self, preds, labels, epsilon=1e-8):
        preds = preds.to(dtype=torch.int)
        labels = labels.to(dtype=torch.int)
        tp = ((preds[:, 1] == 1) & (labels[:, 1] == 1)).to(dtype=torch.float32).sum()
        tn = ((labels[:, 1] == 1) & (preds[:, 1] != 1)).to(dtype=torch.float32).sum()
        fp = ((preds[:, 1] == 1) & (labels[:, 1] != 1)).to(dtype=torch.float32).sum()
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + tn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return precision, recall, f1

        
    def test(self, model, dataset='dev'):
        model.eval()
        all_preds = list()
        all_labels = list()
        with torch.no_grad():
            for idx_batch, batch_inputs in enumerate(self.prepare_batch(self.cfg.batch_size, dataset)):
                batch_preds, batch_labels = model(batch_inputs, is_training=False)
                all_preds.append(batch_preds)
                all_labels.append(batch_labels)

        all_preds = torch.cat(all_preds, dim=0).to(self.cfg.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.cfg.device)

        precision, recall, f1 = self.cal_f1(all_preds, all_labels)
        return precision, recall, f1
