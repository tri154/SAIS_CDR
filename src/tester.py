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

    # def test_dev(self, dev_set=None):
    #     self.dev_set = dev_set if dev_set is not None else self.dev_set

    # def test(self, test_set=None):
    #     self.test_set = test_set if test_set is not None else self.dev_set


    def prepare_batch(self, batch_size, dataset='dev'):
        inputs = self.test_set if dataset == 'test' else self.dev_set 
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

    def cal_f1(self, preds, labels, epsilon=1e-8):
        TP = (preds * labels).sum()
        precision = TP / (preds.sum() + epsilon)
        recall = TP / (labels.sum() + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
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
                if idx_batch == 2:
                    break

        all_preds = torch.cat(all_preds, dim=0).to(self.cfg.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.cfg.device)

        precision, recall, f1 = self.cal_f1(all_preds, all_labels)
        print(precision, recall, f1)
