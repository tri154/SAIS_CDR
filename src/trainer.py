import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import math

class Trainer:
    def __init__(self, cfg, model, train_set=None, tester=None):
        self.cfg = cfg
        self.model = model
        self.train_set = train_set
        self.tester = tester

        print()
        #TODO: optimizer

        # doc_data = {'doc_tokens': doc_tokens,
        #             'doc_title': doc_title,
        #             'doc_start_mpos': doc_start_mpos,
        #             'doc_sent_pos': doc_sent_pos}
    def prepare_batch_train(self, batch_size):
        train_set = self.train_set
        num_batch = math.ceil(len(train_set) / batch_size)

        for idx_batch in range(num_batch):
            batch_inputs = train_set[idx_batch * batch_size:(idx_batch + 1) * batch_size]

            batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
            batch_titles = []

            for doc_input in batch_inputs:
                batch_titles.append(doc_input['doc_title'])
                batch_token_seqs.append(doc_input['doc_tokens'])

                doc_seqs_len = doc_input['doc_tokens'].shape[0]
                batch_token_masks.append(torch.ones(doc_seqs_len))
                doc_tokens_types = torch.zeros(doc_seqs_len)

                for sid in range(len(doc_input['doc_sent_pos'])):
                    start, end = doc_input['doc_sent_pos'][sid][0], doc_input['doc_sent_pos'][sid][1]
                    doc_tokens_types[start:end] = sid % 2

                batch_token_types.append(doc_tokens_types)


            batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long().to(self.cfg.device)
            batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float().to(self.cfg.device)
            batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long().to(self.cfg.device)


            yield {'batch_titles': np.array(batch_titles),
                    'batch_token_seqs': batch_token_seqs,
                    'batch_token_masks': batch_token_masks,
                    'batch_token_types': batch_token_types}
            
    def debug(self):
        for idx_batch, batch_input in enumerate(self.prepare_batch_train(self.cfg.batch_size)):
            self.model(batch_input)
            input("Stop")
                

    def one_epoch_train(self, batch_size):
        self.model.train()
        self.optimizer.zero_grad()

        np.random.shuffle(self.train_set)

        for idx_batch, batch_input in enumerate(self.prepare_batch_train(batch_size)):
            batch_loss = self.model(batch_input)


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
                






        

        
