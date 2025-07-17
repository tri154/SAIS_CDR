import torch
import torch.nn as nn
import torch.nn.functional as F
class Loss:
    # TODO: eliminate clone()
    def __init__(self, cfg):
        self.cfg = cfg
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def AT_loss_original(self, logits, labels):
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, self.cfg.id_rel_thre] = 1.0
        labels[:, self.cfg.id_rel_thre] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss
    
    def AT_pred(self, logits):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        top_v, _ = torch.topk(logits, self.cfg.topk, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def AT_loss(self, batch_RE_reps, batch_epair_rels): #dont use anymore because of clone.
        batch_pos_thre = torch.clone(batch_epair_rels)
        batch_pos_thre[:, self.cfg.id_rel_thre] = 1

        batch_pos_reps = batch_RE_reps + (1 - batch_pos_thre) * self.cfg.small_negative
        batch_pos_loss = - (F.log_softmax(batch_pos_reps, dim=-1) * batch_epair_rels)
        batch_pos_loss = batch_pos_loss.sum(-1)

        batch_neg_thre = 1 - batch_epair_rels
        batch_thre_rels = torch.zeros_like(batch_neg_thre)
        batch_thre_rels[:, self.cfg.id_rel_thre] = 1
        batch_neg_reps = batch_RE_reps + (1 - batch_neg_thre) * self.cfg.small_negative
        batch_neg_loss = - (F.log_softmax(batch_neg_reps, dim=-1) * batch_thre_rels)
        batch_neg_loss = batch_neg_loss.sum(-1)

        return (batch_pos_loss + batch_neg_loss).mean()

    def PSD_loss(self, logits, teacher_logits, current_epoch):
        current_temp = self.cfg.upper_temp - (self.cfg.upper_temp - self.cfg.lower_temp) * current_epoch / (self.cfg.num_epoch - 1.0)
        current_tradeoff = self.cfg.loss_tradeoff * current_epoch / (self.cfg.num_epoch - 1.0)
        loss = self.kd_loss(F.log_softmax(logits / current_temp, dim=1),
                            F.softmax(teacher_logits / current_temp, dim=1))

        return loss, current_tradeoff


    def SC_loss(self, reps, oh_labels):
        '''
            A new loss function, that only works for single label.
        '''

        ####
        # DEBUG
        # labels = torch.ones(25)
        # labels[5] = 2
        # labels[10] = 3
        # reps = torch.rand(25, 50)
        # n_sample = len(reps)
        ####
        device = self.cfg.device
        reps = F.normalize(reps, p=2, dim=1)
        n_sample = len(reps)
        labels = oh_labels.argmax(dim=-1)
        uniques, counts = torch.unique(labels,return_counts=True)
        val2count = dict(zip(uniques.tolist(), counts.tolist()))
        pairs = torch.triu_indices(n_sample, n_sample, offset=1).to(device)

        anchor_values = labels[pairs[0]].to(device)
        candidate_values = labels[pairs[1]].to(device)
        mask = (anchor_values == candidate_values)

        has_1 = (counts == 1).nonzero().squeeze(-1).to(device)
        has_1 = uniques[has_1].to(device) # unique labels that have only 1 sample.
        if len(has_1) != 0:
            mask = mask & ~torch.isin(anchor_values, has_1)

        pairs = pairs[:, mask]
        anchor_values = anchor_values[mask].cpu()
        anchor_values.apply_(val2count.get)
        anchor_values = anchor_values.to(device)

        numerator = torch.exp(torch.sum(reps[pairs[0]] * reps[pairs[1]], dim=-1) / self.cfg.sc_temp).to(device)

        unique_anchor_idx = pairs[0].unique().to(device) # 22

        temp = reps[unique_anchor_idx] # 22, 50

        cached = torch.matmul(temp, reps.T)
        cached = torch.exp(cached / self.cfg.sc_temp) 
        own = cached[torch.arange(len(unique_anchor_idx)), unique_anchor_idx] # 22
        cached = torch.sum(cached, dim=1) - own # 22

        cached = {int(i.item()): cached[idx] for idx, i in enumerate(unique_anchor_idx)}

        denominator = torch.stack([cached[k.item()] for k in pairs[0]]).to(device)

        loss = torch.log(numerator / (denominator + 1e-6)) * (-1 / (anchor_values - 1))
        loss = loss.sum()
        return loss
