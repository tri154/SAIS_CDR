import torch
import torch.nn.functional as F
class Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def ATLOP_loss(self, batch_RE_reps, batch_epair_rels):
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

    def ATLOP_pred(self, logits):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        top_v, _ = torch.topk(logits, self.cfg.topk, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output






        
