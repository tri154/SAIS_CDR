import torch
import torch.nn as nn


class Similarity(nn.Module):
    def __init__(self, hidden_dim):
        super(Similarity, self).__init__()
        self.hidden_dim = hidden_dim
        self.bilinear_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, ent_encode):  # [n_entity, hidden_dim]
        dot = self.dot_product_attention(ent_encode).unsqueeze(0)         # [1, n_entity, n_entity]
        cosine = self.cosine_attention(ent_encode).unsqueeze(0)           # [1, n_entity, n_entity]
        bilinear = self.bilinear_attention(ent_encode).unsqueeze(0)       # [1, n_entity, n_entity]

        similarity = torch.cat([dot, cosine, bilinear], dim=0)
        return similarity

    def dot_product_attention(self, x):
        return torch.matmul(x, x.transpose(0, 1))


    def cosine_attention(self, x, eps=1e-13):
        a_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        b_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.matmul(a_norm, b_norm.transpose(0, 1))

    def bilinear_attention(self, x):
        Wx = torch.matmul(x, self.bilinear_weight)
        return torch.matmul(Wx, x.transpose(0, 1)) 
