import torch
import torch.nn as nn
from torch.nn import Softmax
def INF(B, H, W, device=None):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CC_module(nn.Module):

    def __init__(self, in_dim=256, device=None):
        super(CC_module, self).__init__()
        self.device = device
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width, self.device)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class CNN(nn.Module):
    def __init__(self, emb_size, device=None):
        super(CNN, self).__init__()
        self.device = device
        self.emb_size = emb_size
        self.inter_channel = int(emb_size // 2)

        self.cc_module = CC_module(device=device)

        self.conv_reason_e_l1 = nn.Sequential(
            nn.Conv2d(emb_size, self.inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_reason_e_l2 = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(inplace=True),)
        self.conv_reason_e_l3 = nn.Sequential(
            nn.Conv2d(self.inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(inplace=True),
        )
    def forward(self, relation_map):
        r_rep_e = self.conv_reason_e_l1(relation_map) #[batch_size, inter_channel, ent_num, ent_num]
        cc_output = self.cc_module(r_rep_e)
        r_rep_e_2 = self.conv_reason_e_l2(cc_output)
        cc_output_2 = self.cc_module(r_rep_e_2)
        r_rep_e_3 = self.conv_reason_e_l3(cc_output_2)
        return r_rep_e_3
