import torch
from torch_geometric.nn import RGCNConv
import torch.nn as nn


class RGCN(nn.Module):
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
        return x


if __name__ == "__main__":
    x = torch.randn(4, 128)
    node_type = torch.tensor([0, 0, 1, 2])
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    edge_type = torch.tensor([0, 0, 1, 1, 2])
    model = RGCN(128, 128, 128, 3, 128)

    out = model(x, node_type, edge_index, edge_type)
