import torch
from torch_geometric.nn import RGCNConv
import torch.nn as nn


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers=3):
        super(RGCN, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        edge_type: Edge type labels [num_edges]
        """
        for conv in self.convs[:-1]:
            x = self.activation(conv(x, edge_index, edge_type))
        x = self.convs[-1](x, edge_index, edge_type)
        return x


if __name__ == "__main__":
    x = torch.randn(4, 128)
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    edge_type = torch.tensor([0, 0, 1, 1, 2])
    model = RGCN(128, 128, 128, 3)

    out = model(x, edge_index, edge_type)
    print(out.shape)
    print(out)
