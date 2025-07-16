import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    """
    Example usage:
    x = torch.randn(4, 128)
    node_type = torch.tensor([0, 0, 1, 2])
    edge_index = torch.tensor([
        [0, 1, 0, 1, 3],
        [3, 3, 2, 2, 3]
        ])

    edge_type = torch.tensor([0, 0, 1, 1, 2])
    model = RGCN(128, 128, 128, 3, 128)

    out = model(x, node_type, edge_index, edge_type)
    """

    def __init__(self, in_dim, hidden_dim, num_relations=4, num_node_type=3, type_dim=20, num_layers=1):
        super(RGCN, self).__init__()
        self.activation = nn.ReLU()
        self.num_layers = num_layers
        self.num_node_type = num_node_type
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.node_emb = torch.nn.Embedding(num_node_type, type_dim)

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = in_dim + type_dim if layer == 0 else hidden_dim
            self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations))
           

    def forward(self, x, node_type, edge_index, edge_type):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        edge_type: Edge type labels [num_edges]
        """
        x = torch.cat([x, self.node_emb(node_type)], dim=-1)
        output = list()
        for id, conv in enumerate(self.convs):
            x = self.activation(conv(x, edge_index, edge_type))
            if id == 0:
                output.append(x)
        if self.num_layers != 1:
            output.append(x)
        return output #first layer, and last layer

