import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class APPNPConv(MessagePassing):
    """Approximate personalized propagation used as the global view encoder."""

    def __init__(self, K, alpha, normalize=True, add_self_loops=True, dropout=0.0, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            if self.add_self_loops:
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

            row, col = edge_index[0], edge_index[1]
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            edge_weight = edge_weight * norm if edge_weight is not None else norm

        h = x
        for _ in range(self.K):
            if self.training:
                edge_weight_k = edge_weight * self.dropout(torch.ones_like(edge_weight))
            else:
                edge_weight_k = edge_weight
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight_k)
            x = (1 - self.alpha) * x + self.alpha * h
        return x

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
