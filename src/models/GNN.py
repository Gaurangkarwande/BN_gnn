"""
This file defines the gnn model
"""

from typing import Any, Dict

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    """The gnn model"""

    def __init__(self, config: Dict[str, Any], edge_index: Tensor) -> None:
        super().__init__()

        self.layer1 = GCNConv(config["embedding_dim"], config["gnn_hidden_dim"])
        self.layer2 = GCNConv(config["gnn_hidden_dim"], config["gnn_out_dim"])

        self.edge_index = edge_index

    def forward(self, x: Tensor) -> Tensor:
        """The forward propagation method

        Args:
            X: the input node embeddings (batch_size x num_nodes x embedding_dim)

        Returns: transformed node embeddings (batch_size x num_nodes x gnn_out_dim)
        """

        x = self.layer1(x, self.edge_index)
        x = self.layer2(x, self.edge_index)

        return x
