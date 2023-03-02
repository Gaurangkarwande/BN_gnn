"""
This file defines the gnn model
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import remove_self_loops


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


class GNN_nonstatic(nn.Module):
    """The gnn model"""

    def __init__(
        self, config: Dict[str, Any], edge_index: Tensor, num_nodes: int, inference: bool
    ) -> None:
        super().__init__()

        num_heads = config["gat_heads"]
        self.layer1 = GATv2Conv(
            config["embedding_dim"], config["gnn_hidden_dim"], heads=num_heads, concat=True
        )
        self.layer2 = GATv2Conv(
            config["gnn_hidden_dim"] * num_heads, config["gnn_out_dim"], concat=False
        )

        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.edge_weights = None
        self.inference = inference

    def forward(self, x: Tensor) -> Tensor:
        """The forward propagation method

        Args:
            X: the input node embeddings (batch_size x num_nodes x embedding_dim)

        Returns: transformed node embeddings (batch_size x num_nodes x gnn_out_dim)
        """

        batch_edge_index = self.edge_index
        batch_size = x.shape[0]
        for i in range(1, batch_size):
            batch_edge_index = torch.cat(
                (batch_edge_index, self.edge_index + i * self.num_nodes), 1
            )

        x = x.view(batch_size * self.num_nodes, -1)

        x, (_, batch_edge_attr) = self.layer1(
            x,
            edge_index=batch_edge_index,
            return_attention_weights=True,
        )

        x, (temp, batch_edge_attr) = self.layer2(
            x,
            edge_index=batch_edge_index,
            return_attention_weights=True,
        )
        _, batch_edge_attr = remove_self_loops(temp, batch_edge_attr)

        self.batch_edge_attr = batch_edge_attr
        self.batch_edge_index = batch_edge_index

        edge_weights = batch_edge_attr.view(batch_size, -1)

        if self.inference:
            if self.edge_weights is not None:
                self.edge_weights = torch.cat((self.edge_weights, edge_weights))
            else:
                self.edge_weights = edge_weights

        return x.view(batch_size, self.num_nodes, -1)
