"""
Main model class

1. embeddings
2. gnn
3. mlp
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .GNN import GNN, GNN_nonstatic


class BNNet(nn.Module):
    """Pytoch model that does message passing on Bayesian Networks and downstream prediction"""

    def __init__(
        self,
        config: Dict[str, Any],
        num_nodes: int,
        node_states: List[List[str]],
        edge_index: torch.Tensor,
        terminal_node_ids: List[int],
        target_node_states: int,
    ) -> None:
        """

        Args:
            config: the model hyperparameters
            num_nodes: number of nodes in the graph
            node_states: the categories each node takes
            edge_index: the coo adjacency matrix
            terminal_node_ids: the ids of the nodes which are connected to the target node
            target_node_states: number of prediction classes
        """
        super().__init__()

        num_embeddings_list = [len(state) for state in node_states]
        self.node_embedding_layers = nn.ModuleList(
            [
                nn.Embedding(num_emdeddings, config["embedding_dim"])
                for num_emdeddings in num_embeddings_list
            ]
        )
        self.node_embedding_layers.requires_grad_ = False

        # self.gnn = GNN(config=config, edge_index=edge_index)
        self.gnn = GNN_nonstatic(config=config, edge_index=edge_index, num_nodes=num_nodes)

        self.MLP = nn.Sequential(
            nn.Linear(config["gnn_out_dim"] * 3 * len(terminal_node_ids), len(target_node_states)),
            nn.LeakyReLU(),
            # nn.Linear(config["fc1_out_dim"], len(target_node_states)),
        )

        self.dropout = nn.Dropout()

        self.terminal_node_ids = terminal_node_ids
        self.num_nodes = num_nodes

    def forward(self, X: torch.LongTensor) -> torch.Tensor:
        """The forward propagation method

        Args:
            X: the input tensor (batch_size x num_nodes)

        Returns: tensor with raw prediction scores (batch_size x target_node_states)
        """

        gnn_input = []

        for i, node_embedding_layer in enumerate(self.node_embedding_layers):
            gnn_input.append(node_embedding_layer(X[:, i]))

        x = torch.stack(gnn_input, dim=1)
        x = self.dropout(x)
        x = self.gnn(x)
        x = self.dropout(x)
        x = x[:, self.terminal_node_ids]
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.MLP(x)

        return x
