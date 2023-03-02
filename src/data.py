"""
Data preparation scripts for bayesian networks
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pgmpy.readwrite import BIFReader
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from src.utils import get_terminal_connection_nodes


def load_data(
    fpath_data: Path, target_node: str, fpath_network: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the input and targets for bn dataset

    Args:
        fpath_data: path to the data generated from BN
        target_node: the target node to use
        fpath_network: path to the BN

    Returns: the input features and the target variables
    """

    reader = BIFReader(fpath_network)
    nodes = reader.get_variables()
    target_node_id = nodes.index(target_node)
    nodes.pop(target_node_id)
    df_data = pd.read_csv(fpath_data)
    X, y = df_data[nodes].values, df_data[target_node].values
    return X, y


def reconstruct_adj_mats(
    input_edge_weights: torch.Tensor,
    terminal_edge_weights: torch.Tensor,
    input_edge_index: torch.Tensor,
    terminal_node_ids: List[int],
    node_list: List[str],
) -> torch.Tensor:
    """Reconstruct the adjecency matrices from edge index and terminal nodes and return list of
        adj_matrices

    Args:
        input_edge_weights: edge weights of the input edges
        terminal_edge_weights: edge weights of the terminal edges
        input_edge_index: the edge_index without the target node
        terminal_node_ids: the node ids directly connected to the target node
        node_list: the list of variable names with the last one being the target ndoe

    Returns: adjacency matrices for each sample
    """

    terminal_node_id = len(node_list) - 1
    terminal_edge_index = []

    for node_id in terminal_node_ids:
        terminal_edge_index.append([node_id, terminal_node_id])

    terminal_edge_index = torch.as_tensor(terminal_edge_index).T
    # print(input_edge_index.shape, terminal_edge_index.shape)
    edge_index = torch.concat((input_edge_index, terminal_edge_index), dim=1).cpu()
    edge_weights = torch.concat((input_edge_weights, terminal_edge_weights), dim=1).cpu()

    adj_mats = []

    for e in edge_weights:
        adj_mats.append(to_dense_adj(edge_index, edge_attr=e))

    return torch.stack(adj_mats).squeeze()


class BNDataset(Dataset):
    """Dataset class for the bn"""

    def __init__(
        self,
        df_data: pd.DataFrame,
        target_node: str,
        bn: BIFReader,
        adj_df: pd.DataFrame,
        perturbation_factor: float = 0,
    ) -> None:
        """
        Args:
            df_data: the simulated data as a pandas dataframe
            target_node: the target node to use
            bn: the loaded bayesian network
            adj_df: the adjacency dataframe
            perturbation_factor: the amount of perturbation applied to adjacency matrix
        """
        super().__init__()

        nodes = bn.get_variables()
        target_node_id = nodes.index(target_node)
        nodes.pop(target_node_id)

        self.input_features, self.target_labels = (
            df_data[nodes].values,
            df_data[target_node].values,
        )
        self.input_nodes = nodes
        self.target_node = target_node
        self.variable_states = bn.get_states()
        self.input_states = [self.variable_states[node] for node in self.input_nodes]
        self.target_states = self.variable_states[target_node]
        self.terminal_nodes, self.terminal_node_ids = get_terminal_connection_nodes(
            adj_df, target_node
        )
        self.adj_mat = adj_df.loc[self.input_nodes][self.input_nodes].values
        self.edge_index, self.edge_weights = dense_to_sparse(torch.as_tensor(self.adj_mat))
        self.perturbation_factor = perturbation_factor

    def __len__(self) -> int:
        return len(self.target_labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.as_tensor(self.input_features[idx], dtype=torch.long), torch.as_tensor(
            self.target_labels[idx], dtype=torch.long
        )

    def log_summary(self, logger: logging.Logger) -> None:
        """Log the bayesian network details

        Args:
            logger: the logger to log into
        """

        logger.info(f"Number of input variables : {len(self.input_nodes)}")
        logger.info(f"Target node name : {self.target_node}")
        logger.info(
            f"Terminal nodes : {self.terminal_nodes}, Terminal node ids : {self.terminal_node_ids}"
        )
        logger.info(f"ADJACENCY MATRIX - Number of edges : {len(self.edge_weights)}")
        logger.info(f"ADJACENCY MATRIX - Peturbation from gt : {self.perturbation_factor}")
