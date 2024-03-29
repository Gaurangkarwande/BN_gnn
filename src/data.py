"""
Data preparation scripts for bayesian networks
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianModel
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse

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


class BNDataset(Dataset):
    """Dataset class for the bn"""

    def __init__(
        self,
        df_data: pd.DataFrame,
        target_node: str,
        bn: BayesianModel,
        adj_df: pd.DataFrame,
        noise: Optional[float] = None,
    ) -> None:
        """
        Args:
            df_data: the simulated data as a pandas dataframe
            target_node: the target node to use
            bn: the loaded bayesian network
            adj_df: the adjacency dataframe
            noise: the amount of perturbation applied to adjacency matrix
        """
        super().__init__()

        nodes = list(bn.nodes())
        target_node_id = nodes.index(target_node)
        nodes.pop(target_node_id)

        self.input_features, self.target_labels = (
            df_data[nodes].values,
            df_data[target_node].values,
        )
        self.input_nodes = nodes
        self.target_node = target_node
        self.variable_states = bn.states
        self.input_states = [self.variable_states[node] for node in self.input_nodes]
        self.target_states = self.variable_states[target_node]
        self.terminal_nodes, self.terminal_node_ids = get_terminal_connection_nodes(
            adj_df, target_node
        )

        assert len(self.terminal_nodes) > 0, "Bruh, there are no connections to the target node"
        self.num_edges = adj_df.sum().sum()
        self.adj_mat = adj_df.loc[self.input_nodes][self.input_nodes].values
        self.edge_index, self.edge_weights = dense_to_sparse(torch.as_tensor(self.adj_mat))
        self.noise = noise

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
