"""
Data preparation scripts for bayesian networks
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from torch.utils.data import Dataset

from src.constants import alarm_target
from src.utils import adj_df_from_BIF, get_terminal_connection_nodes


def load_data(fpath_data: Path, fpath_network: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the input and targets for alarm dataset

    Args:
        fpath_data: path to the data generated from BN
        fpath_network: path to the BN

    Returns: the input features and the target variables
    """

    reader = BIFReader(fpath_network)
    nodes = reader.get_variables()
    target_node_id = nodes.index(alarm_target)
    nodes.pop(target_node_id)
    df_data = pd.read_csv(fpath_data)
    X, y = df_data[nodes].values, df_data[alarm_target].values
    return X, y


class AlarmDataset(Dataset):
    """Dataset class for the alarm bn"""

    def __init__(self, df_data: pd.DataFrame, bn: BIFReader) -> None:
        """
        Args:
            df_data: the simulated data as a pandas dataframe
            bn: the loaded bayesian network
        """
        super().__init__()

        nodes = bn.get_variables()
        target_node_id = nodes.index(alarm_target)
        nodes.pop(target_node_id)
        adj_df = adj_df_from_BIF(bn)

        self.input_features, self.target_labels = (
            df_data[nodes].values,
            df_data[alarm_target].values,
        )
        self.input_nodes = nodes
        self.target_node = alarm_target
        self.variable_states = bn.get_states()
        self.input_states = [self.variable_states[node] for node in self.input_nodes]
        self.target_states = self.variable_states[alarm_target]
        self.terminal_nodes, self.terminal_node_ids = get_terminal_connection_nodes(
            bn, alarm_target
        )
        self.adj_mat = adj_df.loc[self.input_nodes][self.input_nodes].values

    def __len__(self) -> int:
        return len(self.target_labels)

    def __getitem__(self, idx):
        return self.input_features[idx], self.target_labels[idx]
