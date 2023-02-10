"""
Util functions for bayesian networks
"""

from typing import List, Tuple
from pgmpy.readwrite import BIFReader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def adj_df_from_BIF(bn: BIFReader) -> pd.DataFrame:
    """Create adjacency matrix form BIF file

    Args:
        bn: the loaded bayesian network

    Returns: the adjacency matrix as a dataframe
    """

    node_list = bn.get_variables()
    num_nodes = len(node_list)
    node_dict = {node: idx for idx, node in enumerate(node_list)}
    adj_mat = np.zeros((num_nodes, num_nodes))

    for edge in bn.get_edges():
        source, destination = edge
        source_id, destination_id = node_dict[source], node_dict[destination]
        adj_mat[source_id, destination_id] = 1.0

    adj_df = pd.DataFrame(adj_mat, index=node_list, columns=node_list)

    return adj_df


def get_terminal_connection_nodes(bn: BIFReader, target: str) -> Tuple[List[str], List[int]]:
    """Return the edge connnections to the target node

    Args:
        bn: the loaded bayesian network
        target: the name of the target node/variable

    Returns: the tuple of the node names and corresponding indices
    """

    node_list = bn.get_variables()
    node_dict = {node: idx for idx, node in enumerate(node_list)}
    nodes, node_ids = list(), list()

    for edge in bn.get_edges():
        source, destination = edge
        if destination == target:
            source_id = node_dict[source]
            nodes.append(source)
            node_ids.append(source_id)

    return nodes, node_ids


def encode_data(df_raw_data: pd.DataFrame, bn: BIFReader) -> Tuple[pd.DataFrame, OrdinalEncoder] :
    """Encode raw categorical data into numerical data that can be fed into neural networks

    Args:
        df_raw_data: the raw dataframe
        bn: the loaded bayesian network
    
    Returns:
        pd.Dataframe: transformed numeric dataframe
        OrdinalEncoder: the encoder fit to the data
    """

    variables = bn.get_variables()
    states = bn.get_states()

    assert list(states.keys()) == variables

    oe = OrdinalEncoder(categories=list(states.values()))
    oe.fit(df_raw_data[variables].values)
    encoded_data = oe.transform(df_raw_data[variables].values)
    return pd.DataFrame(encoded_data, columns=variables), oe
