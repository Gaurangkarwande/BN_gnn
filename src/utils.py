"""
Util functions for bayesian networks
"""

from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pgmpy.readwrite import BIFReader
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def adj_df_from_BIF(bn: BIFReader, target: str, perturbation_factor: float = 0) -> pd.DataFrame:
    """Create adjacency matrix form BIF file

    Args:
        bn: the loaded bayesian network
        target: the name of the target node/variable
        perturbation_factor: the probabiliy that a given edge may be changed. May be deleted or
            added.

    Returns: the adjacency matrix as a dataframe
    """

    node_list = bn.get_variables()
    node_list.pop(node_list.index(target))
    node_list.append(target)

    num_nodes = len(node_list)
    node_dict = {node: idx for idx, node in enumerate(node_list)}
    adj_mat = np.zeros((num_nodes, num_nodes))

    for edge in bn.get_edges():
        source, destination = edge
        source_id, destination_id = node_dict[source], node_dict[destination]
        adj_mat[source_id, destination_id] = 1.0

    adj_df = pd.DataFrame(adj_mat, index=node_list, columns=node_list)
    adj_df = perturb_adj_df(adj_df, perturbation_factor)

    return adj_df


def get_terminal_connection_nodes(adj_df: pd.DataFrame, target: str) -> Tuple[List[str], List[int]]:
    """Return the edge connnections to the target node

    Args:
        bn: the loaded bayesian network
        target: the name of the target node/variable

    Returns: the tuple of the node names and corresponding indices
    """

    nodes, node_ids = list(), list()

    node_id = 0
    for index, value in adj_df[target].items():
        if value == 1:
            nodes.append(index)
            node_ids.append(node_id)
        node_id += 1

    return nodes, node_ids


def encode_data(df_raw_data: pd.DataFrame, bn: BIFReader) -> Tuple[pd.DataFrame, OrdinalEncoder]:
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


def perturb_adj_df(adj_df: pd.DataFrame, perturbation_factor: float) -> pd.DataFrame:
    """Perturb the input adjacency dataframe

    Args:
        adj_mat: the input adjacency dataframe
        perturbation_factor: the probabiliy that a given edge may be changed. May be deleted or
            added.

    Returns: the perturbed adjacency matrix
    """

    if perturbation_factor == 0:
        return adj_df

    pmf = bernoulli(perturbation_factor)

    def flip_edge(edge: int) -> int:
        if edge:
            return float(False)
        flip = pmf.rvs(1)
        return float(not edge) if flip else edge

    adj_df = adj_df.apply(np.vectorize(flip_edge))
    np.fill_diagonal(adj_df.values, 0.0)
    return adj_df


def get_train_test_splits(
    df_data: pd.DataFrame, seed: int, overfit: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into training, validation, and testing splits

    Args:
        df_data: the data
        seed: seed to be used for sampling and splitting
        overfit: whether to sample a smaller dataset for overfitting experiment

    Returns: the train, valid, and test datasets
    """

    if overfit:
        df_data = df_data.sample(n=40, random_state=seed)

    df_train, df_valid = train_test_split(df_data, test_size=0.4, random_state=seed)
    df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=seed)

    return df_train, df_valid, df_test


def get_timestamp() -> str:
    """Returns current timestamp to append to files

    Returns: the timestamp string
    """
    current_timestamp = datetime.now()
    processed_timestamp = (
        str(current_timestamp)[:-7].replace(" ", "_").replace(":", "").replace("-", "") + "_"
    )

    return processed_timestamp


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=4, min_lr=1e-6, factor=0.3, verbose=True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        :verbose: whether to print details
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=verbose,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

    def get_final_lr(self):
        return self.lr_scheduler.state_dict()


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=11, min_delta=1e-5, verbose=True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :verbose: whether to print details
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("INFO: Early stopping")
                self.early_stop = True
