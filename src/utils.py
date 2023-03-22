"""
Util functions for bayesian networks
"""

from datetime import datetime
from typing import List, Tuple

import random
import numpy as np
import pandas as pd
import torch
from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.constants import GLOBAL_SEED


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
    adj_df = pd.DataFrame(
        np.zeros((num_nodes, num_nodes), dtype=float), index=node_list, columns=node_list
    )

    for edge in bn.get_edges():
        src, dst = edge
        adj_df.loc[src][dst] = 1.0

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


def encode_data(
    df_raw_data: pd.DataFrame, bn: BayesianModel
) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode raw categorical data into numerical data that can be fed into neural networks

    Args:
        df_raw_data: the raw dataframe
        bn: the loaded bayesian network

    Returns:
        pd.Dataframe: transformed numeric dataframe
        OrdinalEncoder: the encoder fit to the data
    """

    variables = list(bn.nodes)
    states = bn.states

    categories = [states[variable] for variable in variables]

    oe = OrdinalEncoder(categories=categories)
    oe.fit(df_raw_data[variables].values)
    encoded_data = oe.transform(df_raw_data[variables].values)
    return pd.DataFrame(encoded_data, columns=variables), oe


def random_upper_triangular(n, max_ones):
    # Initialize a matrix of size n x n with all elements set to zero
    random.seed(GLOBAL_SEED)

    matrix = np.zeros((n, n), dtype=int)

    # Fill in the upper triangular portion of the matrix with random binary values
    for i in range(n):
        for j in range(i, n):
            if random.random() < (max_ones / ((n * (n - 1) / 2) - i)):
                matrix[i, j] = 1

    return matrix


def perturb_adj_df(adj_df: pd.DataFrame, noise: float) -> pd.DataFrame:
    """Perturb the input adjacency dataframe

    Args:
        adj_mat: the input adjacency dataframe
        noise: the amount of noise to be added

    Returns: the perturbed adjacency matrix
    """

    if noise == 0:
        return adj_df

    np.random.seed(777)

    adj_mat = adj_df.values
    gt_edges = []
    possible_edges = []

    for i in range(adj_mat.shape[0]):
        for j in range(i+1, adj_mat.shape[0]):
            if adj_mat[i, j] == 1:
                gt_edges.append((i, j))
                adj_mat[i, j] = 0.0
            else:
                possible_edges.append((i, j))

    num_edges_to_flip = int(np.ceil(len(gt_edges) * noise))
    num_edges_retain = len(gt_edges) - num_edges_to_flip
    
    selected_edges = random.sample(gt_edges, num_edges_retain) + random.sample(possible_edges, num_edges_to_flip)

    for edge in selected_edges:
        src, dst = edge
        adj_mat[src, dst] = 1.0


    noise = 0.2
    while adj_mat[:, -1].sum() == 0:
        for i in range(adj_mat.shape[0] - 1):
            if np.random.rand() < noise:
                adj_mat[i, -1] = 1.0
        noise *= 2
    
    return pd.DataFrame(adj_mat, index=adj_df.index, columns=adj_df.columns)


def construct_adj_mat(edge_list: List[Tuple[str, str]], nodes: List[str]) -> pd.DataFrame:
    """Construct an adjacency matrix dataframe from list of edges

    Args:
        edge_list: List of edges where each element is tuple (source, destination)

    Returns: the adjacency matrix as a dataframe
    """

    adj_df = pd.DataFrame(
        np.zeros(shape=(len(nodes), len(nodes)), dtype=float), columns=nodes, index=nodes
    )
    for edge in edge_list:
        src, dst = edge
        adj_df.loc[src][dst] = 1.0
    return adj_df


def create_random_dag(
    nodes: List[str], target_node: str, prob: float, num_edges_gt: int
) -> pd.DataFrame:
    """Create a random DAG with given probability of edge

    Args:
        nodes: the list of variables in the DAG
        target_node: the target variable
        prob: the probability with which to add edges
        num_edges_gt: the number of edges in the gt structure

    Returns: a random DAG adjacency df
    """

    assert prob is not None

    num_nodes = len(nodes)
    nodes.pop(nodes.index(target_node))
    nodes.append(target_node)
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=float)

    edge_prob = (num_edges_gt * (1.0 + prob)) / (num_nodes * (num_nodes - 1) / 2)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                adj_mat[i, j] = 1.0

    if adj_mat[:, -1].sum() == 0:
        for i in range(num_nodes - 1):
            if np.random.rand() < edge_prob:
                adj_mat[i, -1] = 1.0

    return pd.DataFrame(adj_mat, index=nodes, columns=nodes)


def update_bn_model(bn: BayesianModel, adj_df: pd.DataFrame) -> BayesianModel:
    """Update the DAG structure in the bayesian model with the given adjacency matrix

    Args:
        model: the model we want to update
        adj_df: the adjacency df to infuse into the model

    Returns: the updated model
    """

    bn.clear_edges()
    nodes = list(adj_df.columns)
    edges = []

    for src in nodes:
        for dst in nodes:
            if adj_df.loc[src][dst] == 1:
                edges.append((src, dst))

    bn.add_edges_from(edges)
    return bn


def adj_diff_stats(adj_df_gt: pd.DataFrame, adj_df: pd.DataFrame) -> Tuple[int, int, int]:
    """Compare the two adjacency dataframes and return stats about their difference

    Args:
        adj_df_gt: the gt adjacency
        adj_df: the perturbed adjacency

    Returns:
        num_gt_edges_retained: the number of gt edges retained
        num_extra_edges: the number of extra edges added
        num_total_diff: the difference in number of edges in gt and perturbed
    """

    adj_mat_gt = adj_df_gt.values
    adj_mat = adj_df.values

    num_gt_edges_retained = 0
    num_extra_edges = 0

    for i in range(adj_mat_gt.shape[0]):
        for j in range(adj_mat_gt.shape[1]):
            if adj_mat_gt[i, j] == 1:
                if adj_mat[i, j] == 1:
                    num_gt_edges_retained += 1
            else:
                if adj_mat[i, j] == 1:
                    num_extra_edges += 1

    return num_gt_edges_retained, num_extra_edges, np.abs(adj_mat - adj_mat_gt).sum().sum()


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
