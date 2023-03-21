"""
The main script for training
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from pgmpy.utils import get_example_model
from pgmpy.models import BayesianModel
from pgmpy.metrics import structure_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import GLOBAL_SEED
from src.data import BNDataset
from src.models.BNNet import BNNet
from src.utils import (
    EarlyStopping,
    LRScheduler,
    adj_diff_stats,
    encode_data,
    get_timestamp,
    get_train_test_splits,
    construct_adj_mat,
    update_bn_model,
    perturb_adj_df,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: object,
    criterion: Callable,
) -> Tuple[float, float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        optimizer: the optimizer to train the model
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        accuracy: the classification accuracy
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = correct = total = 0
    model.train()
    for batch_id, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        # find predictions
        pred = model(X)
        loss = criterion(pred, y)

        # backpropagate
        loss.backward()
        optimizer.step()

        # get metrics
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        epoch_loss += float(loss.item())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch


def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, criterion: Callable
) -> Tuple[float, float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        accuracy: the classification accuracy
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            # find predictions
            pred = model(X)
            loss = criterion(pred, y)

            # get metrics
            _, pred_classes = pred.max(1)
            total += y.size(0)
            correct += float(pred_classes.eq(y).sum().item())
            epoch_loss += float(loss.item())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch


def inference(
    model: torch.nn.Module, dataloader: DataLoader, criterion: Callable
) -> Tuple[float, float, float, List[int], List[int]]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        time: the time taken for inference
        predictions: the predicted glucose classes
        gts: the ground truth glucose classes
    """

    start = time.time()
    epoch_loss = correct = total = 0
    model.eval()
    gts = []
    predictions = []
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            # find predictions
            pred = model(X)
            loss = criterion(pred, y)

            # get metrics
            _, pred_classes = pred.max(1)
            total += y.size(0)
            correct += float(pred_classes.eq(y).sum().item())
            epoch_loss += float(loss.item())

            gts.extend(y.tolist())
            predictions.extend(pred_classes.tolist())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch, predictions, gts


def train_BN(
    bn_name: str,
    bn: BayesianModel,
    df_data_train: pd.DataFrame,
    df_data_val: pd.DataFrame,
    df_data_test: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    writer: SummaryWriter,
    verbose: bool,
    dirpath_out: Path,
    noise: Optional[float] = None,
):
    start = time.time()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # sort the causal dag

    node_list = list(nx.topological_sort(bn))
    edge_list = list(bn.edges)

    # identify target node

    target_node = node_list[-1]

    # get gt and perturbed adjacency matrix

    adj_df_gt = construct_adj_mat(edge_list=edge_list, nodes=node_list)
    adj_df = perturb_adj_df(adj_df=adj_df_gt, noise=noise)

    bn = update_bn_model(bn.copy(), adj_df)

    train_set = BNDataset(
        df_data=df_data_train,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        noise=noise,
    )
    val_set = BNDataset(
        df_data=df_data_val,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        noise=noise,
    )
    test_set = BNDataset(
        df_data=df_data_test,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        noise=noise,
    )

    # log data summary

    num_gt_edges_retained, num_extra_edges, num_total_diff = adj_diff_stats(adj_df_gt, adj_df)

    adj_dict = {
        "noise": noise,
        "num_edges_gt": adj_df_gt.values.sum().sum(),
        "num_edges_perturb": adj_df.values.sum().sum(),
        "num_gt_edges_retained": num_gt_edges_retained,
        "num_extra_edges": num_extra_edges,
        "num_total_diff_edges": num_total_diff,
    }

    train_set.log_summary(logger)

    logger.info(f"Number of edge perturb: {adj_dict['num_edges_perturb']}")
    logger.info(f"Number of gt edges retained: {adj_dict['num_gt_edges_retained']}")
    logger.info(f"Number of different edges: {adj_dict['num_total_diff']}\n")

    logger.info(f"Number of training samples: {len(df_train)}")
    logger.info(f"Number of validation samples: {len(df_valid)}")
    logger.info(f"Number of testing samples: {len(df_test)}")

    # create dataloaders

    dataloader_train = DataLoader(train_set, batch_size=config["batch_size_train"])
    dataloader_valid = DataLoader(val_set, batch_size=config["batch_size_val"])
    dataloader_test = DataLoader(test_set, batch_size=config["batch_size_test"])

    example_data = next(iter(dataloader_valid))

    # create model

    edge_index = train_set.edge_index.to(device)
    model = BNNet(
        config=config,
        num_nodes=len(train_set.input_nodes),
        node_states=train_set.input_states,
        edge_index=edge_index,
        terminal_node_ids=train_set.terminal_node_ids,
        target_node_states=train_set.target_states,
    )

    # Criterion optimizer early stopping lr_scheduler

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    early_stopping = EarlyStopping(patience=config["patience"], verbose=verbose)
    lr_scheduler = LRScheduler(optimizer, verbose=verbose)

    # transfer to GPU

    if verbose:
        if device == torch.device("cuda"):
            print("Training on GPU")
        else:
            print("No GPU available, training on CPU")

    model.to(device)

    writer.add_graph(model, example_data[0].to(device))

    # setup paths

    fpath_loss_curve_plot = dirpath_out.joinpath("loss_curve.jpg")
    fpath_acc_curve_plot = dirpath_out.joinpath("acc_curve.jpg")
    fpath_inference_df = dirpath_out.joinpath("inference.csv")
    dirpath_model_history = dirpath_out.joinpath("model_history")
    dirpath_checkpoint = dirpath_out.joinpath("model_checkpoint")
    dirpath_checkpoint.mkdir()
    dirpath_model_history.mkdir()
    fpath_checkpoint = None

    # training

    logger.info("****** Training ******")
    if verbose:
        print("****** Training ******")

    train_history_loss = []
    train_history_acc = []

    val_history_loss = []
    val_history_acc = []

    best_valid_acc = 0
    for epoch in range(config["num_epochs"]):
        loss_train, acc_train, time_train = train(
            model=model, dataloader=dataloader_train, optimizer=optimizer, criterion=criterion
        )
        loss_valid, acc_valid, _ = evaluate(
            model=model, dataloader=dataloader_valid, criterion=criterion
        )
        if verbose:
            print(
                f"Epoch {epoch}: Train Loss= {loss_train:.3f}, Train Acc= {acc_train:.3f} \t"
                f"Valid Loss= {loss_valid:.3f}, Valid Acc={acc_valid:.3f} \t"
                f" Training Time ={time_train:.2f} s"
            )
        logger.info(
            f"Epoch {epoch}: Train Loss= {loss_train:.3f}, Train Acc= {acc_train:.3f} \t"
            f"Valid Loss= {loss_valid:.3f}, Valid Acc={acc_valid:.3f} \t"
            f" Training Time ={time_train:.2f} s"
        )

        writer.add_scalar("Loss/train", loss_train, epoch)
        writer.add_scalar("Accuracy/train", acc_train, epoch)
        writer.add_scalar("Loss/valid", loss_valid, epoch)
        writer.add_scalar("Accuracy/valid", acc_valid, epoch)

        if acc_valid > best_valid_acc:
            best_valid_acc = acc_valid
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_valid_acc,
            }
            fpath_checkpoint = dirpath_checkpoint.joinpath("best_checkpoint.pth")
            torch.save(checkpoint, fpath_checkpoint)
            if verbose:
                print(f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}")
            logger.info(
                f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}"
            )
        lr_scheduler(loss_valid)
        early_stopping(loss_valid)

        writer.add_scalar("Early Stopping", early_stopping.counter, epoch)

        train_history_loss.append(loss_train)
        train_history_acc.append(acc_train)

        val_history_loss.append(loss_valid)
        val_history_acc.append(acc_valid)

        if early_stopping.early_stop:
            break
    
    writer.flush()

    logger.info(f"Final scheduler state {lr_scheduler.get_final_lr()}\n")
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # save loss curves
    plt.plot(range(len(train_history_loss)), train_history_loss, label="Train CE Loss")
    plt.plot(range(len(val_history_loss)), val_history_loss, label="Val CE Loss")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(fpath_loss_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

    # save acc curves
    plt.plot(range(len(train_history_acc)), train_history_acc, label="Train Accuracy")
    plt.plot(range(len(val_history_acc)), val_history_acc, label="Val Accuracy")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig(fpath_acc_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

    # save curves data

    np.save(
        dirpath_model_history.joinpath("train_history_loss.npy"),
        train_history_loss,
        allow_pickle=True,
    )
    np.save(
        dirpath_model_history.joinpath("train_history_acc.npy"),
        train_history_acc,
        allow_pickle=True,
    )
    np.save(
        dirpath_model_history.joinpath("val_history_loss.npy"), val_history_loss, allow_pickle=True
    )
    np.save(
        dirpath_model_history.joinpath("val_history_loss.npy"), val_history_acc, allow_pickle=True
    )

    logger.info("******* Finished Training *******")
    if verbose:
        print("******* Finished training ********")

    # inference on test set
    checkpoint = torch.load(fpath_checkpoint)
    model = BNNet(
        config=config,
        num_nodes=len(train_set.input_nodes),
        node_states=train_set.input_states,
        edge_index=edge_index,
        terminal_node_ids=train_set.terminal_node_ids,
        target_node_states=train_set.target_states,
    )
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    loss_test, acc_test, _, predicted_values, gts = inference(
        model=model, dataloader=dataloader_test, criterion=criterion
    )

    df_data_test["predicted_values"] = predicted_values
    df_data_test.to_csv(fpath_inference_df)
    time_taken = time.time() - start

    # find bif score on updated bn

    BIF_perturbed = structure_score(bn, df_data_test_BIF)

    # compile results

    results_dict = {
        "target_node": target_node,
        "loss_test": loss_test,
        "acc_test": acc_test,
        "bif_perturbed": BIF_perturbed,
    }

    logger.info(f"Final test CE loss: {loss_test}")
    logger.info(f"Final test accuracy: {acc_test}")
    logger.info(f"BIF score: {BIF_perturbed}")
    logger.info(f"Total time taken: {time_taken} s")

    if verbose:
        print(f"Final test CE loss: {loss_test}")
        print(f"Final test accuracy: {acc_test}")
        print(f"Total time taken: {time_taken} s")
        print(f"BIF score: {BIF_perturbed}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results_dict, adj_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for training BNNet model")
    parser.add_argument("--bn_name", type=str, help="name of bayesian network", required=True)
    parser.add_argument(
        "--fpath_config", type=Path, help="path to yaml config", required=False, default=None
    )
    parser.add_argument(
        "--dirpath_results",
        type=Path,
        help="path to where training runs will be recorded",
        required=True,
    )
    parser.add_argument(
        "--overfit",
        type=bool,
        help="whether to run overfit experiment on small data sample",
        default=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dirpath_save = args.dirpath_results.joinpath(get_timestamp() + "_" + args.bn_name)
    dirpath_save.mkdir()

    # get bayesian network and simulate data
    bn = get_example_model(args.bn_name)
    df_data = bn.simulate(10000, seed=GLOBAL_SEED)

    # get BIF dataframe for scoring

    _, _, df_data_test_BIF = get_train_test_splits(df_data, GLOBAL_SEED, overfit=False)

    # find BIF score on gt model

    bif_gt = structure_score(bn, df_data_test_BIF)

    with open(args.fpath_config, "r") as f:
        config = yaml.safe_load(f)

    # create logger
    logging.basicConfig(
        filename=dirpath_save.joinpath(f"bn_{args.bn_name}.log"),
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("bn")

    writer = SummaryWriter(log_dir=dirpath_save)

    # encode data
    df_data, encoder = encode_data(df_data, bn)
    df_train, df_valid, df_test = get_train_test_splits(df_data, GLOBAL_SEED, args.overfit)

    # log experiment parameters

    logger.info("Training options")
    logger.info(f"BN name: {str(args.bn_name)}")
    logger.info(f"dirpath_save: {str(dirpath_save)}")
    logger.info(f"The config file: {config}")
    logger.info(f"overfit experiment: {args.overfit}")

    writer.add_text(args.bn_name, f"Num nodes: {len(bn.nodes)}, Num GT edges: {len(bn.edges)}")

    noise = 1

    # log bn details

    logger.info("BN details")
    logger.info(f"Num nodes: {len(bn.nodes)}")
    logger.info(f"Num GT edges: {len(bn.edges)}")
    logger.info(f"Noise: {noise}")

    results_dict, adj_dict = train_BN(
        bn_name=args.bn_name,
        bn=bn,
        df_data_train=df_train,
        df_data_val=df_valid,
        df_data_test=df_test,
        config=config,
        logger=logger,
        writer=writer,
        dirpath_out=dirpath_save,
        verbose=True,
        noise=noise,
    )

    logger.info(f"BIF score GT: {bif_gt}")

    results_dict.update({'bif_gt' : bif_gt})
    writer.add_hparams(
        adj_dict,
        results_dict,
    )
    writer.flush()
