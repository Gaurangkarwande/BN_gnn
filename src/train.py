"""
The main script for training
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from pgmpy.readwrite import BIFReader
from torch.utils.data import DataLoader

from src.constants import ALARM, ALARM_TARGET, GLOBAL_SEED, HEPAR_TARGET, HEPAR
from src.data import BNDataset
from src.models.BNNet import BNNet
from src.utils import (
    EarlyStopping,
    LRScheduler,
    adj_df_from_BIF,
    encode_data,
    get_timestamp,
    get_train_test_splits,
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
    config: Dict[str, Any],
    logger: logging.Logger,
    df_data_train: pd.DataFrame,
    df_data_val: pd.DataFrame,
    df_data_test: pd.DataFrame,
    bn: BIFReader,
    verbose: bool,
    dirpath_out: Path,
    encoder: object,
):
    start = time.time()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # create datasets

    target_node = HEPAR_TARGET
    perturbation_factor = 0.0
    adj_df = adj_df_from_BIF(bn, target_node, perturbation_factor)

    train_set = BNDataset(
        df_data=df_data_train,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        perturbation_factor=perturbation_factor,
    )
    val_set = BNDataset(
        df_data=df_data_val,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        perturbation_factor=perturbation_factor,
    )
    test_set = BNDataset(
        df_data=df_data_test,
        target_node=target_node,
        bn=bn,
        adj_df=adj_df,
        perturbation_factor=perturbation_factor,
    )

    # log network summary

    train_set.log_summary(logger)

    # create dataloaders

    dataloader_train = DataLoader(train_set, batch_size=config["batch_size_train"])
    dataloader_valid = DataLoader(val_set, batch_size=config["batch_size_val"])
    dataloader_test = DataLoader(test_set, batch_size=config["batch_size_test"])

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

    # setup paths

    fpath_loss_curve_plot = dirpath_out.joinpath("loss_curve.jpg")
    fpath_acc_curve_plot = dirpath_out.joinpath("acc_curve.jpg")
    fpath_inference_df = dirpath_out.joinpath("inference.csv")
    dirpath_checkpoint = dirpath_out.joinpath("model_checkpoint")
    dirpath_checkpoint.mkdir()
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

        train_history_loss.append(loss_train)
        train_history_acc.append(acc_train)

        val_history_loss.append(loss_valid)
        val_history_acc.append(acc_valid)

        if early_stopping.early_stop:
            break

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

    # save loss curves
    plt.plot(range(len(train_history_acc)), train_history_acc, label="Train Accuracy")
    plt.plot(range(len(val_history_acc)), val_history_acc, label="Val Accuracy")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig(fpath_acc_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

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

    logger.info(f"Final test CE loss: {loss_test}")
    logger.info(f"Final test accuracy: {acc_test}")
    logger.info(f"Total time taken: {time_taken} s")

    if verbose:
        print(f"Final test CE loss: {loss_test}")
        print(f"Final test accuracy: {acc_test}")
        print(f"Total time taken: {time_taken} s")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return loss_test, acc_test


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for training BNNet model")
    parser.add_argument("--fpath_data", type=Path, help="path to generated data", required=True)
    parser.add_argument("--fpath_bif", type=Path, help="path to bayesian network", required=True)
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
    dirpath_save = args.dirpath_results.joinpath(get_timestamp() + "training_record")
    dirpath_save.mkdir()
    df_data = pd.read_csv(args.fpath_data, dtype=str)
    bn = BIFReader(args.fpath_bif)

    with open(args.fpath_config, "r") as f:
        config = yaml.safe_load(f)
    df_data, encoder = encode_data(df_data, bn)
    df_train, df_valid, df_test = get_train_test_splits(df_data, GLOBAL_SEED, args.overfit)

    # create logger
    logging.basicConfig(
        filename=dirpath_save.joinpath(f"bn_{HEPAR}.log"),
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("bn")

    # log experiment parameters

    logger.info("Training options")
    logger.info(f"fpath_data: {str(args.fpath_data)}")
    logger.info(f"fpath_bn: {str(args.fpath_bif)}")
    logger.info(f"dirpath_save: {str(dirpath_save)}")
    logger.info(f"The config file: {config}")

    # log data information

    logger.info(f"overfit experiment: {args.overfit}")
    logger.info(f"Number of training samples: {len(df_train)}")
    logger.info(f"Number of validation samples: {len(df_valid)}")
    logger.info(f"Number of testing samples: {len(df_test)}")

    train_BN(
        config=config,
        logger=logger,
        df_data_train=df_train,
        df_data_val=df_valid,
        df_data_test=df_test,
        bn=bn,
        verbose=True,
        dirpath_out=dirpath_save,
        encoder=encoder,
    )
