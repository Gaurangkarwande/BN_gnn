"""
Script for running the experiments on all
"""

import argparse
from collections import defaultdict
from typing import Any, Dict
import yaml
import logging
import pandas as pd

from pathlib import Path
from pgmpy.utils import get_example_model
from pgmpy.metrics import structure_score
from torch.utils.tensorboard import SummaryWriter

from src.utils import get_timestamp, get_train_test_splits, encode_data
from src.constants import GLOBAL_SEED
from src.train import train_BN

# BN_NAMES = ["sachs", "alarm", "hepar2", "diabetes"] # first four
BN_NAMES = ["asia", "cancer", "child", "water", "hailfinder", "munin"]
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def my_custom_logger(logger_name, logger_fpath, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    message_format_string = "%(asctime)s %(message)s"
    time_format_string = "%m/%d/%Y %H:%M:%S"
    log_format = logging.Formatter(message_format_string, time_format_string)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_fpath)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def run_exp_for(bn_name: str, dirpath_save: Path, config: Dict[str, Any]) -> None:
    """Run experiment for the given BN

    Args:
        bn_name: the bn name
        dirpath_save: the path to where results wil be save
        config: the training config

    """

    dirpath_save = dirpath_save.joinpath(bn_name)
    dirpath_save.mkdir()
    inference_dict = defaultdict(list)

    # get bayesian network and simulate data
    bn = get_example_model(bn_name)
    df_data = bn.simulate(10000, seed=GLOBAL_SEED)

    # get BIF dataframe for scoring

    _, _, df_data_test_BIF = get_train_test_splits(df_data, GLOBAL_SEED, overfit=False)

    # find BIF score on gt model

    bif_gt = structure_score(bn, df_data_test_BIF)
    # create logger

    logger = my_custom_logger(
        logger_name=bn_name,
        logger_fpath=dirpath_save.joinpath(f"bn_{bn_name}.log"),
        level=logging.INFO,
    )

    writer = SummaryWriter(log_dir=dirpath_save)

    # encode data
    df_data, encoder = encode_data(df_data, bn)
    df_train, df_valid, df_test = get_train_test_splits(df_data, GLOBAL_SEED, overfit=False)

    # log experiment parameters

    logger.info(f"BN name: {str(bn_name)}")
    logger.info(f"Num nodes: {len(bn.nodes)}")
    logger.info(f"Num GT edges: {len(bn.edges)}")

    for noise in NOISE_LEVELS:
        dirpath_out = dirpath_save.joinpath(f"noise_{noise}")
        dirpath_out.mkdir()

        results_dict, adj_dict = train_BN(
            bn_name=bn_name,
            bn=bn,
            df_data_train=df_train,
            df_data_val=df_valid,
            df_data_test=df_test,
            df_data_test_BIF=df_data_test_BIF,
            config=config,
            logger=logger,
            writer=writer,
            dirpath_out=dirpath_out,
            verbose=True,
            noise=noise,
        )
        logger.info(f"BIF score GT: {bif_gt}")
        logger.info("\n\n**********************************\n\n")
        target_node = results_dict.pop("target_node")
        results_dict.update({"bif_gt": bif_gt})
        writer.add_hparams(adj_dict, results_dict)
        writer.flush()
        results_dict["target_node"] = target_node
        results_dict.update(adj_dict)

        for key, val in results_dict.items():
            inference_dict[key].append(val)

    df_inference = pd.DataFrame.from_dict(inference_dict)
    df_inference.to_csv(dirpath_save.joinpath(f"{bn_name}_inference.csv"), index=None)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parser for running thesis experiments BNNet model"
    )
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
    dirpath_save = args.dirpath_results.joinpath(get_timestamp())
    dirpath_save.mkdir()

    with open(args.fpath_config, "r") as f:
        config = yaml.safe_load(f)

    for bn_name in BN_NAMES:
        run_exp_for(bn_name=bn_name, dirpath_save=dirpath_save, config=config)
