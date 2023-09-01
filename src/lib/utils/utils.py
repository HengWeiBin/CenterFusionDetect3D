from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import time
import logging
import torch


def createLogger(cfg, phase="train"):
    """
    This function will make output directory and creates a logger.
    They will be returned.

    Args:
        cfg: yacs config object
        phase: train or test

    Returns:
        logger: logger object
        final_output_dir: output directory
    """
    root_output_dir = Path("output")
    if not root_output_dir.exists():
        print(f"=> creating {root_output_dir}")
        root_output_dir.mkdir()

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    final_output_dir = root_output_dir / cfg.NAME / time_str
    if not final_output_dir.exists():
        print(f"=> creating {final_output_dir}")
        final_output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = final_output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{cfg.NAME}_{time_str}_{phase}.log"
    log_head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=log_head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add StreamHandler allow print to console when calling info()
    console = logging.StreamHandler()
    logger.addHandler(console)

    return logger, str(final_output_dir)


def saveModel(log, model, output_file, optimizer=None):
    """
    Save model and log to output_file

    Args:
        log: log object
        model: model object
        output_file: output file path
        optimizer: optimizer object

    Returns:
        None
    """
    if isinstance(model, torch.nn.DataParallel):
        log["state_dict"] = model.module.state_dict()
    else:
        log["state_dict"] = model.state_dict()

    if not (optimizer is None):
        log["optimizer"] = optimizer.state_dict()

    # TODO: Add epoch key
    torch.save(log, output_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
