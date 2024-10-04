from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths  # add lib to PYTHONPATH
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

from utils import createLogger, plotResults
from config import config, updateConfig, updateDatasetAndModelConfig
from dataset import getDataset
from model import getModel, loadModel
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Center Fusion 3D Object Detection network"
    )

    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    updateConfig(config, args)


def main():
    logger, output_dir = createLogger(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.enabled = config.CUDNN.ENABLED
    if config.GPUS[0] != -1 and torch.cuda.device_count() < len(config.GPUS):
        errorMsg = f"Not enough available gpu! {torch.cuda.device_count()} < {len(config.GPUS)}"
        logger.critical(errorMsg)
        raise RuntimeError(errorMsg)

    # Reduce some precision to speed up training if GPU is Ampere
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")

    dataset = getDataset(config.DATASET.DATASET)
    updateDatasetAndModelConfig(config, dataset, output_dir)
    logger.info(config)

    model = getModel(config)
    start_epoch = 1
    log = {}
    if config.MODEL.LOAD_DIR != "":
        log, model, start_epoch = loadModel(model, config)
    log.update({"memory": []})
    val_dataset = dataset(config, config.DATASET.VAL_SPLIT)
    trainer = Trainer(config, model, logger, log, output_dir, val_dataset, start_epoch)

    # Calculate number of parameters
    total_params = {"frozen": 0, "trainable": 0}
    name_params = {"head": 0, "backbone": 0, "neck": 0, "combiner": 0}
    for name, param in model.named_parameters():
        name = name.lower()
        # Count number of parameters in each part of the model
        # if "head" in name or "combiner" in name:
        if any([part in name for part in ["head", "combiner", "base"]]):
            if "head" in name:
                name_params["head"] += param.numel()
            if "combiner" in name:
                name_params["combiner"] += param.numel()
            if "base" in name:
                name_params["backbone"] += param.numel()
        else:
            name_params["neck"] += param.numel()

        # Count number of trainable and frozen parameters
        if param.requires_grad:
            total_params["trainable"] += param.numel()
        else:
            total_params["frozen"] += param.numel()

    logger.info(model)
    logger.info("Number of parameters:")
    logger.info(f"{' '*4}{'Backbone:':<10}{name_params['backbone'] * 1e-6:>5.2f}M")
    logger.info(f"{' '*4}{'Neck:':<10}{name_params['neck'] * 1e-6:>5.2f}M")
    logger.info(f"{' '*4}{'Head:':<10}{name_params['head'] * 1e-6:>5.2f}M")
    logger.info(f"{' '*4}{'Combiner:':<10}{name_params['combiner'] * 1e-6:>5.2f}M")
    logger.info(f"{' '*4}{'Total:':<10}{sum(total_params.values()) * 1e-6:>5.2f}M")
    logger.info(f"{' '*4}{'Trainable:':<10}{total_params['trainable'] * 1e-6:>5.2f}M")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE // max(len(config.GPUS), 1),
        shuffle=False,
        num_workers=config.WORKERS if config.EVAL else config.WORKERS // 2,
        pin_memory=True,
    )

    if config.EVAL:
        # Run evaluation only
        if "test" in config.DATASET.VAL_SPLIT:
            trainer.test(val_loader)
            return

        trainer.val(val_loader)
        return

    train_loader = torch.utils.data.DataLoader(
        dataset(config, config.DATASET.TRAIN_SPLIT),
        batch_size=config.TRAIN.BATCH_SIZE // max(len(config.GPUS), 1),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        drop_last=True,
        pin_memory=True,
    )

    trainer.train(train_loader, val_loader)
    plotResults(log, output_dir)
    logger.info("Done")


if __name__ == "__main__":
    parse_args()
    main()
