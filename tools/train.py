from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths  # initialize paths
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import json
import os
from utils import createLogger, saveModel
from config import config, updateConfig, updateDatasetAndModelConfig
from dataset import getDataset


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

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    if torch.cuda.device_count() < len(config.GPUS):
        logger.critical(
            f"Not enough available gpu! {torch.cuda.device_count()} < {len(config.gpus)}"
        )
        raise RuntimeError(
            f"Not enough available gpu! {torch.cuda.device_count()} < {len(config.gpus)}"
        )
    if len(config.GPUS):
        device = torch.device("cuda")
        torch.multiprocessing.set_start_method("spawn")
    else:
        device = torch.device("cpu")

    dataset = getDataset(config.DATASET.DATASET)
    updateDatasetAndModelConfig(config, dataset)
    logger.info(config)

    model = None  # TODO: create_model(config)
    optimizer = None  # TODO: get_optimizer(config, model)
    start_epoch = 0
    lr = config.TRAIN.LR
    if config.MODEL.LOAD_DIR != "":
        pass
        # TODO: load_model
        # model, optimizer, start_epoch = load_model(
        #     model, config.MODEL.LOAD_DIR, config, optimizer)
    trainer = None  # TODO: Trainer(config, model, optimizer)

    val_loader = torch.utils.data.DataLoader(
        dataset(config, config.DATASET.VAL_SPLIT, device),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    if config.EVAL:
        # TODO: eval
        return

    train_loader = torch.utils.data.DataLoader(
        dataset(config, config.DATASET.TRAIN_SPLIT, device),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    log = dict()

    for _ in train_loader:
        pass

    for epoch in range(start_epoch + 1, config.TRAIN.EPOCHS + 1):
        # train
        trainer.train(epoch, train_loader, logger, log)

        # validation
        if config.TRAIN.VAL_INTERVALS > 0 and epoch % config.TRAIN.VAL_INTERVALS == 0:
            with torch.no_grad():
                preds = trainer.val(epoch, val_loader, logger, log)

            # evaluate using dataset-official scripts
            if config.TEST.OFFICIAL_EVAL:
                val_loader.dataset.run_eval(preds, output_dir)

                with open(os.path.join(output_dir, "metrics_summary.json"), "r") as f:
                    metrics = json.load(f)
                logger.info(f'AP/overall: {metrics["mean_ap"]*100.0}')

                for k, v in metrics["mean_dist_aps"].items():
                    logger.info(f"AP/{k}: {v * 100.0}")

                for k, v in metrics["tp_errors"].items():
                    logger.info(f"Scores/{k}: {v}")

                logger.info(f'Scores/NDS: {metrics["nd_score"]}')

        # save model
        if epoch in config.TRAIN.SAVE_POINT:
            saveModel(log, model, os.path.join(output_dir, f"model_{epoch}.pt"))
        saveModel(log, model, os.path.join(output_dir, "model_last.pt"))

        # adjust learning rate
        if epoch in config.TRAIN.LR_STEP:
            lr = config.TRAIN.LR * (0.1 ** config.TRAIN.LR_STEP.index(epoch) + 1)
            logger.info(f"Changing learning rate to {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    logger.info("Done")


if __name__ == "__main__":
    parse_args()
    main()
