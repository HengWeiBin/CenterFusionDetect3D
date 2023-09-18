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

from utils import createLogger, saveModel, plotResults
from config import config, updateConfig, updateDatasetAndModelConfig
from dataset import getDataset
from model import getModel, loadModel
from trainer import Trainer

try:
    import wandb
except ImportError:
    wandb = None


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


def getOptimizer(config, model):
    if config.TRAIN.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
    elif config.TRAIN.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), config.TRAIN.LR, momentum=0.9, weight_decay=0.0001
        )
    else:
        assert 0, config.TRAIN.OPTIMIZER
    return optimizer


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
            f"Not enough available gpu! {torch.cuda.device_count()} < {len(config.GPUS)}"
        )
        raise RuntimeError(
            f"Not enough available gpu! {torch.cuda.device_count()} < {len(config.GPUS)}"
        )
    if len(config.GPUS):
        device = torch.device("cuda")
        torch.multiprocessing.set_start_method("spawn")
    else:
        device = torch.device("cpu")

    dataset = getDataset(config.DATASET.DATASET)
    updateDatasetAndModelConfig(config, dataset, output_dir)
    logger.info(config)

    model = getModel(config)
    optimizer = getOptimizer(config, model)
    start_epoch = 0
    log = {}
    lr = config.TRAIN.LR
    if config.MODEL.LOAD_DIR != "":
        log, model, optimizer, start_epoch = loadModel(model, config, optimizer)
    trainer = Trainer(config, model, optimizer)
    trainer.setDevice(config)
    log.update({"memory": []})

    val_loader = torch.utils.data.DataLoader(
        dataset(config, config.DATASET.VAL_SPLIT, device),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
    )

    if config.EVAL:
        # Run evaluation only
        with torch.no_grad():
            preds = trainer.val(1, val_loader, logger, log)
        val_loader.dataset.run_eval(preds, output_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        dataset(config, config.DATASET.TRAIN_SPLIT, device),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        drop_last=True,
    )

    for epoch in range(start_epoch + 1, config.TRAIN.EPOCHS + 1):
        # train
        trainer.train(epoch, train_loader, logger, log)

        # save model
        if config.TRAIN.SAVE_INTERVALS > 0 and epoch % config.TRAIN.SAVE_INTERVALS == 0:
            saveModel(log, model, epoch, os.path.join(output_dir, f"model_{epoch}.pt"), optimizer)
        saveModel(log, model, epoch, os.path.join(output_dir, "model_last.pt"), optimizer)

        # validation
        if config.TRAIN.VAL_INTERVALS > 0 and epoch % config.TRAIN.VAL_INTERVALS == 0:
            with torch.no_grad():
                preds = trainer.val(epoch, val_loader, logger, log)

            # evaluate using dataset-official scripts
            if config.TEST.OFFICIAL_EVAL:
                val_loader.dataset.run_eval(preds, output_dir)

                # log validation result
                with open(
                    os.path.join(
                        output_dir,
                        f"nuscenes_eval_det_output_{config.DATASET.VAL_SPLIT}",
                        "metrics_summary.json",
                    ),
                    "r",
                ) as f:
                    metrics = json.load(f)
                logger.info(f'AP/overall: {metrics["mean_ap"]*100.0}%')

                for k, v in metrics["mean_dist_aps"].items():
                    print(f"AP/{k}: {v * 100.0}%")

                for k, v in metrics["tp_errors"].items():
                    print(f"Scores/{k}: {v}")

                logger.info(f'Scores/NDS: {metrics["nd_score"]}')

        # adjust learning rate
        if epoch in config.TRAIN.LR_STEP:
            lr = config.TRAIN.LR * (0.1 ** (config.TRAIN.LR_STEP.index(epoch) + 1))
            logger.info(f"Changing learning rate to {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    plotResults(log, output_dir)
    logger.info("Done")


if __name__ == "__main__":
    parse_args()
    if wandb is not None:
        wandb.init(
            config=config,
            resume="allow",
            project="CenterFusionDetect3D",
            name=config.NAME,
            job_type="train" if not config.EVAL else "eval",
        )
    else:
        print("wandb is not installed, wandb logging is disabled")

    main()
    if wandb is not None:
        wandb.finish()
