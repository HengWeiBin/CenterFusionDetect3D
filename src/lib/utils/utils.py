from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import time
import timeit
import logging
import torch
import matplotlib.pyplot as plt
from functools import wraps
from lightning.pytorch.utilities import rank_zero_only
import tqdm
import os
import psutil
import math
import warnings


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    @rank_zero_only
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def safe_run(func):
    """
    This decorator catches exception thrown by func and print it
    Use this decorator to avoid crash when running any function
    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Running {func.__name__} failed: {type(e).__name__} - {e}"
            logging.info(msg)
            warnings.warn(msg)

    return func_wrapper


def return_time(func):
    """
    This decorator returns the time taken by func
    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        return result, elapsed_time

    return func_wrapper


def createFolder(folder, parents=False, exist_ok=False):
    """
    This function will create a folder if it does not exist.

    Args:
        folder(PosixPath or string): folder path

    Returns:
        None
    """
    if isinstance(folder, str):
        folder = Path(folder)

    if not folder.exists():
        print(f"=> creating {folder}")
        folder.mkdir(parents=parents, exist_ok=exist_ok)


def findCheckpoint(output_dir):
    """
    This function will find the latest checkpoint in the output directory.

    2024-05-08 Updates:
    Due to sucks checkpoint loading of torch lightning, this function will only be used
    for checking the existence of checkpoint. The actual model loading will be done

    Args:
        output_dir: output directory

    Returns:
        bool: True if checkpoint exists, False otherwise
    """
    output_dir = Path(output_dir)
    checkpoints = [x for x in output_dir.glob("**/*.ckpt")]
    if not checkpoints:
        return False
    return True


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
    createFolder(root_output_dir)

    if cfg.NAME not in os.environ:
        # Find old folder if resume and exists
        final_output_dir = None
        if not cfg.EVAL and cfg.TRAIN.RESUME and cfg.MODEL.LOAD_DIR:
            model_root = os.path.split(cfg.MODEL.LOAD_DIR)[0]
            # check checkpoint in model folder, ignore if load_dir is in root folder
            final_output_dir = Path(model_root) if findCheckpoint(model_root) else None

        time_str = time.strftime("%Y-%m-%d-%H-%M")
        if final_output_dir is None:
            final_output_dir = root_output_dir / cfg.NAME / time_str
            createFolder(final_output_dir, parents=True, exist_ok=True)
        os.environ[cfg.NAME] = str(final_output_dir)

        log_dir = final_output_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{cfg.NAME}_{time_str}_{phase}.log"
        log_head = "%(asctime)-15s %(message)s"
        logging.basicConfig(filename=str(log_file), format=log_head)
    else:
        final_output_dir = Path(os.environ[cfg.NAME])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add StreamHandler allow print to console when calling info()
    logger.addHandler(TqdmLoggingHandler())

    return logger, str(final_output_dir)


def getProgressBarMessage(phase, loss_stats, avgLossStats, batch_size, current_epoch):
    """
    This function will generate a progress bar message.
    It shows the average loss of each loss type and the memory usage.

    Args:
        phase: "Train" or "Val".
        loss_stats: dictionary of loss values.
        batch_size: batch size.

    Returns:
        pbar_msg: progress bar message.
    """
    pbar_msg = f"{phase:<5} epoch {current_epoch}: "
    for lossStat in avgLossStats:
        avgLossStats[lossStat].update(loss_stats[lossStat], batch_size)
        pbar_msg += f"{avgLossStats[lossStat].avg:9.2f}"
    mem_used = psutil.virtual_memory()[3] / 1e9
    pbar_msg += f"{mem_used:9.2f}"

    return pbar_msg


def stackDictionary(dictionary):
    """
    This function will stack tensors in a dictionary along the first dimension.

    Args:
        dictionary: dictionary of tensors.

    Returns:
        dictionary: dictionary of stacked tensors.
    """
    for key in dictionary:
        if isinstance(dictionary[key], torch.Tensor):
            dictionary[key] = torch.cat(tuple(dictionary[key]), dim=0)
        elif isinstance(dictionary[key], dict):
            dictionary[key] = stackDictionary(dictionary[key])

    return dictionary


@rank_zero_only
@safe_run
def saveModel(log, model, epoch, output_file):
    """
    Save model and log to output_file

    Args:
        log: log object
        model: model object
        output_file: output file path

    Returns:
        None
    """
    if isinstance(model, torch.nn.DataParallel):
        log["state_dict"] = model.module.state_dict()
    else:
        log["state_dict"] = model.state_dict()

    log["epoch"] = epoch

    torch.save(log, output_file)


@rank_zero_only
@safe_run
def removeNan(x, y):
    new_x = []
    new_y = []

    for x_, y_ in zip(x, y):
        if not math.isnan(y_) and y_ != 0:
            new_x.append(x_)
            new_y.append(y_)

    return new_x, new_y


@rank_zero_only
@safe_run
def plotResults(logs, output_dir):
    """
    Plot training and validation loss

    Args:
        logs: list of dictionary containing loss
        output_dir: output directory

    Returns:
        None
    """
    if isinstance(logs, dict):
        logs = [logs]

    for log in logs:
        if "train" not in log or "val" not in log:
            logs.remove(log)
    if not logs:
        return

    heads = [
        "total",
        # 2D head-weights
        "heatmap",
        "widthHeight",
        "reg",
        # 3D head-weights
        "depth",
        "depth2",
        "rotation",
        "rotation2",
        "dimension",
        "amodal_offset",
        # other head-weights
        "nuscenes_att",
        "velocity",
    ]
    output_dir = Path(output_dir)
    fig = plt.figure(figsize=(25, 8))
    for log in logs:
        for i, key in enumerate(heads):
            if key not in log["train"] or key not in log["val"]:
                continue

            if isinstance(log["train"][key], list):
                x_train = range(len(log["train"][key]))
                y_train = log["train"][key]

            elif isinstance(log["train"][key], dict):
                x_train = log["train"][key].keys()
                y_train = log["train"][key].values()

            if isinstance(log["val"][key], list):
                val_interval = len(log["train"][key]) // len(log["val"][key])
                x_val = range(
                    val_interval,
                    len(log["train"][key]),
                    val_interval,
                )[: len(log["val"][key])]
                y_val = log["val"][key][: len(x_val)]

            elif isinstance(log["val"][key], dict):
                x_val = log["val"][key].keys()
                y_val = log["val"][key].values()

            plt.subplot(2, 6, i + 1)
            plt.plot(*removeNan(x_train, y_train), label="train_loss")
            plt.scatter(*removeNan(x_val, y_val), label="val_loss", marker=".")
            plt.title(key)
            # plt.xlim([140, 230])
            # plt.ylim([min(list(y_train) + list(y_val)), min(10, max(list(y_train) + list(y_val)))])
    plt.suptitle("Train Loss", fontsize="xx-large")
    fig.supylabel("loss", fontsize="x-large")
    fig.supxlabel("epoch", fontsize="x-large")
    plt.legend()
    plt.savefig(output_dir / "plot.png")

    # Plot memory usage
    fig = plt.figure()
    for log in logs:
        plt.plot(log["memory"])
    plt.title("System Memory")
    plt.xlabel("Steps")
    plt.ylabel("Memory Used (GB)")
    plt.savefig(output_dir / "memory_used.png")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


if __name__ == "__main__":
    log_files = [
        "models/CenterFusion_e230.pth",
        "models/DLA-34_DeepV4.3.3_CoordCBREmbedStepLR.pt",
        "models/DLA-34_DeepV1.8_CoordCLR.pt",
    ]
    logs = []
    for log_file in log_files:
        log = torch.load(log_file)
        if "state_dict" in log:
            log.pop("state_dict")
        logs.append(log)

    plotResults(logs, "output/")
