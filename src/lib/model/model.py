from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .networks.dla import DLASeg

_network_factory = {"dla": DLASeg}


def getModel(config):
    """
    Create model by config.

    Args:
        config : yacs config

    Returns:
        model : torch.nn.Module
    """
    arch = config.MODEL.ARCH
    if "_" in arch:
        num_layers = int(arch[arch.find("_") + 1 :])
        arch = arch[: arch.find("_")]
    else:
        num_layers = 0
    model_class = _network_factory[arch]
    model = model_class(num_layers, config=config)
    return model


def loadModel(model, config, optimizer=None):
    """
    Load model from checkpoint.

    Args:
        model : torch.nn.Module
        config : yacs config
        optimizer : torch.optim.Optimizer

    Returns:
        model : torch.nn.Module
        optimizer : torch.optim.Optimizer
        start_epoch : int
    """
    start_epoch = 0
    checkpoint = torch.load(
        config.MODEL.LOAD_DIR, map_location=lambda storage, _: storage
    )
    print("loaded {}, epoch {}".format(config.MODEL.LOAD_DIR, checkpoint["epoch"]))
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    # convert data_parallal model to normal model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # freeze backbone network
    if config.MODEL.FREEZE_BACKBONE:
        for name, module in model.named_children():
            if name in config.layers_to_freeze:
                for name, layer in module.named_children():
                    for param in layer.parameters():
                        param.requires_grad = False

    # resume optimizer parameters
    if optimizer is not None and config.TRAIN.RESUME:
        if "optimizer" in checkpoint:
            start_epoch = checkpoint["epoch"]
            start_lr = config.TRAIN.LR
            for step in config.TRAIN.LR_STEP:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
