from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import re
import logging
import glob

from .networks.dla import DLASeg

_network_factory = {
    "dla": DLASeg,
}
EARLY_FUSION = ["early"]


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
        num_layers = arch[arch.find("_") + 1 :]
        arch = arch[: arch.find("_")]
    else:
        num_layers = 0

    in_channels = 3
    if config.DATASET.RADAR_PC and config.MODEL.FUSION_STRATEGY in EARLY_FUSION:
        nPcChannels = 3 * max(
            1, config.DATASET.ONE_HOT_PC * int(config.DATASET.MAX_PC_DIST)
        )
        in_channels = 3 + nPcChannels

    model_class = _network_factory[arch]
    model = model_class(num_layers, in_channels=in_channels, config=config)
    return model


def loadStateDictMapper():
    mapFiles = glob.glob("src/lib/model/weightsMapper/*.csv")
    mapper = {}  # olds to new
    for file in mapFiles:
        with open(file, "r") as f:
            lines = f.readlines()
            mapper.update(dict(line.strip().split(",") for line in lines))

    return mapper


def elasticLoadStateDict(model, state_dict):
    """
    Load state_dict to model with elastic transform.

    Args:
        model : torch.nn.Module
        state_dict : dict

    Returns:
        model : torch.nn.Module
    """
    logging.info("Performing elastic load.")
    mapper = loadStateDictMapper()

    # convert data_parallal model to normal model
    state_dict_ = state_dict
    state_dict = {}
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    finalStateDict = {}
    for k in state_dict:
        if k in mapper:
            newK = mapper[k]
        else:
            newK = toggleWeightName(k, to="new")
            newK = newK if newK in model_state_dict else k
        if newK in model_state_dict:
            if state_dict[k].shape != model_state_dict[newK].shape:
                logging.info(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[newK].shape, state_dict[k].shape
                    )
                )
                finalStateDict[newK] = model_state_dict[newK]
            else:
                finalStateDict[newK] = state_dict[k]
        else:
            logging.info(f"Drop parameter {k}.")

    # load weights with non-strick mode if possible
    for k in model_state_dict:
        oldK = toggleWeightName(k, to="old")
        oldKv2 = toggleWeightName(k, to="oldv2")
        if k not in state_dict and oldK not in state_dict and oldKv2 not in state_dict:

            # Check if the key was paired by mapper
            foundMapped = False
            for oldKMap, newKMap in mapper.items():
                if newKMap == k and oldKMap in state_dict:
                    foundMapped = True
                    break
            if foundMapped:
                continue

            logging.info("No param {}.".format(k))
            finalStateDict[k] = model_state_dict[k]

            # Activate the layer if it is not in the checkpoint
            attrs = k.split(".")[:-1]
            targetLayer = model
            for attr in attrs:
                targetLayer = getattr(targetLayer, attr)
            targetLayer.requires_grad = True

    logging.info(model.load_state_dict(finalStateDict, strict=False))

    return model


def loadModel(model, config):
    """
    Load model from checkpoint.

    Args:
        model : torch.nn.Module
        config : yacs config

    Returns:
        checkpoint (dict) : model log
        model : torch.nn.Module
        start_epoch : int
    """
    checkpoint = torch.load(
        config.MODEL.LOAD_DIR, map_location=lambda storage, _: storage
    )
    start_epoch = 1
    if "epoch" in checkpoint and config.TRAIN.RESUME:
        start_epoch = checkpoint["epoch"] + 1
    print("loaded {}, epoch {}".format(config.MODEL.LOAD_DIR, checkpoint["epoch"]))

    if checkpoint["state_dict"].keys() != model.state_dict().keys():
        model = elasticLoadStateDict(model, checkpoint["state_dict"])
    else:
        print(model.load_state_dict(checkpoint["state_dict"], strict=False))

    checkpoint = renewCheckpoint(checkpoint)

    return checkpoint, model, start_epoch


def toggleWeightName(name, to="new"):
    """
    Transform the name of pretrained weights to new or old model name.

    Args:
        name : str
        to : str
            "new" or "old"

    Returns:
        name : str
    """
    if to not in ["new", "old", "oldv1", "oldv2"]:
        raise ValueError("to must be new or old")

    heads = [
        "reg",
        "depth2",
        "rotation2",
        "heatmap",
        "widthHeight",
        "depth",
        "rotation",
        "dimension",
        "amodal_offset",
        "activation",
        "nuscenes_att",
        "velocity",
    ]
    oldToNew = {
        "dep_sec.": "detectHead_0.depth2.",
        "rot_sec.": "detectHead_0.rotation2.",
        "hm.": "detectHead_0.heatmap.",
        "wh.": "detectHead_0.widthHeight.",
        "dep.": "detectHead_0.depth.",
        "dim.": "detectHead_0.dimension.",
        "rot.": "detectHead_0.rotation.",
        "amodel_offset.": "detectHead_0.amodal_offset.",
        "actf": "activation",
        "conv.conv_offset_mask": "conv_offset_mask",
    }
    oldUpNodeRegex = ".*_up.*_\d.conv.(weight|bias)"
    newUpNodeRegex = ".*_up.*_\d.(weight|bias)"

    newToOld = {v: k for k, v in oldToNew.items()}
    toggleDict = oldToNew if to == "new" else newToOld
    if to == "oldv2":
        for head in heads:
            toggleDict[f"detectHead_0.{head}."] = f"{head}."

    # return name if it is already a new name
    if to == "new":
        # check name but avoid deformable param
        for value in oldToNew.values():
            if (value in name and value != "conv_offset_mask") or (
                re.match(oldUpNodeRegex, name) is None
                and re.match(newUpNodeRegex, name) is not None
            ):
                return name

        # replace and return name (deformable param)
        if re.match(oldUpNodeRegex, name) is not None:
            name = name.replace("conv.weight", "weight")
            name = name.replace("conv.bias", "bias")
            return name

        # transform detect head from oldv2 to new
        for head in heads:
            if name.startswith(head):
                name = name.replace(head, "detectHead_0." + head)
                break
    elif (  # to old (deformable param)
        re.match(newUpNodeRegex, name) is not None
        and re.match(oldUpNodeRegex, name) is None
    ):
        name = name.replace("weight", "conv.weight")
        name = name.replace("bias", "conv.bias")
        return name

    # transform name
    for k, v in toggleDict.items():
        if k in name:
            name = name.replace(k, v)
            break

    return name


def renewCheckpoint(ckpt):
    """
    Renew checkpoint format for compatibility.
    Old format:
        ckpt = {train: {loss1: [1, 2, 3], loss2: ...}, val:...}
    New format:
        ckpt = {train: {loss1: {1: 1, 2: 2, 3: 3}, loss2: ...}, val:...}

    Args:
        ckpt(dict): model checkpoint with train and val loss

    Returns:
        ckpt(dict): new format
    """
    heads = [
        "total",
        # 2D head
        "heatmap",
        "widthHeight",
        "reg",
        # 3D head
        "depth",
        "depth2",
        "rotation",
        "rotation2",
        "dimension",
        "amodal_offset",
        # nuscenes head
        "nuscenes_att",
        "velocity",
    ]

    for loss in heads:
        # ==================== process train loss ====================
        if (
            "train" in ckpt
            and loss in ckpt["train"]
            and isinstance(ckpt["train"][loss], list)
        ):
            newTrainLossLog = {}
            for i, value in enumerate(ckpt["train"][loss]):
                newTrainLossLog[i + 1] = value
            ckpt["train"][loss] = newTrainLossLog

        # ==================== process val loss ====================
        if (
            "val" in ckpt
            and loss in ckpt["val"]
            and isinstance(ckpt["val"][loss], list)
        ):
            newValLossLog = {}
            val_interval = len(ckpt["train"][loss]) // len(ckpt["val"][loss])
            x_val = range(
                val_interval,
                len(ckpt["train"][loss]),
                val_interval,
            )[: len(ckpt["val"][loss])]
            y_val = ckpt["val"][loss][: len(x_val)]

            for i in range(len(x_val)):
                newValLossLog[x_val[i]] = y_val[i]
            ckpt["val"][loss] = newValLossLog

    return ckpt
