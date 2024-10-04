from yacs.config import CfgNode as CN
import os
import warnings


def updateConfig(config, args):
    """
    Update config with args

    Args:
        config: config object
        args: yaml file path from argparser

    Returns:
        None
    """
    config.defrost()

    if getattr(args, "cfg", None) is not None:
        config.merge_from_file(args.cfg)
    config.merge_from_list(args.opts)

    if config.DATASET.RADAR_PC:
        if config.MODEL.FRUSTUM and config.MODEL.FUSION_STRATEGY != "middle":
            warnings.warn(
                "Frustum association is only available for radar point cloud fusion with middle strategy."
            )
            warnings.warn("Disabling frustum association...")
            config.MODEL.FRUSTUM = False

        config.DATASET.PC_REVERSE = False
        if config.DATASET.PC_ROI_METHOD != "points":
            config.DATASET.PC_REVERSE = True

    else:
        if config.MODEL.FRUSTUM:
            warnings.warn(
                "Frustum association is only available for radar point cloud fusion."
            )
            warnings.warn("Disabling frustum association...")
            config.MODEL.FRUSTUM = False

        if config.MODEL.FUSION_STRATEGY is not None:
            warnings.warn(
                "Fusion strategy is only available for radar point cloud fusion."
            )
            warnings.warn("Disabling fusion strategy...")
            config.MODEL.FUSION_STRATEGY = None

    if config.TRAIN.WARM_EPOCHS:
        if config.TRAIN.LR_SCHEDULER != "StepLR":
            warnings.warn("Warmup epochs are only available for StepLR scheduler.")
            warnings.warn("Disabling warmup epochs...")
            config.TRAIN.WARM_EPOCHS = 0

        if config.TRAIN.RESUME:
            warnings.warn(
                "Attention: Warmup epochs enabled with resume training. This may affect the training."
            )

    if config.MODEL.LOAD_DIR == "" and config.MODEL.NORM_EVAL:
        warnings.warn(
            "Norm eval (tune mode) may affect the training if no pretrained model is loaded."
        )

    config.freeze()


def updateConfigHeads(config):
    """
    Updates the config heads based on the dataset

    Args:
        config: config object

    Returns:
        None
    """
    # Basic heads
    heads = {
        # 2D heads
        "heatmap": config.DATASET.NUM_CLASSES,
        "reg": 2,
        "widthHeight": 2,
        # 3D heads
        "depth": 1,
        "rotation": 8,
        "dimension": 3,
        "amodal_offset": 2,
    }

    # Other heads
    if config.DATASET.DATASET == "nuscenes":
        heads.update({"nuscenes_att": 8, "velocity": 3})

    # Point cloud heads
    if config.DATASET.RADAR_PC and config.MODEL.FUSION_STRATEGY == "middle":
        heads.update({"depth2": 1, "rotation2": 8})

    # Aleatoric uncertainty heads
    if config.TRAIN.UNCERTAINTY_LOSS:
        heads.update({"uncertainty": 1})

    config.heads = CN()
    for k, v in heads.items():
        exec(f"config.heads.{k} = {v}")


def updateConfigHeadsWeights(config):
    """
    Updates the config heads weights based on their heads

    Args:
        config: config object

    Returns:
        None
    """
    weightDict = {
        # 2D head-weights
        "heatmap": config.LOSS_WEIGHTS.HEATMAP,
        "widthHeight": config.LOSS_WEIGHTS.DIMENSION_2D,
        "reg": config.LOSS_WEIGHTS.AMODAL_OFFSET,
        "bbox2d": config.LOSS_WEIGHTS.BBOX_2D,
        # 3D head-weights
        "depth": config.LOSS_WEIGHTS.DEPTH,
        "depth2": config.LOSS_WEIGHTS.DEPTH,
        "rotation": config.LOSS_WEIGHTS.ROTATION,
        "rotation2": config.LOSS_WEIGHTS.ROTATION,
        "dimension": config.LOSS_WEIGHTS.DIMENSION_3D,
        "amodal_offset": config.LOSS_WEIGHTS.AMODAL_OFFSET,
        "bbox3d": config.LOSS_WEIGHTS.BBOX_3D,
        "lidar_depth": config.LOSS_WEIGHTS.LIDAR_DEPTH,
        "radar_depth": config.LOSS_WEIGHTS.RADAR_DEPTH,
        # other head-weights
        "nuscenes_att": config.LOSS_WEIGHTS.NUSCENES_ATT,
        "velocity": config.LOSS_WEIGHTS.VELOCITY,
    }
    config.weights = CN()
    for k, v in weightDict.items():
        exec(f"config.weights.{k} = {v}")


def updateConvNumOfHeads(config):
    """
    Assigns the number of convolutions layer for each head

    Args:
        config: config object

    Returns:
        None
    """
    head_conv = {head: [256] for head in config.heads}

    if config.DATASET.RADAR_PC:
        if config.MODEL.FUSION_STRATEGY == "middle":
            head_conv.update({"depth2": [256, 256, 256], "rotation2": [256, 256, 256]})
        if config.DATASET.DATASET == "nuscenes":
            head_conv.update(
                {"velocity": [256, 256, 256], "nuscenes_att": [256, 256, 256]}
            )

    config.head_conv = CN()
    for k, v in head_conv.items():
        exec(f"config.head_conv.{k} = {v}")


def updateDatasetAndModelConfig(config, dataset, output_dir=None):
    """
    Update dataset and model config with dataset

    Args:
        config: config object
        dataset: dataset object
        output_dir: final output directory

    Returns:
        None
    """
    config.defrost()

    # save config
    if output_dir is not None:
        config.OUTPUT_DIR = output_dir
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(config.dump())

    # update config
    config.DATASET.NUM_CLASSES = dataset.num_categories
    if config.MODEL.INPUT_SIZE is None:
        config.MODEL.INPUT_SIZE = dataset.default_resolution
    config.MODEL.OUTPUT_SIZE = (
        config.MODEL.INPUT_SIZE[0] // 4,
        config.MODEL.INPUT_SIZE[1] // 4,
    )
    if not config.MODEL.FREEZE_BACKBONE:
        config.MODEL.DEFREEZE = 0

    updateConfigHeads(config)
    updateConfigHeadsWeights(config)
    updateConvNumOfHeads(config)

    config.freeze()
