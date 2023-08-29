from yacs.config import CfgNode as CN
import yaml
import os

def updateConfig(config, args):
    '''
    Update config with args

    Args:
        config: config object
        args: yaml file path from argparser

    Returns:
        None
    '''
    config.defrost()

    if args.cfg is not None:
        config.merge_from_file(args.cfg)
    config.merge_from_list(args.opts)
    
    config.freeze()

def updateConfigHeads(config):
    '''
    Updates the config heads based on the dataset

    Args:
        config: config object

    Returns:
        None
    '''
    heads = {
        # 2D heads
        'heatmap': config.DATASET.NUM_CLASSES,
        'widthHeight': 2,
        'reg': 2,

        # 3D heads
        'depth': 1,
        'rotation': 8,
        'dimension': 3,
        'amodal_offset': 2}
    
    # nuscenes attribute head
    if config.LOSS.NUSCENES_ATT:
        heads.update({'nuscenes_att': 8})

    # Point cloud heads
    if config.DATASET.NUSCENES.RADAR_PC:
        heads.update({
            'depth2': 1,
            'rotation2': 8,
            'velocity': 3,
            })
        
    config.heads = CN()
    for k, v in heads.items():
        exec(f"config.heads.{k} = {v}")
            
def updateConfigHeadsWeights(config):
    '''
    Updates the config heads weights based on their heads

    Args:
        config: config object

    Returns:
        None
    '''
    weightDict = {
        # 2D head-weights
        'heatmap': config.LOSS.WEIGHTS.HEATMAP,
        'widthHeight': config.LOSS.WEIGHTS.BBOX,
        'reg': config.LOSS.WEIGHTS.AMODAL_OFFSET,

        # 3D head-weights
        'depth': config.LOSS.WEIGHTS.DEPTH,
        'rotation': config.LOSS.WEIGHTS.ROTATION,
        'dimension': config.LOSS.WEIGHTS.DIMENSION,
        'amodal_offset': config.LOSS.WEIGHTS.AMODAL_OFFSET,

        # other head-weights
        'nuscenes_att': config.LOSS.WEIGHTS.NUSCENES_ATT,
        'velocity': config.LOSS.WEIGHTS.VELOCITY,
    }
    config.weights = CN()
    for k, v in weightDict.items():
        exec(f"config.weights.{k} = {v}")

def updateConvNumOfHeads(config):
    '''
    Assigns the number of convolutions layer for each head

    Args:
        config: config object

    Returns:
        None
    '''
    head_conv = {
        head: [256 if head != 'reg' else 1] for head in config.heads
        }
    head_conv.update({
        'depth2': [256, 256, 256],
        'rotation2': [256, 256, 256],
        'velocity': [256, 256, 256],
        'nuscenes_att': [256, 256, 256],
    })
    config.head_conv = CN()
    for k, v in head_conv.items():
        exec(f"config.head_conv.{k} = {v}")

def updateDatasetAndModelConfig(config, dataset, output_dir):
    '''
    Update dataset and model config with dataset

    Args:
        config: config object
        dataset: dataset object
        output_dir: final output directory

    Returns:
        None
    '''
    config.defrost()
    
    # save config
    config.OUTPUT_DIR = output_dir
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(config.dump())

    # update config
    config.DATASET.NUM_CLASSES = dataset.num_categories
    if config.MODEL.INPUT_SIZE is None:
        config.MODEL.INPUT_SIZE = dataset.default_resolution
    config.MODEL.OUTPUT_SIZE = (config.MODEL.INPUT_SIZE[0] // 4,
                                config.MODEL.INPUT_SIZE[1] // 4)

    updateConfigHeads(config)
    updateConfigHeadsWeights(config)
    updateConvNumOfHeads(config)

    config.freeze()