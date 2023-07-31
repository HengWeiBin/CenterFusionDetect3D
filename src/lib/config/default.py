from yacs.config import CfgNode as CN

_Cfg = CN()
_Cfg.NAME = 'CenterFusion'

_Cfg.CUDNN = CN()
_Cfg.CUDNN.BENCHMARK = True
_Cfg.CUDNN.DETERMINISTIC = False
_Cfg.CUDNN.ENABLED = True

_Cfg.GPUS = (0,)
_Cfg.WORKERS = 4
_Cfg.DEBUG = 0
_Cfg.EVAL = False
_Cfg.RANDOM_SEED = 0

_Cfg.DATASET = CN()
_Cfg.DATASET.DATASET = 'nuscenes'
_Cfg.DATASET.ROOT = 'data/'
_Cfg.DATASET.RANDOM_CROP = False
_Cfg.DATASET.MAX_CROP = True
_Cfg.DATASET.SHIFT = 0.2
_Cfg.DATASET.SCALE = 0
_Cfg.DATASET.ROTATE = 0
_Cfg.DATASET.FILP = 0.5
_Cfg.DATASET.COLOR_AUG = True

_Cfg.DATASET.NUSCENES = CN()
_Cfg.DATASET.NUSCENES.POINTCLOUD = True
_Cfg.DATASET.NUSCENES.TRAIN_SPLIT = 'train'
_Cfg.DATASET.NUSCENES.VAL_SPLIT = 'mini_val'
_Cfg.DATASET.NUSCENES.MAX_PC = 1000
_Cfg.DATASET.NUSCENES.MAX_PC_DIST = 60.0
_Cfg.DATASET.NUSCENES.PC_Z_OFFSET = 0.0
_Cfg.DATASET.NUSCENES.PILLAR_DIMS = '1.5,0.2,0.2'

_Cfg.MODEL = CN()
_Cfg.MODEL.LOAD_DIR = ''
_Cfg.MODEL.ARCH = "dla_34"
_Cfg.MODEL.FREEZE_BACKBONE = False
_Cfg.MODEL.FUSION_STRATEGY = 'middle'

_Cfg.MODEL.GENERIC_NET = CN()
_Cfg.MODEL.GENERIC_NET.BACKBONE = "dla34"
_Cfg.MODEL.GENERIC_NET.NECK = "dlaup"

_Cfg.LOSS = CN()
_Cfg.LOSS.REG_LOSS = 'l1'
_Cfg.LOSS.NUSCENES_ATT = True
_Cfg.LOSS.VELOCITY = True

_Cfg.LOSS.WEIGHT = CN()
_Cfg.LOSS.WEIGHT.HEATMAP = 1
_Cfg.LOSS.WEIGHT.OFFSET = 1
_Cfg.LOSS.WEIGHT.BBOX = 0.1
_Cfg.LOSS.WEIGHT.DEPTH = 1
_Cfg.LOSS.WEIGHT.DIMENTION = 1
_Cfg.LOSS.WEIGHT.ROTATION = 1
_Cfg.LOSS.WEIGHT.NUSCENES_ATT = 1
_Cfg.LOSS.WEIGHT.VELOCITY = 1

_Cfg.TRAIN = CN()
_Cfg.TRAIN.IMAGE_SIZE = [448, 800]
_Cfg.TRAIN.BATCH_SIZE = 26
_Cfg.TRAIN.SHUFFLE = True
_Cfg.TRAIN.EPOCHS = 60
_Cfg.TRAIN.RESUME = False
_Cfg.TRAIN.OPIMIZER = 'adam'
_Cfg.TRAIN.LR = 2.5e-4
_Cfg.TRAIN.LR_STEP = 50
_Cfg.TRAIN.SAVE_POINT = [20, 40, 50]
_Cfg.TRAIN.VAL_INTERVAL = 10
_Cfg.TRAIN.SCALE_FACTOR = 16

_Cfg.TEST = CN()
_Cfg.TEST.IMAGE_SIZE = [448, 800]
_Cfg.TEST.BATCH_SIZE = 1
_Cfg.TEST.K = 100
_Cfg.TEST.FOCAL_LENGTH = -1

def updateConfig(config, args):
    config.defrost()

    if args.cfg is not None:
        config.merge_from_file(args.cfg)
    config.merge_from_list(args.opts)
    
    config.freeze()