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
_Cfg.DATASET.FLIP = 0.5
_Cfg.DATASET.COLOR_AUG = True
_Cfg.DATASET.TRAIN_SPLIT = 'train'
_Cfg.DATASET.VAL_SPLIT = 'mini_val'

_Cfg.DATASET.NUSCENES = CN()
_Cfg.DATASET.NUSCENES.RADAR_PC = True
_Cfg.DATASET.NUSCENES.MAX_PC = 1000
_Cfg.DATASET.NUSCENES.MAX_PC_DIST = 60.0
_Cfg.DATASET.NUSCENES.PC_Z_OFFSET = 0.0
_Cfg.DATASET.NUSCENES.PC_ROI_METHOD = 'pillars'
_Cfg.DATASET.NUSCENES.PILLAR_DIMS = (1.5,0.2,0.2)

_Cfg.MODEL = CN()
_Cfg.MODEL.LOAD_DIR = ''
_Cfg.MODEL.ARCH = "dla_34"
_Cfg.MODEL.FREEZE_BACKBONE = False
_Cfg.MODEL.FUSION_STRATEGY = 'middle'
_Cfg.MODEL.FRUSTUM = True
_Cfg.MODEL.INPUT_SIZE = (448, 800)

_Cfg.MODEL.GENERIC_NET = CN()
_Cfg.MODEL.GENERIC_NET.BACKBONE = "dla34"
_Cfg.MODEL.GENERIC_NET.NECK = "dlaup"

_Cfg.MODEL.DLA = CN()
_Cfg.MODEL.DLA.NODE = "DeformConv"

_Cfg.LOSS = CN()
_Cfg.LOSS.REG_LOSS = 'l1'
_Cfg.LOSS.NUSCENES_ATT = True
_Cfg.LOSS.VELOCITY = True

_Cfg.LOSS.WEIGHTS = CN()
_Cfg.LOSS.WEIGHTS.HEATMAP = 1
_Cfg.LOSS.WEIGHTS.AMODAL_OFFSET = 1
_Cfg.LOSS.WEIGHTS.BBOX = 0.1
_Cfg.LOSS.WEIGHTS.DEPTH = 1
_Cfg.LOSS.WEIGHTS.DIMENSION = 1
_Cfg.LOSS.WEIGHTS.ROTATION = 1
_Cfg.LOSS.WEIGHTS.NUSCENES_ATT = 1
_Cfg.LOSS.WEIGHTS.VELOCITY = 1

_Cfg.TRAIN = CN()
_Cfg.TRAIN.BATCH_SIZE = 26
_Cfg.TRAIN.SHUFFLE = True
_Cfg.TRAIN.EPOCHS = 60
_Cfg.TRAIN.RESUME = False
_Cfg.TRAIN.OPTIMIZER = 'adam'
_Cfg.TRAIN.LR = 2.5e-4
_Cfg.TRAIN.LR_STEP = (50,)
_Cfg.TRAIN.SAVE_POINT = (20, 40, 50)
_Cfg.TRAIN.VAL_INTERVALS = 10
_Cfg.TRAIN.SCALE_FACTOR = 16

_Cfg.TEST = CN()
_Cfg.TEST.BATCH_SIZE = 1
_Cfg.TEST.K = 100
_Cfg.TEST.FOCAL_LENGTH = -1
_Cfg.TEST.OFFICIAL_EVAL = False