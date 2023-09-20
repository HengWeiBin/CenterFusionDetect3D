from yacs.config import CfgNode as CN

_Cfg = CN()
_Cfg.NAME = 'CenterFusion'

_Cfg.CUDNN = CN()
_Cfg.CUDNN.BENCHMARK = True
_Cfg.CUDNN.DETERMINISTIC = False
_Cfg.CUDNN.ENABLED = True

_Cfg.GPUS = (0,)              # -1 for CPU
_Cfg.WORKERS = 4
_Cfg.DEBUG = 0                # 0: no debug, 1: show debug images, 2: save debug images
_Cfg.EVAL = False             # only evaluate the val split and quit
_Cfg.RANDOM_SEED = 0
_Cfg.MIXED_PRECISION = False  # FP16 training
_Cfg.CONF_THRESH = 0.3        # confidence threshold for visualization

_Cfg.DATASET = CN()
_Cfg.DATASET.DATASET = 'nuscenes'   # currently support nuscenes only
_Cfg.DATASET.ROOT = 'data/'
_Cfg.DATASET.RANDOM_CROP = False    # random crop data augmentation from CornerNet
_Cfg.DATASET.MAX_CROP = True        # used when the training dataset has inbalanced aspect ratios
_Cfg.DATASET.SHIFT = 0.2            # shift augmentation factor
_Cfg.DATASET.SCALE = 0              # scale augmentation factor
_Cfg.DATASET.ROTATE = 0             # rotation augmentation factor
_Cfg.DATASET.FLIP = 0.5             # flip augmentation factor
_Cfg.DATASET.COLOR_AUG = True       # color augmenation from CornerNet
_Cfg.DATASET.TRAIN_SPLIT = 'train'  # ['train','mini_train']
_Cfg.DATASET.VAL_SPLIT = 'mini_val' # ['val','mini_val','test']

_Cfg.DATASET.NUSCENES = CN()
_Cfg.DATASET.NUSCENES.RADAR_PC = True               # use radar point cloud
_Cfg.DATASET.NUSCENES.MAX_PC = 1000                 # maximum number of points in the point cloud
_Cfg.DATASET.NUSCENES.MAX_PC_DIST = 60.0            # remove points beyond max_pc_dist (meters)
_Cfg.DATASET.NUSCENES.PC_Z_OFFSET = 0.0             # raise all Radar points in z direction
_Cfg.DATASET.NUSCENES.PC_ROI_METHOD = 'pillars'     # pillars or heatmap
_Cfg.DATASET.NUSCENES.PILLAR_DIMS = (1.5,0.2,0.2)   # Radar pillar dimensions (h,w,l)

_Cfg.MODEL = CN()
_Cfg.MODEL.LOAD_DIR = ''
_Cfg.MODEL.ARCH = "dla_34"              # model architecture
_Cfg.MODEL.FREEZE_BACKBONE = False      # freeze the backbone network and only train heads
_Cfg.MODEL.FUSION_STRATEGY = 'middle'   # TODO
_Cfg.MODEL.FRUSTUM = True               # Enable frustum association
_Cfg.MODEL.K = 100                      # max number of output objects
_Cfg.MODEL.INPUT_SIZE = (448, 800)

_Cfg.MODEL.GENERIC_NET = CN()
_Cfg.MODEL.GENERIC_NET.BACKBONE = "dla34"
_Cfg.MODEL.GENERIC_NET.NECK = "dlaup"

_Cfg.MODEL.DLA = CN()
_Cfg.MODEL.DLA.NODE = "DeformConv"      # [DeformConv | GlobalConv | Conv]

_Cfg.LOSS = CN()
_Cfg.LOSS.NUSCENES_ATT = True
_Cfg.LOSS.VELOCITY = True

_Cfg.LOSS.WEIGHTS = CN()
_Cfg.LOSS.WEIGHTS.HEATMAP = 1       # keypoint heatmaps
_Cfg.LOSS.WEIGHTS.AMODAL_OFFSET = 1 # keypoint local offsets
_Cfg.LOSS.WEIGHTS.BBOX = 0.1        # bounding box size
_Cfg.LOSS.WEIGHTS.DEPTH = 1         # depthloss weight for depth
_Cfg.LOSS.WEIGHTS.DIMENSION = 1     # 3d bounding box size
_Cfg.LOSS.WEIGHTS.ROTATION = 1      # orientation
_Cfg.LOSS.WEIGHTS.NUSCENES_ATT = 1  # nuscenes attribute
_Cfg.LOSS.WEIGHTS.VELOCITY = 1      # velocity

_Cfg.TRAIN = CN()
_Cfg.TRAIN.BATCH_SIZE = 26
_Cfg.TRAIN.SHUFFLE = True       # shuffle training dataloader
_Cfg.TRAIN.EPOCHS = 60
_Cfg.TRAIN.RESUME = False
_Cfg.TRAIN.OPTIMIZER = 'adam'
_Cfg.TRAIN.LR = 2.5e-4
_Cfg.TRAIN.LR_STEP = (50,)      # drop learning rate step
_Cfg.TRAIN.SAVE_INTERVALS = 10  # interval to save model
_Cfg.TRAIN.VAL_INTERVALS = 10   # number of epochs to run validation
_Cfg.TRAIN.SCALE_FACTOR = 16

_Cfg.TEST = CN()
_Cfg.TEST.BATCH_SIZE = 1
_Cfg.TEST.OFFICIAL_EVAL = False # use official evaluation script