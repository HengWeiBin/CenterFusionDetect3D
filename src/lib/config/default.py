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
_Cfg.WANDB_RESUME = True      # resume wandb run by run_id if possible
_Cfg.WANDB_RESUBMIT = False   # resubmit wandb run by run_id if possible

_Cfg.DATASET = CN()
_Cfg.DATASET.DATASET = 'nuscenes'   # ["nuscenes"]
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
_Cfg.DATASET.RADAR_PC = True               # use radar point cloud
_Cfg.DATASET.MAX_PC = 1000                 # maximum number of points in the point cloud
_Cfg.DATASET.MAX_PC_DIST = 60.0            # remove points beyond max_pc_dist (meters)
_Cfg.DATASET.PC_Z_OFFSET = 0.0             # raise all Radar points in z direction
_Cfg.DATASET.PC_ROI_METHOD = 'pillars'     # pillars or heatmap
_Cfg.DATASET.PILLAR_DIMS = (1.5,0.2,0.2)   # Radar pillar dimensions (h,w,l)
_Cfg.DATASET.ONE_HOT_PC = False            # use one-hot encoding for Radar point cloud features
_Cfg.DATASET.DECOUPLE_REP = False          # decouple representation of 3D objects from MonoFlex
_Cfg.DATASET.HEATMAP_REP = "2d"            # ["2d", "3d"]

_Cfg.MODEL = CN()
_Cfg.MODEL.LOAD_DIR = ''
_Cfg.MODEL.ARCH = "dla_34"              # model architecture
_Cfg.MODEL.FREEZE_BACKBONE = False      # freeze the backbone network and only train heads
_Cfg.MODEL.NORM_EVAL = False            # freeze the backbone batchnorm during training
_Cfg.MODEL.NORM_2D = False               # normalize any 2D input/output
_Cfg.MODEL.DEFREEZE = -1                # number of epochs to defreeze backbone (-1 disable)
_Cfg.MODEL.FUSION_STRATEGY = 'middle'   # ['early','middle']
_Cfg.MODEL.FRUSTUM = True               # Enable frustum association
_Cfg.MODEL.K = 100                      # max number of output objects
_Cfg.MODEL.INPUT_SIZE = (448, 800)

_Cfg.MODEL.DLA = CN()
_Cfg.MODEL.DLA.NODE = "DeformConv"      # [DeformConv | GlobalConv | Conv]

_Cfg.LOSS_WEIGHTS = CN()
_Cfg.LOSS_WEIGHTS.HEATMAP = 1.          # keypoint heatmaps
_Cfg.LOSS_WEIGHTS.AMODAL_OFFSET = 1.    # keypoint local offsets
_Cfg.LOSS_WEIGHTS.DIMENSION_2D = 0.1    # 2d bounding box size
_Cfg.LOSS_WEIGHTS.DEPTH = 1.            # depthloss weight for depth
_Cfg.LOSS_WEIGHTS.DIMENSION_3D = 1.     # 3d bounding box size
_Cfg.LOSS_WEIGHTS.ROTATION = 1.         # orientation
_Cfg.LOSS_WEIGHTS.NUSCENES_ATT = 1.     # nuscenes attribute
_Cfg.LOSS_WEIGHTS.VELOCITY = 1.         # velocity
_Cfg.LOSS_WEIGHTS.BBOX_2D = 0.          # 2d bounding box
_Cfg.LOSS_WEIGHTS.BBOX_3D = 0.          # 3d bounding box
_Cfg.LOSS_WEIGHTS.LIDAR_DEPTH = 0.      # auxiliary lidar depth loss
_Cfg.LOSS_WEIGHTS.RADAR_DEPTH = 0.      # auxiliary radar depth loss

_Cfg.TRAIN = CN()
_Cfg.TRAIN.BATCH_SIZE = 26
_Cfg.TRAIN.SHUFFLE = True           # shuffle training dataloader
_Cfg.TRAIN.EPOCHS = 60
_Cfg.TRAIN.WARM_EPOCHS = 5          # Gradually warm-up(increase) learning rate
_Cfg.TRAIN.RESUME = False
_Cfg.TRAIN.OPTIMIZER = 'adam'
_Cfg.TRAIN.LR = 2.5e-4
_Cfg.TRAIN.LR_STEP = (50,)          # drop learning rate step
_Cfg.TRAIN.SAVE_INTERVALS = 10      # interval to save model
_Cfg.TRAIN.VAL_INTERVALS = 10       # number of epochs to run validation
_Cfg.TRAIN.SCALE_FACTOR = 16
_Cfg.TRAIN.LR_SCHEDULER = "StepLR"     # ['CLR', 'StepLR']
_Cfg.TRAIN.UNCERTAINTY_LOSS = False # enable aleatoric uncertainty loss

_Cfg.TEST = CN()
_Cfg.TEST.BATCH_SIZE = 1
_Cfg.TEST.OFFICIAL_EVAL = False # use official evaluation script