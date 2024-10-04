from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import logging
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.render import visualize_sample
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger

from utils import safe_run
from utils import ddd
from utils.image import affineTransform, getAffineTransform

try:
    import wandb
    from wandb.vendor.pynvml import pynvml
    from wandb import AlertLevel
    from wandb.sdk.internal.system.assets.gpu import gpu_in_use_by_this_process

    pynvml.nvmlInit()
except ImportError:
    wandb = None

WARNING_TEMP = 85


class WandbLogger(LightningWandbLogger):
    image = None
    imgId = None
    dataset = None
    targetPcHmOverlay = None
    targetBox3DOverlay = None
    predPcHmOverlay = None
    predBox3DOverlay = None
    bev = None
    sampleToken = None
    conf_thresh = None
    ckptSubmitPending = None
    log = {}
    toleranceCounter = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        WandbLogger.reset()

    def __del__(self):
        if wandb and wandb.run:
            wandb.finish()

    @staticmethod
    @rank_zero_only
    @safe_run
    def reset():
        """
        Reset logger
        """
        WandbLogger.image = None
        WandbLogger.imgId = None
        WandbLogger.dataset = None
        WandbLogger.targetPcHmOverlay = None
        WandbLogger.targetBox3DOverlay = None
        WandbLogger.predPcHmOverlay = None
        WandbLogger.predBox3DOverlay = None
        WandbLogger.bev = None
        WandbLogger.sampleToken = None
        WandbLogger.conf_thresh = None
        WandbLogger.transMatInput = None
        WandbLogger.log = {}

    @staticmethod
    @rank_zero_only
    @safe_run
    def addGroundTruth(dataset, imageId, pc_hm, config):
        """
        Add ground truth to be logged

        Args:
            batch: batch of data

        Returns:
            None
        """
        if WandbLogger.image is not None:
            return

        # Save image id and dataset for render bev
        WandbLogger.imgId = imageId
        WandbLogger.dataset = dataset

        # Get image info and annotations
        imageInfo = dataset.coco.loadImgs(ids=[imageId])[0]
        ann_ids = dataset.coco.getAnnIds(imgIds=[imageId])
        anns = dataset.coco.loadAnns(ids=ann_ids)
        if "sample_token" in imageInfo:
            WandbLogger.sampleToken = imageInfo["sample_token"]

        # Get transformation matrix
        center = np.array([imageInfo["width"] / 2.0, imageInfo["height"] / 2.0])
        input_size = (config.MODEL.INPUT_SIZE[1], config.MODEL.INPUT_SIZE[0])
        if config.DATASET.MAX_CROP:
            scale = max(imageInfo["height"], imageInfo["width"]) * 1.0
        else:
            scale = np.array(
                [imageInfo["width"], imageInfo["height"]], dtype=np.float32
            )
        WandbLogger.transMatInput = getAffineTransform(
            center,
            scale,
            0,
            input_size,
        )

        # Read and resize image
        image = cv2.imread(os.path.join(dataset.img_dir, imageInfo["file_name"]))
        WandbLogger.image = cv2.resize(image, input_size)

        # Render ground truth
        WandbLogger.conf_thresh = config.CONF_THRESH
        if pc_hm is not None:
            WandbLogger.drawPcHeatmap(pc_hm.cpu().numpy(), isTarget=True)
        WandbLogger.drawBox3D(
            {
                "predictBoxes": anns,
                "meta": {
                    "calib": imageInfo["calib"],
                    "transMatInput": WandbLogger.transMatInput,
                },
            },
            isTarget=True,
        )

    @staticmethod
    @rank_zero_only
    @safe_run
    def addPredict(predictBoxes, pc_hm, calib):
        """
        Add prediction to be logged

        Args:
            predictBoxes: predicted boxes (one)
            pc_hm: point cloud heatmap
            calib: calibration matrix

        Returns:
            None
        """
        if WandbLogger.bev is not None:
            return

        if WandbLogger.sampleToken is not None:
            WandbLogger.renderNuscBev(predictBoxes)

        if pc_hm is not None:
            WandbLogger.drawPcHeatmap(pc_hm.cpu().numpy(), isTarget=False)

        WandbLogger.drawBox3D(
            {
                "predictBoxes": predictBoxes,
                "meta": {"calib": calib, "transMatInput": WandbLogger.transMatInput},
            },
            isTarget=False,
        )

    @staticmethod
    @rank_zero_only
    @safe_run
    def drawPcHeatmap(pc_hm, isTarget: bool):
        """
        Overlay point cloud heatmap on input image

        Args:
            pc_hm: point cloud heatmap

        Returns:
            None
        """
        image = WandbLogger.image.copy()
        pc_hm = (pc_hm * 255).astype(np.uint8)
        input_shape = WandbLogger.image.shape
        pc_hm = cv2.resize(pc_hm, (input_shape[1], input_shape[0]))
        image[pc_hm > 0] = 0
        image[:, :, 1][pc_hm > 0] = pc_hm[pc_hm > 0]

        if isTarget:
            WandbLogger.targetPcHmOverlay = image
        else:
            WandbLogger.predPcHmOverlay = image

    @staticmethod
    @rank_zero_only
    @safe_run
    def drawBox3D(batch, isTarget: bool, drawOnTarget=False):
        """
        Draw 3D bounding box on input image

        Args:
            batch(dict): batch of data {"predictBoxes": ..., "meta": {"calib": ..., "transMatInput"},}
            isTarget: whether the box is ground truth or prediction
            drawOnTarget: whether to draw target with prediction box

        Returns:
            None
        """
        if not isTarget and drawOnTarget and WandbLogger.targetBox3DOverlay is not None:
            image = WandbLogger.targetBox3DOverlay.copy()
        else:
            image = WandbLogger.image.copy()
        calib = np.array(batch["meta"]["calib"]).reshape(1, 1, 3, 4)
        predictBoxes = batch["predictBoxes"]

        # draw 3D bounding box only for first image in batch
        for predictBox in predictBoxes:
            if "score" in predictBox and predictBox["score"] < WandbLogger.conf_thresh:
                continue
            if "bboxes3d" in predictBox:
                bbox3D = np.array(predictBox["bboxes3d"]).reshape(1, 1, 8, 3)
            elif {"dimension", "location", "yaw"} <= predictBox.keys():
                bbox3D = ddd.get3dBox(
                    np.array(predictBox["dimension"]).reshape(1, 1, 3),
                    np.array(predictBox["location"]).reshape(1, 1, 3),
                    np.array(predictBox["yaw"]).reshape(1, 1),
                )  # (B, K, 8, 3)
            else:
                raise ValueError(
                    "predictBox should contain either bboxes3d or dimension, location, yaw"
                )
            projectedBox3D = ddd.project3DPoints(bbox3D, calib)
            projectedBox3D = affineTransform(
                projectedBox3D[0, 0], batch["meta"]["transMatInput"]
            )
            image = ddd.draw3DBox(image, projectedBox3D, same_color=isTarget)

        if isTarget:
            WandbLogger.targetBox3DOverlay = image
        else:
            WandbLogger.predBox3DOverlay = image

    @staticmethod
    @rank_zero_only
    @safe_run
    def renderNuscBev(predictBoxes, nusc=None):
        """
        This function renders BEV image of current prediction and ground truth for nuscenes dataset
        """
        if nusc is None:
            nusc = NuScenes(
                version=WandbLogger.dataset.SPLITS[WandbLogger.dataset.split],
                dataroot=WandbLogger.dataset.img_dir,
                verbose=False,
            )
        cfg = config_factory("detection_cvpr_2019")
        nuscResult = WandbLogger.dataset.convert_eval_format(
            {WandbLogger.imgId: predictBoxes}
        )
        nuscPredBoxes = EvalBoxes.deserialize(nuscResult["results"], DetectionBox)
        nuscGtBoxes = load_gt(nusc, WandbLogger.dataset.split, DetectionBox)
        nuscPredBoxes = add_center_dist(nusc, nuscPredBoxes)
        nuscGtBoxes = add_center_dist(nusc, nuscGtBoxes)
        nuscPredBoxes = filter_eval_boxes(nusc, nuscPredBoxes, cfg.class_range)
        nuscGtBoxes = filter_eval_boxes(nusc, nuscGtBoxes, cfg.class_range)
        visualize_sample(
            nusc,
            WandbLogger.sampleToken,
            nuscGtBoxes,
            nuscPredBoxes,
            eval_range=max(cfg.class_range.values()),
            savepath="./tempNuscBev.png",
            verbose=False,
        )
        WandbLogger.bev = cv2.imread("./tempNuscBev.png")
        if os.path.exists("./tempNuscBev.png"):
            os.remove("./tempNuscBev.png")

    @staticmethod
    @rank_zero_only
    @safe_run
    def renderVisualizeResult():
        """
        Log visualization result to wandb
        """
        if WandbLogger.image is None or not (wandb and wandb.run):
            return

        images = [
            WandbLogger.targetPcHmOverlay,
            WandbLogger.targetBox3DOverlay,
            WandbLogger.predPcHmOverlay,
            WandbLogger.predBox3DOverlay,
            WandbLogger.bev,
        ]
        titles = ["target/pc_hm", "target/box_3d", "pred/pc_hm", "pred/box_3d", "bev"]

        wandbImages = []
        for i in range(len(images)):
            if images[i] is None:
                logging.info(f"Warning: {titles[i]} is None")
                continue
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            wandbImages.append(wandb.Image(images[i], caption=f"{titles[i]}"))
        WandbLogger.log.update({"val/boxes": wandbImages[:-1]})
        WandbLogger.log.update({"val/bev": wandbImages[-1]})
        logging.info(f"Visualized {WandbLogger.sampleToken}")

    @staticmethod
    @rank_zero_only
    @safe_run
    def show():
        """
        Show visualization result
        """
        if WandbLogger.image is None:
            return

        images = [
            WandbLogger.targetPcHmOverlay,
            WandbLogger.targetBox3DOverlay,
            WandbLogger.predPcHmOverlay,
            WandbLogger.predBox3DOverlay,
            WandbLogger.bev,
        ]
        titles = ["target/pc_hm", "target/box_3d", "pred/pc_hm", "pred/box_3d", "bev"]

        for i in range(len(images)):
            if images[i] is None:
                continue
            cv2.imshow(titles[i], images[i])

        WandbLogger.reset()

    @staticmethod
    @rank_zero_only
    @safe_run
    def commit():
        """
        Commit wandb log
        """
        if wandb and wandb.run and WandbLogger.log:
            if WandbLogger.ckptSubmitPending:
                ckpt = WandbLogger.ckptSubmitPending
                losses = list(ckpt["train"])
                stopEpoch = max(ckpt["train"]["total"]) + 1
                startEpoch = 141 if stopEpoch > 141 else 0
                for epoch in range(startEpoch, stopEpoch):
                    tempLog = {}
                    for loss in losses:
                        if loss in ckpt["train"] and epoch in ckpt["train"][loss]:
                            tempLog.update(
                                {f"train/{loss}": ckpt["train"][loss][epoch]}
                            )
                        if loss in ckpt["val"] and epoch in ckpt["val"][loss]:
                            tempLog.update({f"val/{loss}": ckpt["val"][loss][epoch]})
                    wandb.log(tempLog)
                    wandb.log({})
                WandbLogger.ckptSubmitPending = None

            wandb.log(WandbLogger.log)
            WandbLogger.reset()

    @staticmethod
    @rank_zero_only
    def checkGPUTemperature():
        """
        Check GPU temperature and log to wandb
        """
        self_PID = os.getpid()
        if not (wandb and wandb.run):
            return

        if WandbLogger.toleranceCounter is None:
            WandbLogger.toleranceCounter = ToleranceCounter(5)

        device_count = pynvml.nvmlDeviceGetCount()
        isSafe = True
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            if not gpu_in_use_by_this_process(handle, self_PID):
                continue

            # Get monitoring information
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            powerWatts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            powerLimit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000

            # Monitor GPU temperature and power
            # If temperature is higher than WARNING_TEMP, send warning alert
            # If power is throttling, send error alert and raise RuntimeError
            if temperature > WARNING_TEMP:
                wandb.alert(
                    title="GPU Temperature Warning",
                    text=f"GPU({i}) temperature is {temperature} degree celsius",
                    level=AlertLevel.WARN,
                )

                if ((powerWatts / powerLimit) * 100) < 90:
                    isSafe = False
                    wandb.alert(
                        title="GPU Power Throttling Warning",
                        text=f"Detected GPU power throttling on GPU({i})",
                        level=AlertLevel.WARN,
                    )

        if not WandbLogger.toleranceCounter.step(isSafe):
            wandb.alert(
                title="GPU Power Throttling Error",
                text="GPU power is throttling, training suspended",
                level=AlertLevel.ERROR,
            )
            raise RuntimeError("GPU power is throttling, training suspended")


@rank_zero_only
def initWandb(config, ckpt, output_dir):
    """
    This function initializes wandb
    Continue training if wandb_id is provided in ckpt

    Args:
        config: config object
        ckpt: checkpoint dict
        output_dir: output directory

    Returns:
        wandb object if wandb is installed
    """
    if wandb is not None:
        # Generate wandb id
        resumeSuccess = False
        if config.WANDB_RESUME and "wandb_id" in ckpt and not config.EVAL:
            wandb_id = ckpt["wandb_id"]
            resumeSuccess = True
        else:
            wandb_id = wandb.util.generate_id()
            ckpt["wandb_id"] = wandb_id

        # Initialize wandb
        if config.WANDB_RESUBMIT and not resumeSuccess:
            WandbLogger.ckptSubmitPending = ckpt.copy()

        return WandbLogger(
            name=config.NAME,
            save_dir=output_dir,
            dir=output_dir,
            id=wandb_id,
            project="CameraRadarFusionDetect3D",
            job_type="train" if not config.EVAL else "eval",
            config=config,
            resume="allow",
        )
    else:
        logging.info("wandb is not installed, wandb logging is disabled")


class ToleranceCounter:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.counter = 0
        self.safeCounter = 0

    def step(self, isSafe):
        if isSafe:
            self.safeCounter += 1
        else:
            self.counter += 1

        if self.counter >= self.tolerance:
            return False

        if self.safeCounter >= self.tolerance:
            self.reset()

        return True
