from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.render import visualize_sample

from utils.ddd import get3dBox, project3DPoints, draw3DBox

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    image = None
    imgId = None
    dataset = None
    targetPcHmOverlay = None
    targetBox3DOverlay = None
    predPcHmOverlay = None
    predBox3DOverlay = None
    bev = None
    sampleToken = None
    log = {}

    @staticmethod
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
        WandbLogger.log = {}

    @staticmethod
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
        WandbLogger.sampleToken = imageInfo["sample_token"]

        # Read and resize image
        image = cv2.imread(os.path.join(dataset.img_dir, imageInfo["file_name"]))
        WandbLogger.image = cv2.resize(
            image,
            (config.MODEL.INPUT_SIZE[1], config.MODEL.INPUT_SIZE[0]),
        )

        # Render ground truth
        WandbLogger.drawPcHeatmap(pc_hm.cpu().numpy().transpose(1, 2, 0), isTarget=True)
        WandbLogger.drawBox3D(
            {"predictBoxes": anns, "calib": imageInfo["calib"]}, isTarget=True
        )

    @staticmethod
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
        if WandbLogger.predPcHmOverlay is not None:
            return

        WandbLogger.renderNuscBev(predictBoxes)
        WandbLogger.drawPcHeatmap(
            pc_hm.cpu().numpy().transpose(1, 2, 0), isTarget=False
        )
        WandbLogger.drawBox3D(
            {"predictBoxes": predictBoxes, "calib": calib}, isTarget=False
        )

    @staticmethod
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
        if pc_hm.shape[2] == 3:
            pc_hm = cv2.cvtColor(pc_hm, cv2.COLOR_BGR2GRAY)
        input_shape = WandbLogger.image.shape
        pc_hm = cv2.resize(pc_hm, (input_shape[1], input_shape[0]))
        image[pc_hm > 0] = 0
        image[:, :, 1][pc_hm > 0] = pc_hm[pc_hm > 0]

        if isTarget:
            WandbLogger.targetPcHmOverlay = image
        else:
            WandbLogger.predPcHmOverlay = image

    @staticmethod
    def drawBox3D(batch, isTarget: bool):
        """
        Draw 3D bounding box on input image

        Args:
            batch: batch of data
            isTarget: whether the box is ground truth or prediction

        Returns:
            None
        """
        image = WandbLogger.image.copy()
        calib = batch["calib"]
        predictBoxes = batch["predictBoxes"]

        # draw 3D bounding box only for first image in batch
        for predictBox in predictBoxes:
            bbox3D = get3dBox(
                predictBox["dimension"],
                predictBox["location"],
                predictBox["yaw"],
            )
            projectedBox2D = project3DPoints(bbox3D, calib) / 2
            image = draw3DBox(image, projectedBox2D)

        if isTarget:
            WandbLogger.targetBox3DOverlay = image
        else:
            WandbLogger.predBox3DOverlay = image

    @staticmethod
    def renderNuscBev(predictBoxes):
        nusc = WandbLogger.dataset.nusc
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
        )
        WandbLogger.bev = cv2.imread("./tempNuscBev.png")
        if os.path.exists("./tempNuscBev.png"):
            os.remove("./tempNuscBev.png")

    @staticmethod
    def syncVisualizeResult():
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
                print(f"Warning: {titles[i]} is None")
                continue
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            wandbImages.append(wandb.Image(images[i], caption=f"{titles[i]}"))
        WandbLogger.log.update({"val/boxes": wandbImages[:4]})
        WandbLogger.log.update({"val/bev": wandbImages[4]})
        wandb.log(WandbLogger.log)
        print(f"Visualized {WandbLogger.sampleToken}")

        WandbLogger.reset()
