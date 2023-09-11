from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os

from model import fusionDecode
from utils.ddd import get3dBox, project3DPoints, draw3DBox
from utils import postProcess

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    image = None
    targetPcHmOverlay = None
    targetBox3DOverlay = None
    predPcHmOverlay = None
    predBox3DOverlay = None

    @staticmethod
    def reset():
        """
        Reset logger
        """
        WandbLogger.image = None
        WandbLogger.targetPcHmOverlay = None
        WandbLogger.targetBox3DOverlay = None
        WandbLogger.predPcHmOverlay = None
        WandbLogger.predBox3DOverlay = None

    @staticmethod
    def addGroundTruth(coco, imageDir, imageId, pc_hm, config):
        """
        Add ground truth to be logged

        Args:
            batch: batch of data

        Returns:
            None
        """
        if WandbLogger.image is None:
            imageInfo = coco.loadImgs(ids=[imageId])[0]
            ann_ids = coco.getAnnIds(imgIds=[imageId])
            anns = coco.loadAnns(ids=ann_ids)

            image = cv2.imread(os.path.join(imageDir, imageInfo["file_name"]))
            WandbLogger.image = cv2.resize(
                image,
                (config.MODEL.INPUT_SIZE[1], config.MODEL.INPUT_SIZE[0]),
            )

            WandbLogger.drawPcHeatmap(
                pc_hm.cpu().numpy().transpose(1, 2, 0), isTarget=True
            )
            WandbLogger.drawBox3D(
                {"predictBoxes": anns, "calib": imageInfo["calib"]},
                config,
                isTarget=True,
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
        if WandbLogger.predPcHmOverlay is None:
            WandbLogger.drawPcHeatmap(
                pc_hm.cpu().numpy().transpose(1, 2, 0), isTarget=False
            )
            WandbLogger.drawBox3D(
                {"predictBoxes": predictBoxes, "calib": calib}, None, isTarget=False
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
    def drawBox3D(batch, config, isTarget: bool):
        """
        Draw 3D bounding box on input image

        Args:
            batch: batch of data
            config: config file

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
    def syncVisualizeResult():
        """
        Log visualization result to wandb
        """
        if not (wandb and wandb.run):
            return

        images = [
            WandbLogger.targetPcHmOverlay,
            WandbLogger.targetBox3DOverlay,
            WandbLogger.predPcHmOverlay,
            WandbLogger.predBox3DOverlay,
        ]
        titles = ["target/pc_hm", "target/box_3d", "pred/pc_hm", "pred/box_3d"]

        wandbImages = []
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            wandbImages.append(wandb.Image(images[i], caption=f"{titles[i]}"))
        wandb.log({"val/boxes": wandbImages})

        WandbLogger.reset()
