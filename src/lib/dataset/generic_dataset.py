from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import cv2
import os
import pycocotools.coco as coco
import torch
import time
from pathlib import Path
from torchvision.transforms import (
    ColorJitter,
    Normalize,
    Lambda,
    Compose,
    RandomOrder,
    ToTensor,
)
from lightning.pytorch.utilities import rank_zero_only

from utils.image import (
    getAffineTransform,
    affineTransform,
    lightingAug,
    getGaussianRadius,
    drawGaussianHeatRegion,
)
from utils.pointcloud import (
    getDistanceThresh,
    cvtPcDepthToHeatmap,
)
from utils.ddd import get3dBox, project3DPoints, draw3DBox
from utils.image import getGaussianRadius
from utils.utils import safe_run, createFolder, AverageMeter
from config import config, updateDatasetAndModelConfig
import warnings


class GenericDataset(torch.utils.data.Dataset):
    num_categories = None
    class_ids = None
    max_objs = None
    focalLength = 1200
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [4, 6],
        [3, 5],
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [6, 12],
        [5, 11],
        [11, 12],
        [12, 14],
        [14, 16],
        [11, 13],
        [13, 15],
    ]
    ignore_val = 1

    # change these vectors to actual mean and std to normalize
    pc_mean = np.zeros((18, 1))
    pc_std = np.ones((18, 1))
    imgDebugIndex = 0

    def __init__(self, config=None, split=None, ann_path=None, img_dir=None):
        super(GenericDataset, self).__init__()
        if config is not None and split is not None:
            self.split = split
            self.config = config
            self.enable_meta = (
                (config.TEST.OFFICIAL_EVAL and split in ["val", "mini_val", "test"])
                or config.EVAL
                or config.weights.bbox3d > 0
            )
            self.temporal = False

        if ann_path is not None and img_dir is not None:
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()
            self.img_dir = img_dir

        # initiaize the color augmentation
        self.colorAugmentor = Compose(
            [
                ToTensor(),
                RandomOrder(
                    [
                        ColorJitter(brightness=0.4),
                        ColorJitter(contrast=0.4),
                        ColorJitter(saturation=0.4),
                    ]
                ),
                Lambda(lightingAug),
                Normalize(self.mean, self.std),
            ]
        )
        self.sizeThresh = [
            [0, 0.0018, 0.0085][l]
            for l in range(len(self.config.MODEL.PYRAMID_OUT_SIZE))
        ]

    def __getitem__(self, index, prev=False):
        """
        item:
            image: image after augmentation
            calib: camera matrix
            pc_2d: radar point cloud in image coordinate
            pc_3d: radar point cloud in camera coordinate
            pc_N: number of radar point cloud
            pc_dep: radar point cloud depth
            pc_hm: radar point cloud heatmap [depth, vel_x, vel_z]
            heatmap: heatmap for each class
            classIds: class id
            mask: all mask show which data is valid
            truncMask: mask for truncation
            widthHeight: width and height of bounding box (w, h)
            reg: diffence between center(float) and center_int
            depth: depth(m)
            dimension: dimension of object (h, w, l)
            amodal_offset: amodal offset from bbox center
            nuscenes_att: nuscenes attribute (mask)
            nuscenes_att_mask: nuscenes attribute (range mask)
            velocity: velocity
            rotbin: rotation bin
            rotres: rotation residual
            meta: meta data
        """
        # ====== Load Image and Annotation ====== #
        img, anns, img_info, img_path = self.loadImageAnnotation(
            self.images[index], self.img_dir
        )
        center = np.array(
            [img_info["width"] / 2.0, img_info["height"] / 2.0], dtype=np.float32
        )
        if self.config.DATASET.MAX_CROP:
            scale = max(img_info["height"], img_info["width"]) * 1.0
        else:
            scale = np.array([img_info["width"], img_info["height"]], dtype=np.float32)
        if "calib" in img_info:
            calib = np.array(img_info["calib"], dtype=np.float32)
        else:
            calib = np.array(
                [
                    [self.focalLength, 0, img_info["width"] / 2, 0],
                    [0, self.focalLength, img_info["height"] / 2, 0],
                    [0, 0, 1, 0],
                ]
            )

        # ====== data augmentation for training set (image) ====== #
        scaleFactor, rotateFactor, isFliped = 1, 0, False
        if "train" in self.split:
            center, scaleFactor, rotateFactor = self.getAugmentParam(
                center, scale, img_info["width"], img_info["height"]
            )
            scale *= scaleFactor
            if np.random.random() < self.config.DATASET.FLIP:
                isFliped = True
                img = img[:, ::-1, :]
                anns = self.flipAnnotations(
                    anns,
                    img_info["width"],
                    getattr(img_info, "velocity_trans_matrix", None),
                )

        # ====== Get the affine transformation matrix ====== #
        transMatInput = getAffineTransform(
            center,
            scale,
            rotateFactor,
            [self.config.MODEL.INPUT_SIZE[1], self.config.MODEL.INPUT_SIZE[0]],
        )
        transMatOutput = getAffineTransform(
            center,
            scale,
            rotateFactor,
            [self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]],
        )
        item = {
            "image": self.transformInput(img, transMatInput),
            "calib": calib,
        }

        # ====== Load Radar Point Cloud ====== #
        if self.config.DATASET.RADAR_PC:
            pc_2d, pc_N, pc_dep, pc_3d = self.loadRadarPointCloud(
                img, img_info, transMatInput, transMatOutput, isFliped
            )
            item.update(
                {"pc_2d": pc_2d, "pc_3d": pc_3d, "pc_N": pc_N, "pc_dep": pc_dep}
            )

        if self.config.LOSS_WEIGHTS.LIDAR_DEPTH > 0:
            pc_lidar = self.loadLidarPointCloud(img_info, isFliped)
            item["pc_lidar"] = pc_lidar

        # ====== Initialize the return variables ====== #
        target = {}
        self.initReturn(item, target)
        num_objs = min(len(anns), self.max_objs)
        for i in range(num_objs):
            ann = anns[i]
            classId = int(self.class_ids[ann["category_id"]])

            if classId > self.num_categories or classId <= -999:
                continue
            bbox = self.transformBbox(ann["bbox"], transMatOutput)

            self.addInstance(
                item,
                target,
                i,
                classId - 1,
                bbox,
                ann,
                transMatOutput,
                scaleFactor,
            )

        if self.config.DATASET.RADAR_PC and not self.config.MODEL.FRUSTUM:
            item["pc_hm"] = item["pc_dep"].copy()
            # normalize depth
            maxDist = self.config.DATASET.MAX_PC_DIST
            if self.config.DATASET.ONE_HOT_PC:
                item["pc_hm"][: int(maxDist)] /= maxDist
                item["pc_hm"][: int(maxDist)] = 1 - item["pc_hm"][: int(maxDist)]
            else:
                item["pc_hm"][0] /= maxDist
                item["pc_hm"][0] = 1 - item["pc_hm"][0]
        item["target"] = target

        # ====== Get previous frame data ====== #
        if self.temporal and not prev:
            prev_id = img_info["prev_id"]
            prev_item = self.__getitem__(prev_id, prev=True)
            item["prev"] = prev_item

        # ====== Debug ====== #
        if self.config.DEBUG > 0 or self.enable_meta:
            # get velocity transformation matrix
            if "velocity_trans_matrix" in img_info:
                velocity_mat = np.array(
                    img_info["velocity_trans_matrix"], dtype=np.float32
                )
            else:
                velocity_mat = np.eye(4)

            meta = {
                "center": center,
                "scale": scale,
                "img_id": img_info["id"],
                "img_path": img_path,
                "img_width": img_info["width"],
                "img_height": img_info["height"],
                "isFliped": isFliped,
                "velocity_mat": velocity_mat,
                "target": target,
            }
            item["meta"] = meta

        return item

    def getDefaultCalib(self, width, height):
        """
        Get the default camera matrix.

        Args:
            width (int): image width
            height (int): image height

        Returns:
            np.array: default camera matrix
        """
        return np.array(
            [
                [self.focalLength, 0, width / 2, 0],
                [0, self.focalLength, height / 2, 0],
                [0, 0, 1, 0],
            ]
        )

    def loadImageAnnotation(self, img_id, img_dir):
        """
        Get the image, annotations, image info and image path.

        Args:
            img_id (int): image id
            img_dir (str): image directory

        Returns:
            tuple: image, annotations, image info and image path
        """
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def getBorder(self, border: int, size: int) -> int:
        """
        This function returns the smallest multiple of the border.

        Args:
            border (int): The border.
            size (int): The size of the region that the border contains.

        Returns:
            int: The smallest multiple of the border.
        """
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def getAugmentParam(self, center, scale, width, height):
        """
        Generate random augmentation parameters

        Args:
            center(list of int): The center of the bounding box before augmentation.
            scale(float): The scale of the bounding box before augmentation.
            width(int): The width of the image.
            height(int): The height of the image.

        Returns:
            list of int
                The center of the bounding box after augmentation.
            float
                The scale factor of the bounding box after augmentation.
            float
                The rotation factor of the bounding box after augmentation.
        """
        if self.config.DATASET.RANDOM_CROP:
            scaleFactor = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self.getBorder(128, width)
            h_border = self.getBorder(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

        else:
            scaleFactor = self.config.DATASET.SCALE
            shiftFactor = self.config.DATASET.SHIFT
            scaleFactor = np.clip(
                np.random.randn() * scaleFactor + 1, 1 - scaleFactor, 1 + scaleFactor
            )
            center[0] += scale * np.clip(
                np.random.randn() * shiftFactor, -2 * shiftFactor, 2 * shiftFactor
            )
            center[1] += scale * np.clip(
                np.random.randn() * shiftFactor, -2 * shiftFactor, 2 * shiftFactor
            )

        if np.random.random() < self.config.DATASET.ROTATE:
            rotateFactor = self.config.DATASET.ROTATE
            rotateFactor = np.clip(
                np.random.randn() * rotateFactor, -rotateFactor * 2, rotateFactor * 2
            )
        else:
            rotateFactor = 0

        return center, scaleFactor, rotateFactor

    def flipAnnotations(self, anns, width, vel_trans_mat=None):
        """
        This function flips the annotations horizontally.
        It does this by flipping the bounding boxes, the rotation angles,
        the amodal centers, and the velocities

        Args:
            anns (list): A list of annotations.
            width (int): The width of the image.

        Returns:
            A list of annotations that have been flipped horizontally.
        """
        for k in range(len(anns)):
            bbox = anns[k]["bbox"]
            anns[k]["bbox"] = [width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if "rotation" in self.config.heads and "alpha" in anns[k]:
                anns[k]["alpha"] = (
                    np.pi - anns[k]["alpha"]
                    if anns[k]["alpha"] > 0
                    else -np.pi - anns[k]["alpha"]
                )

            if "amodal_offset" in self.config.heads and "amodal_center" in anns[k]:
                anns[k]["amodal_center"][0] = width - anns[k]["amodal_center"][0] - 1

            if (
                self.config.DATASET.RADAR_PC
                and "velocity" in anns[k]
                and vel_trans_mat is not None
            ):
                anns[k]["velocity"][0] *= -1

                vel = anns[k]["velocity"]
                vel = np.array([vel[0], vel[1], vel[2], 0], np.float32)
                anns[k]["velocity_cam"] = np.dot(np.linalg.inv(vel_trans_mat), vel)

        return anns

    def transformInput(self, img, transformMat):
        """
        Affine transform the image and apply color augmentation

        Args:
            img: [HxWx3] image
            transformMat: affine transform matrix

        Returns:
            [CxHxW] transformed image
        """
        result = cv2.warpAffine(
            img,
            transformMat,
            (self.config.MODEL.INPUT_SIZE[1], self.config.MODEL.INPUT_SIZE[0]),
            flags=cv2.INTER_LINEAR,
        )
        result = result.astype(np.float32) / 255.0
        if "train" in self.split and self.config.DATASET.COLOR_AUG:
            result = self.colorAugmentor(result)
        else:
            result = (result - self.mean) / self.std
            result = result.transpose(2, 0, 1)
            result = torch.from_numpy(result)

        return result

    def initReturn(self, item, target):
        """
        Initialize the return dictionary

        Args:
            item (dict): The dictionary to be returned.
            target (dict): The dictionary containing the empty annotations.
                |- bboxes (np.array): [max_objs, 4] (xyxy)
                |- scores (np.array): [max_objs]
                |- centers (np.array): [max_objs, 2]
                |- widthHeight (np.array): [max_objs, 2]
                |- reg (np.array): [max_objs, 2]
                |- depth (np.array): [max_objs]
                |- dimension (np.array): [max_objs, 3]
                |- amodal_offset (np.array): [max_objs, 2]
                |- rotation (np.array): [max_objs, 8]
                |- bboxes3d (np.array): [max_objs, 8, 3]

        Returns:
            None
        """
        for i, (h, w) in enumerate(self.config.MODEL.PYRAMID_OUT_SIZE):
            item[f"heatmap{i}"] = np.zeros((self.num_categories, h, w), np.float32)
        item["classIds"] = np.zeros((self.max_objs), dtype=np.int64)
        item["mask"] = np.zeros((self.max_objs), dtype=np.float32)
        item["truncMask"] = np.zeros((self.max_objs), dtype=np.float32)
        item["widthHeight"] = np.zeros((self.max_objs, 2), dtype=np.float32)

        target["bboxes"] = np.zeros((self.max_objs, 4), dtype=np.float32)
        target["scores"] = np.zeros((self.max_objs), dtype=np.float32)
        target["centers"] = np.zeros((self.max_objs, 2), dtype=np.float32)
        target["heatCenters"] = np.zeros((self.max_objs, 2), dtype=np.float32)
        target["bboxes3d"] = np.zeros((self.max_objs, 8, 3), dtype=np.float32)

        regression_head_dims = {
            "reg": 2,
            "dimension": 3,
            "amodal_offset": 2,
        }

        for head in regression_head_dims:
            if head in self.config.heads:
                item[head] = np.zeros(
                    (self.max_objs, regression_head_dims[head]), dtype=np.float32
                )

        if {f"depth{i if i else ''}" for i in range(5)} & set(self.config.heads):
            item["depth"] = np.zeros((self.max_objs, 1), dtype=np.float32)

        if {f"rotation{i if i else ''}" for i in range(5)} & set(self.config.heads):
            item["rotbin"] = np.zeros((self.max_objs, 2), dtype=np.int64)
            item["rotres"] = np.zeros((self.max_objs, 2), dtype=np.float32)
            target["rotation"] = np.zeros((self.max_objs, 8), dtype=np.float32)

    def transformBbox(self, bbox, transMatOut):
        """
        Transform bbox according to affine transform matrix.

        Args:
            bbox: [x1, y1, w, h]
            transMatOut: affine transform matrix

        Returns:
            bbox: [x1, y1, x2, y2]
        """
        # convert bbox from [x1, y1, w, h] to [x1, y1, x2, y2]
        bbox = np.array(
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32
        )

        rect = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
            ]
        )
        rect[:] = affineTransform(rect, transMatOut)
        bbox = np.array(
            [rect[:, 0].min(), rect[:, 1].min(), rect[:, 0].max(), rect[:, 1].max()]
        )

        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.config.MODEL.OUTPUT_SIZE[1] - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.config.MODEL.OUTPUT_SIZE[0] - 1)
        return bbox

    def addInstance(
        self,
        item,
        target,
        i,
        classId,
        bbox,
        ann,
        transMatOutput,
        scaleFactor,
    ):
        """
        Add data instance into item and target.
        """
        height, width = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if height <= 0 or width <= 0:
            return
        center = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        center_int = center.astype(np.int32)

        outputHeight, outputWidth = self.config.MODEL.OUTPUT_SIZE
        item["classIds"][i] = classId
        item["mask"][i] = 1
        item["truncMask"][i] = ann["truncated"]

        # Get target layer for current bounding box
        boxAreaPercent = (height * width) / (outputHeight * outputWidth)
        for l in range(len(self.sizeThresh)):
            thresh = self.sizeThresh[l : l + 2]
            if len(thresh) == 1:
                layer = len(self.sizeThresh) - 1
            elif boxAreaPercent >= thresh[0] and boxAreaPercent < thresh[1]:
                layer = l
                break

        # Convert bounding box coordinate to target layer
        layerOutputHeight, layerOutputWidth = self.config.MODEL.PYRAMID_OUT_SIZE[layer]
        heightScale = layerOutputHeight / outputHeight
        widthScale = layerOutputWidth / outputWidth
        layerBoxHeight = height * heightScale
        layerBoxWidth = width * widthScale
        amodal_center = (
            affineTransform(
                np.array(ann["amodal_center"]).reshape(1, -1), transMatOutput
            )
            if "amodal_center" in ann
            else None
        )

        # Define the heatmap center
        objOutside = False
        if self.config.DATASET.HEATMAP_REP == "2d" or amodal_center is None:
            heatCenter = center * np.array([widthScale, heightScale])
        elif self.config.DATASET.HEATMAP_REP == "3d":
            heatCenter = amodal_center.reshape(-1).copy()
            heatCenter[0] = np.clip(heatCenter[0], 0, outputWidth - 1)
            heatCenter[1] = np.clip(heatCenter[1], 0, outputHeight - 1)
            if not (heatCenter == amodal_center).all():
                objOutside = True
        else:
            raise ValueError(
                f"Invalid heatmap representation: {self.config.DATASET.HEATMAP_REP}"
            )

        # Generate heatmap
        if objOutside:
            # for outside objects, generate 1-dimensional heatmap
            edge_heatmap_ratio = 0.5
            radius_x, radius_y = (
                layerBoxWidth * edge_heatmap_ratio,
                layerBoxHeight * edge_heatmap_ratio,
            )
            radius_x, radius_y = max(1, int(radius_x)), max(1, int(radius_y))
            radius = (radius_x, radius_y)
        else:
            # for inside objects, generate circular heatmap
            radius = getGaussianRadius(
                (math.ceil(layerBoxHeight), math.ceil(layerBoxWidth))
            )
            radius = max(0, int(radius))
        drawGaussianHeatRegion(item[f"heatmap{layer}"][classId], heatCenter, radius)

        target["bboxes"][i] = bbox
        target["centers"][i] = center
        target["heatCenters"][i] = heatCenter

        if "reg" in self.config.heads:
            item["reg"][i] = center - heatCenter

        if "amodal_offset" in self.config.heads and amodal_center is not None:
            item["amodal_offset"][i] = amodal_center - heatCenter
            if self.config.MODEL.NORM_2D:
                item["amodal_offset"][i] /= np.array([outputWidth, outputHeight])

        if "widthHeight" in item:
            item["widthHeight"][i] = (
                (width / outputWidth, height / outputHeight)
                if self.config.MODEL.NORM_2D
                else (width, height)
            )

        if (
            "nuscenes_att" in self.config.heads
            and ("attributes" in ann)
            and ann["attributes"] > 0
        ):
            att = int(ann["attributes"] - 1)
            item["nuscenes_att"][i][att] = 1
            item["nuscenes_att_mask"][i][self.nuscenes_att_range[att]] = 1

        if (
            "velocity" in self.config.heads
            and ("velocity_cam" in ann)
            and min(ann["velocity_cam"]) > -1000
        ):
            item["velocity"][i] = np.array(ann["velocity_cam"], np.float32)[:3]

        if "rotation" in self.config.heads:
            if "alpha" in ann:
                alpha = ann["alpha"]
                if alpha < np.pi / 6.0 or alpha > 5 * np.pi / 6.0:
                    item["rotbin"][i, 0] = 1
                    item["rotres"][i, 0] = alpha - (-0.5 * np.pi)
                if alpha > -np.pi / 6.0 or alpha < -5 * np.pi / 6.0:
                    item["rotbin"][i, 1] = 1
                    item["rotres"][i, 1] = alpha - (0.5 * np.pi)
                target["rotation"][i] = self.processAlpha(ann["alpha"])
            else:
                target["rotation"][i] = self.processAlpha(0)

        if "depth" in ann and {"depth", "depth2"} & set(self.config.heads):
            item["depth"][i] = ann["depth"] * scaleFactor

        if "dimension" in self.config.heads and "dimension" in ann:
            item["dimension"][i] = ann["dimension"]

        if {"dimension", "location", "yaw"} <= set(ann):
            target["bboxes3d"][i] = get3dBox(
                np.array(ann["dimension"]).reshape(1, 1, 3),
                np.array(ann["location"]).reshape(1, 1, 3),
                np.array(ann["yaw"]).reshape(1, 1),
            )

        if self.config.DATASET.RADAR_PC and self.config.MODEL.FRUSTUM:
            distanceThreshold = getDistanceThresh(
                item["calib"].reshape(1, 3, 4),
                center.reshape(1, 1, 2),
                np.array(ann["dimension"]).reshape(1, 1, 3),
                np.array(ann["alpha"]).reshape(1, 1, 1),
            )[0, 0]
            cvtPcDepthToHeatmap(
                item["pc_hm"],
                item["pc_dep"],
                ann["depth"],
                bbox,
                distanceThreshold,
                self.config.DATASET.MAX_PC_DIST,
            )

    def processAlpha(self, alpha):
        """
        Convert alpha to eight bins.

        Args:
            alpha: alpha angle

        Returns:
            ret(list): eight bins
        """
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6.0 or alpha > 5 * np.pi / 6.0:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6.0 or alpha < -5 * np.pi / 6.0:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def loadRadarPointCloud(self, *_):
        raise NotImplementedError

    def loadLidarPointCloud(self, *_):
        raise NotImplementedError

    def getDepthMap(self, maxDistance: int, isOneHot: bool) -> np.ndarray:
        """
        Due to the depthmap may be different in different dataset, we need to implement this function in the subclass.

        Args:
            maxDistance(int): the maximum distance of the point cloud
        """
        raise NotImplementedError

    def drawPcHeat(self, *_):
        """
        Draw the heatmap of the point cloud contained different values in different dataset,
        we need to implement this function in the subclass.
        """
        raise NotImplementedError

    def drawPcPoints(self, *_):
        """
        Draw the point cloud points in the heatmap, we need to implement this function in the subclass.
        """
        raise NotImplementedError

    def processPointCloud(
        self, pc_2d, pc_3d, img, transMatInput, transMatOutput, img_info
    ):
        """
        Process the point cloud data

        Args:
            pc_2d: the 2D point cloud data [x, y, d]
            pc_3d: the 3D point cloud data
            img: the original image
            transMatInput: the transformation matrix from the original image to the input image
            transMatOutput: the transformation matrix from the input image to the output image
            img_info: the image info

        Returns:
            pc_2d: masked 2D point cloud data [x, y, d]
            pc_3d: masked 3D point cloud data
            depthMap: the depthmap of the point cloud [d, vel_x, vel_z]
        """
        # initialize the depth map
        outputWidth, outputHeight = self.config.MODEL.OUTPUT_SIZE[::-1]
        transformedPoints, mask = self.transformPointCloud(
            pc_2d, transMatOutput, outputWidth, outputHeight
        )
        isOneHot = self.config.DATASET.ONE_HOT_PC
        maxDistance = int(self.config.DATASET.MAX_PC_DIST)
        depthMap = self.getDepthMap(maxDistance, isOneHot)

        if mask is not None:
            pc_N = sum(mask)
            pc_2d = pc_2d[:, mask]
            pc_3d = pc_3d[:, mask]
        else:
            pc_N = pc_2d.shape[1]

        # generate point cloud channels
        if self.config.DATASET.PC_ROI_METHOD == "pillars":
            boxesInput2D, pillar_wh = self.getPcPillarsSize(
                img_info, pc_3d, transMatInput, transMatOutput
            )
            if self.config.DEBUG:
                self.debugPillar(
                    img,
                    pc_2d,
                    transMatInput,
                    transMatOutput,
                    boxesInput2D,
                    pillar_wh,
                )
        elif self.config.DATASET.PC_ROI_METHOD == "points":
            depthMap = self.drawPcPoints(
                depthMap,
                transformedPoints[:2],  # x, y
                transformedPoints[2],  # depth
                maxDistance,
                isOneHot,
                pc_3d,
            )
            return transformedPoints, pc_3d, depthMap

        for i in range(pc_N):
            point = transformedPoints[:, i]
            depth = point[2]
            center = point[:2]
            method = self.config.DATASET.PC_ROI_METHOD
            if method == "pillars":
                box = [
                    max(center[1] - pillar_wh[1, i], 0),  # y1
                    center[1],  # y2
                    max(center[0] - pillar_wh[0, i] / 2, 0),  # x1
                    min(center[0] + pillar_wh[0, i] / 2, outputWidth),  # x2
                ]

            elif method == "heatmap":
                radius = (1.0 / depth) * 250 + 5
                radius = getGaussianRadius((radius, radius))
                radius = max(0, int(radius))
                x, y = int(center[0]), int(center[1])
                left, right = min(x, radius), min(outputWidth - x, radius + 1)
                top, bottom = min(y, radius), min(outputHeight - y, radius + 1)
                box = [y - top, y + bottom, x - left, x + right]

            else:
                raise ValueError(f"Invalid PC_ROI_METHOD: {method}")

            box = np.round(box).astype(np.int32)
            depthMap = self.drawPcHeat(
                depthMap, box, depth, maxDistance, isOneHot, pc_3d[:, i]
            )

        return transformedPoints, pc_3d, depthMap

    def transformPointCloud(
        self, pc_2d, transformMat, img_width, img_height, filter_out=True
    ):
        """
        Transform 2D point cloud using transformation matrix

        Args:
            pc_2d: 2D point cloud # [x, y] (2, N) or [x, y, d] (3, N)
            transformMat: transformation matrix (2, 3)
            img_width(int): output image width
            img_height(int): output image height
            filter_out: filter out points outside image

        Returns:
            out: transformed points # [x, y] (2, N) or [x, y, d] (3, N)
            mask: filtered points
        """
        if pc_2d.shape[1] == 0:
            return pc_2d, []

        pc_t = np.expand_dims(pc_2d[:2, :].T, 0)  # [3,N] -> [1,N,2]
        transformedPoints = cv2.transform(pc_t, transformMat)
        transformedPoints = np.squeeze(transformedPoints, 0).T  # [1,N,2] -> [2,N]

        # remove points outside image
        if filter_out:
            mask = (
                (transformedPoints[0, :] < img_width)
                & (transformedPoints[1, :] < img_height)
                & (0 < transformedPoints[0, :])
                & (0 < transformedPoints[1, :])
            )
            out = np.concatenate((transformedPoints[:, mask], pc_2d[2:, mask]), axis=0)
        else:
            mask = None
            out = np.concatenate((transformedPoints, pc_2d[2:, :]), axis=0)

        return out, mask

    def getPcPillarsSize(self, img_info, pc_3d, transMatInput, transMatOutput):
        """
        Get the size of the point cloud pillars for every point

        Args:
            img_info: image infomation
            pc_3d: 3D point cloud in camera coordinate [x, y, z] (>=3, N)
            transMatInput: transformation matrix from origin image to input size
            transMatOutput: transformation matrix from input to output size

        Returns:
            pillar_wh: width and height of the point cloud pillars [w, h] (2, N)
        """
        pillar_dims = self.config.DATASET.PILLAR_DIMS
        boxesInput2D = None  # for debug

        # for i, center in enumerate(pc_3d[:3, :].T):
        centers = pc_3d[:3, :].T
        B, K = 1, len(centers)
        pillar_dims = np.array(pillar_dims).reshape(1, 1, 3)
        pillar_dims = pillar_dims.repeat(K, 1)  # (B, K, 3)

        # Create a 3D pillar at pc location for the full-size image
        centers = np.array(centers).reshape(B, K, 3)  # (B, K, 3)
        boxOrigin3D = get3dBox(pillar_dims, centers, np.zeros((B, K)))  # (B, K, 8, 3)
        calib = np.array(img_info["calib"]).reshape(1, 1, 3, 4)
        calib = calib.repeat(K, 1)  # (B, K, 3, 4)
        boxOrigin2D = project3DPoints(boxOrigin3D, calib)  # (B, K, 8, 2)
        pointsOrigin2D = boxOrigin2D.reshape((-1, 2)).T  # (B, K, 8, 2) -> (2, B*K*8)

        # save the box for debug plots
        if self.config.DEBUG:
            pointsInput2D, _ = self.transformPointCloud(
                pointsOrigin2D,
                transMatInput,
                self.config.MODEL.INPUT_SIZE[1],
                self.config.MODEL.INPUT_SIZE[0],
                filter_out=False,
            )  # (2, B*K*8)
            boxesInput2D = pointsInput2D.T.reshape(
                (-1, 8, 2)
            )  # (2, B*K*8) -> (B * K, 8, 2)

        # transform points
        pointsOutput2D, _ = self.transformPointCloud(
            pointsOrigin2D,
            transMatOutput,
            self.config.MODEL.OUTPUT_SIZE[1],
            self.config.MODEL.OUTPUT_SIZE[0],
            filter_out=False,
        )  # (2, B*K*8)

        boxOutput2D = pointsOutput2D.T.reshape(
            (B, -1, 8, 2)
        )  # (2, B*K*8) -> (B, K, 8, 2)

        # get the bounding box in [xyxy] format
        bbox = np.stack(
            [
                np.min(boxOutput2D[:, :, :, 0], 2),
                np.min(boxOutput2D[:, :, :, 1], 2),
                np.max(boxOutput2D[:, :, :, 0], 2),
                np.max(boxOutput2D[:, :, :, 1], 2),
            ],
            axis=-1,
        )  # (B, K, 4)

        # store height and width of the 2D box
        # pillar_wh = np.zeros((2, pc_3d.shape[1]))
        pillar_wh = np.concatenate(
            [bbox[:, :, 2] - bbox[:, :, 0], bbox[:, :, 3] - bbox[:, :, 1]]
        )

        return boxesInput2D, pillar_wh

    @rank_zero_only
    @safe_run
    def logValidResult(self, logger, output_dir):
        """
        This function will log result from file to logger and console.

        Args:
            logger: logger object
            output_dir: output directory

        Returns:
            None
        """
        raise NotImplementedError

    def debugPillar(
        self, img, pc_2d, transMatInput, transMatOutput, boxesInput2D, pillar_wh
    ):
        """
        This function is used to debug the point cloud pillars.
        It plots the pillars, radar point cloud on the original image, input image, and output image.
        And output the images to the debug directory or show directly.

        Args:
            img: the original image
            pc_2d: the 2D point cloud data [x, y, d]
            transMatInput: the transformation matrix from the original image to the input image
            transMatOutput: the transformation matrix from the original image to the output image
            boxesInput2D: the 3D bounding boxes of the point pillars on the input image
            pillar_wh: the width and height of the point pillars

        Returns:
            None
        """
        inputHeight, inputWidth = (
            self.config.MODEL.INPUT_SIZE[0],
            self.config.MODEL.INPUT_SIZE[1],
        )
        outputHeight, outputWidth = (
            self.config.MODEL.OUTPUT_SIZE[0],
            self.config.MODEL.OUTPUT_SIZE[1],
        )

        imgOrigin = img.copy()

        imgInput2D = cv2.warpAffine(
            img,
            transMatInput,
            (inputWidth, inputHeight),
            flags=cv2.INTER_LINEAR,
        )
        imgOutput2D = cv2.warpAffine(
            img,
            transMatOutput,
            (outputWidth, outputHeight),
            flags=cv2.INTER_LINEAR,
        )
        imgInput3D = cv2.warpAffine(
            img,
            transMatInput,
            (inputWidth, inputHeight),
            flags=cv2.INTER_LINEAR,
        )

        originMask = np.ones((*imgOrigin.shape[:2], 3), np.uint8) * 255
        imgOverlayInput = imgInput2D.copy()
        imgOverlayOrigin = img.copy()

        pcInput, _ = self.transformPointCloud(
            pc_2d, transMatInput, inputWidth, inputHeight
        )
        pcInput = pcInput[:3, :].T
        # pcOutput, _ = self.transformPointCloud(
        #     pc_2d, transMatOutput, outputWidth, outputHeight
        # )

        pillarInputWh = pillar_wh * (inputWidth / outputWidth)
        # pillarOutputWh = pillar_wh
        pillarOriginWh = pillarInputWh * 2

        colors = cv2.applyColorMap(
            ((pcInput[:, 2] / 60) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        for i in range(len(pcInput) - 1, -1, -1):
            point = pcInput[i]
            # color = int((point[2] / self.config.DATASET.MAX_PC_DIST) * 255)
            # color = (0, color, 0)
            color = colors[i, 0].tolist()

            # Draw pillar box on input image
            pillarTopLInput = (
                max(int(point[0] - pillarInputWh[0, i] / 2), 0),
                max(int(point[1] - pillarInputWh[1, i]), 0),
            )
            pillarBotRInput = (
                min(int(point[0] + pillarInputWh[0, i] / 2), inputWidth),
                int(point[1]),
            )

            # Draw radar point on input image
            imgInput2D = cv2.circle(
                imgInput2D, (int(point[0]), int(point[1])), 3, color, -1
            )

            # Draw pillar box on original image
            pillarTopLOrigin = (
                max(int(pc_2d[0, i] - pillarOriginWh[0, i] / 2), 0),
                max(int(pc_2d[1, i] - pillarOriginWh[1, i]), 0),
            )
            pillarBotROrigin = (
                min(int(pc_2d[0, i] + pillarOriginWh[0, i] / 2), imgOrigin.shape[1]),
                int(pc_2d[1, i]),
            )

            # Draw radar point on original image
            imgOrigin = cv2.circle(
                imgOrigin, (int(pc_2d[0, i]), int(pc_2d[1, i])), 6, color, -1
            )

            # Draw radar point on output image
            imgOutput2D = cv2.circle(
                imgOutput2D, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1
            )

            # Draw pillar box on blank image
            cv2.rectangle(
                originMask,
                pillarTopLOrigin,
                pillarBotROrigin,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

            # plot 3d pillars
            imgInput3D = draw3DBox(
                imgInput3D,
                boxesInput2D[i].astype(np.int32),
                [114, 159, 207],
                same_color=False,
            )

            # Overlay pillar mask on input image
            cv2.rectangle(
                imgOverlayInput,
                pillarTopLInput,
                pillarBotRInput,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

            # Overlay pillar mask on original image
            cv2.rectangle(
                imgOverlayOrigin,
                pillarTopLOrigin,
                pillarBotROrigin,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

        # ==================== Output ====================
        debugDir = os.path.join(self.config.OUTPUT_DIR, "debug")
        outputFunc = cv2.imshow
        if self.config.DEBUG > 1:
            if not os.path.exists(debugDir):
                os.makedirs(debugDir)
            outputFunc = lambda n, img: cv2.imwrite(n, img) and cv2.imshow(n, img)

        dirHead = os.path.join(debugDir, f"{self.imgDebugIndex}_")
        outputFunc(f"{dirHead}pillarInput2D.jpg", imgInput2D)
        outputFunc(f"{dirHead}pillarOrigin2D.jpg", imgOrigin)
        outputFunc(f"{dirHead}pillarOutput2D.jpg", imgOutput2D)
        outputFunc(f"{dirHead}pillarInputOverlay.jpg", imgOverlayInput)
        outputFunc(f"{dirHead}pillarOriginOverlay.jpg", imgOverlayOrigin)
        outputFunc(f"{dirHead}pillarInput3D.jpg", imgInput3D)
        outputFunc(f"{dirHead}pillarOriginMask.jpg", originMask)
        outputFunc(f"{dirHead}imgOrigin.jpg", img)
        self.imgDebugIndex += 1 if outputFunc != cv2.imshow else 0

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            exit()

    @staticmethod
    def getDemoInstance(args):
        raise NotImplementedError


class GenericDemo:
    TIME_STATS = [
        "total",
        "load",
        "preprocess",
        "net",
        "decode",
        "postprocess",
        "merge",
        "display",
    ]

    def __init__(self, args):
        from detector import Detector
        from dataset import getDataset

        if args.sample is not None and args.max > 0:
            warnings.warn("Scene demo will be ignored when sample is specified.")

        time_str = time.strftime("%Y-%m-%d-%H-%M")
        self.output_dir = Path("output") / "Demo" / time_str
        createFolder(self.output_dir, parents=True, exist_ok=True)
        dataset = getDataset(config.DATASET.DATASET)
        updateDatasetAndModelConfig(config, dataset, str(self.output_dir))
        self.detector = Detector(config, show=False, pause=False)
        self.detector.visualization = args.save or not args.not_show
        self.args = args
        self.writer = {}
        self.fps = None
        self.averageFps = AverageMeter()
        self.frameTimeStats = {stat: AverageMeter() for stat in self.TIME_STATS}

    def run(self, args):
        raise NotImplementedError

    def initWriter(self, **outputSizes):
        # Release previous writer
        for key in self.writer:
            if self.writer[key] is not None:
                self.writer[key].release()
        self.writer.clear()

        # Create new writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for key, shape in outputSizes.items():
            self.writer[key] = cv2.VideoWriter(
                os.path.join(self.output_dir, f"demo_{key}.mp4"),
                fourcc,
                self.fps,
                shape,
            )

    def updateTimeBar(self, modelRet):
        """
        Update the output time bar with the time statistics.

        Args:
            modelRet: the output of the detector

        Returns:
            None
        """
        time_str = "InferenceTime: "
        for stat in self.TIME_STATS:
            self.frameTimeStats[stat].update(modelRet[stat])
            time_str = time_str + "{} {:.3f}s | ".format(
                stat, self.frameTimeStats[stat].avg
            )
        self.averageFps.update(1 / modelRet["total"])
        time_str += f" {self.averageFps.avg:.1f}fps"
        print(time_str, end="\r", flush=True)

    def showAttention(self, depthmaps, allCamImages, allPcHmOut):
        """
        Show attention maps overlay on the original image.
        Currently support only single camera image.

        Args:
            depthmaps:  dictionary containing the attention maps
                        get from the model output ret["depthmaps"]
            allCamImages: the original image
            allPcHmOut: the heatmap of the point cloud
        """
        if not self.args.show_attention:
            return

        if not self.args.single:
            warnings.warn(
                "Visualization of attention map currently not supported for multiple camera images."
            )
            return

        for attHead in ["depth", "rotation", "velocity", "nuscenes_att"]:
            for attKey in ["AttImg", "AttRadar"]:
                attentionOutput = f"{attHead}{attKey}"
                if attentionOutput in depthmaps:
                    smallImage = cv2.resize(allCamImages, allPcHmOut.shape[1::-1])
                    attMap = cv2.applyColorMap(
                        depthmaps[attentionOutput][0], cv2.COLORMAP_JET
                    )
                    attMap = cv2.addWeighted(attMap, 0.5, smallImage, 1, 0)
                    cv2.imshow(attentionOutput, attMap)
