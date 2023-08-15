from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import cv2
import os
from collections import defaultdict
import pycocotools.coco as coco
import torch
from timeit import default_timer as timer
from torchvision.transforms import ColorJitter, Normalize, Lambda, Compose, RandomOrder

from utils.image import (
    getAffineTransform,
    affineTransform,
    lightingAug,
    getGaussianRadius,
    drawGaussianHeatRegion,
)
from utils.pointcloud import (
    map_pointcloud_to_image,
    getDistanceThresh,
    cvtPcDepthToHeatmap,
)
from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from utils.ddd import get3dBox, project3DPoints
from utils.image import getGaussianRadius

# from utils.image import flip, color_aug
# from utils.image import affine_transform
# from utils.pointcloud import map_pointcloud_to_image, pc_dep_to_hm
# from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
# from nuscenes.utils.data_classes import Box
# from pyquaternion import Quaternion
# from nuscenes.utils.geometry_utils import view_points
# from utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d
# from utils.ddd_utils import comput_corners_3d, alpha2rot_y, get_pc_hm


class GenericDataset(torch.utils.data.Dataset):
    # default_resolution = None
    # class_name = None
    num_categories = None
    class_ids = None
    max_objs = None
    focalLength = 1200
    # num_joints = 17
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
    img_ind = 0

    def __init__(
        self, config=None, split=None, ann_path=None, img_dir=None, device=None
    ):
        super(GenericDataset, self).__init__()
        if config is not None and split is not None:
            self.split = split
            self.config = config
            self.enable_meta = (
                config.TEST.OFFICIAL_EVAL and split in ["val", "mini_val", "test"]
            ) or config.EVAL

        if ann_path is not None and img_dir is not None:
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()
            self.img_dir = img_dir

        self.device = device if device is not None else torch.device("cpu")

        # initiaize the color augmentation
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self.colorAugmentor = Compose(
            [
                RandomOrder(
                    [
                        ColorJitter(brightness=0.4),
                        ColorJitter(contrast=0.4),
                        ColorJitter(saturation=0.4),
                    ]
                ),
                Lambda(lightingAug),
                Normalize(mean, std),
            ]
        )

    def __getitem__(self, index):
        """
        item:
            image: image after augmentation
            classIds: class id
            mask: TODO
            widthHeight: width and height of bounding box (w, h)
            wh_mask: TODO
            index: index for get feature map
            reg: TODO
            reg_mask: TODO
            heatmap: heatmap for each class
            nuscenes_att: nuscenes attribute (mask)
            nuscenes_att_mask: nuscenes attribute (range mask)
            velocity: velocity
            velocity_mask: velocity mask
            depth: depth TODO:unit?
            depth_mask: depth mask
            dimension: dimension of object (w, h, l) TODO
            dimension_mask: dimension mask
            amodal_offset: amodal offset from bbox center
            amodal_offset_mask: amodal offset mask
            pc_hm: radar point cloud heatmap [depth, vel_x, vel_z]
            pc_dep: radar point cloud depth
            rotbin: rotation bin
            rotres: rotation residual
            rot_mask: rotation mask

        target:
            bboxes
            scores
            classIds
            centers
            nuscenes_att
            velocity
            depth
            dimension
            amodal_offset
            reg
            rotation
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
        scaleFactor, rotateFactor, isFliped = 1, 0, True
        if "train" in self.split:
            center, scaleFactor, rotateFactor = self.getAugmentParam(
                center, scale, img_info["width"], img_info["height"]
            )
            scale *= scaleFactor
            if np.random.random() < self.config.DATASET.FLIP:
                isFliped = True
                img = img[:, ::-1, :]
                anns = self.filpAnnotations(anns, img_info["width"])

        # ====== Get the affine transformation matrix ====== #
        transMatInput = getAffineTransform(
            center,
            scaleFactor,
            rotateFactor,
            [self.config.MODEL.INPUT_SIZE[1], self.config.MODEL.INPUT_SIZE[0]],
        )
        transMatOutput = getAffineTransform(
            center,
            scaleFactor,
            rotateFactor,
            [self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]],
        )

        # ====== Load Radar Point Cloud ====== #
        if self.config.DATASET.NUSCENES.RADAR_PC:
            pc_2d, pc_N, pc_dep, pc_3d = self.loadRadarPointCloud(
                img, img_info, transMatInput, transMatOutput, isFliped
            )
            item.update(
                {"pc_2d": pc_2d, "pc_3d": pc_3d, "pc_N": pc_N, "pc_dep": pc_dep}
            )

        # ====== Initialize the return variables ====== #
        item = {
            "image": self.transformInput(img, transMatInput),
            "calib": calib,
        }
        target = {"bboxes": [], "scores": [], "classIds": [], "centers": []}
        self.initReturn(item, target)
        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            classId = int(self.class_ids[ann["category_id"]])

            if classId > self.num_categories or classId <= -999:
                continue
            bbox = self.transformBbox(ann["bbox"], transMatOutput)

            if classId <= 0 or ("iscrowd" in ann and ann["iscrowd"] > 0):
                self._mask_ignore_or_crowd(item, classId, bbox)
                continue

            self.addInstance(
                item,
                target,
                k,
                classId - 1,
                bbox,
                ann,
                transMatOutput,
                scaleFactor,
            )

        # ====== Debug ====== #
        if self.config.DEBUG > 0 or self.enable_meta:
            target = self._format_gt_det(target)

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
                "target": target,
                "img_id": img_info["id"],
                "img_path": img_path,
                "calib": calib,
                "img_width": img_info["width"],
                "img_height": img_info["height"],
                "isFliped": isFliped,
                "velocity_mat": velocity_mat,
            }
            item["meta"] = meta

        return item

    def get_default_calib(self, width, height):
        return np.array(
            [
                [self.focalLength, 0, width / 2, 0],
                [0, self.focalLength, height / 2, 0],
                [0, 0, 1, 0],
            ]
        )

    def loadImageAnnotation(self, img_id, img_dir):
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

    def filpAnnotations(self, anns, width):
        """
        This function flips the annotations horizontally.
        It does this by flipping the bounding boxes, the rotation angles,
        the amodel centers, and the velocities

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

            # TODO: nannotation typo amodel -> amodal
            if "amodal_offset" in self.config.heads and "amodel_center" in anns[k]:
                anns[k]["amodel_center"][0] = width - anns[k]["amodel_center"][0] - 1

            if self.config.LOSS.VELOCITY and "velocity" in anns[k]:
                anns[k]["velocity"][0] *= -1

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
        # t1 = timer()
        result = cv2.warpAffine(
            img,
            transformMat,
            (self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]),
            flags=cv2.INTER_LINEAR,
        )
        # t2 = timer()

        result = result.astype(np.float32) / 255.0
        result = result.transpose(2, 0, 1)
        result = torch.from_numpy(result).to(self.device)
        if "train" in self.split and self.config.DATASET.COLOR_AUG:
            # t3 = timer()
            result = self.colorAugmentor(result)
            # t4 = timer()
        # t5 = timer()
        # print('\n' + f'warpAffine: {(t2 - t1) * 1000:.1f}ms')
        # print(f'color_aug: {(t4 - t3) * 1000:.1f}ms')
        # print(f'other: {(t5 - t4) * 1000:.1f}ms')
        return result

    def initReturn(self, item, target):
        item["heatmap"] = np.zeros(
            (
                self.num_categories,
                self.config.MODEL.OUTPUT_SIZE[0],
                self.config.MODEL.OUTPUT_SIZE[1],
            ),
            np.float32,
        )
        item["index"] = np.zeros((self.max_objs), dtype=np.int64)
        item["classIds"] = np.zeros((self.max_objs), dtype=np.int64)
        item["mask"] = np.zeros((self.max_objs), dtype=np.float32)

        if self.config.DATASET.NUSCENES.RADAR_PC:
            item["pc_hm"] = np.zeros(
                (
                    3,
                    self.config.MODEL.OUTPUT_SIZE[0],
                    self.config.MODEL.OUTPUT_SIZE[1],
                ),
                np.float32,
            )

        regression_head_dims = {
            "widthHeight": 2,
            "reg": 2,
            "depth": 1,
            "dimension": 3,
            "amodal_offset": 2,
            "nuscenes_att": 8,
            "velocity": 3,
        }

        for head in regression_head_dims:
            if head in self.config.heads:
                item[head] = np.zeros(
                    (self.max_objs, regression_head_dims[head]), dtype=np.float32
                )
                item[head + "_mask"] = np.zeros(
                    (self.max_objs, regression_head_dims[head]), dtype=np.float32
                )
                target[head] = []

        if "rotation" in self.config.heads:
            item["rotbin"] = np.zeros((self.max_objs, 2), dtype=np.int64)
            item["rotres"] = np.zeros((self.max_objs, 2), dtype=np.float32)
            item["rot_mask"] = np.zeros((self.max_objs), dtype=np.float32)
            target.update({"rotation": []})

    def _mask_ignore_or_crowd(self, item, classId, bbox):
        """
        Mask out specific region(bbox) in heatmap.
        Only single class is masked out if classId is specified.

        Args:
            item(dict): data item
            classId(int): class id
            bbox(array): [x1, y1, x2, y2]

        Returns:
            None
        """
        ignore_val = 1
        if classId == 0:
            # ignore all classes
            # mask out crowd region
            region = item["heatmap"][
                :, int(bbox[1]) : int(bbox[3]) + 1, int(bbox[0]) : int(bbox[2]) + 1
            ]
        else:
            # mask out one specific class
            region = item["heatmap"][
                abs(classId) - 1,
                int(bbox[1]) : int(bbox[3]) + 1,
                int(bbox[0]) : int(bbox[2]) + 1,
            ]
        np.maximum(region, ignore_val, out=region)

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
        for i in range(4):
            rect[i] = affineTransform(rect[i], transMatOut)
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
        k,
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
        radius = getGaussianRadius((math.ceil(height), math.ceil(width)))
        radius = max(0, int(radius))
        center = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        center_int = center.astype(np.int32)
        item["classIds"][k] = classId
        item["mask"][k] = 1
        if "widthHeight" in item:
            item["widthHeight"][k] = float(width), float(height)
            item["wh_mask"][k] = 1
        item["index"][k] = (
            center_int[1] * self.config.MODEL.OUTPUT_SIZE[1] + center_int[0]
        )
        item["reg"][k] = center - center_int
        item["reg_mask"][k] = 1
        drawGaussianHeatRegion(item["heatmap"][classId], center_int, radius)

        target["bboxes"].append(
            np.array(
                [
                    center[0] - width / 2,
                    center[1] - height / 2,
                    center[0] + width / 2,
                    center[1] + height / 2,
                ],
                dtype=np.float32,
            )
        )
        target["scores"].append(1)
        target["classIds"].append(classId)
        target["centers"].append(center)

        if "nuscenes_att" in self.config.heads:
            if ("attributes" in ann) and ann["attributes"] > 0:
                att = int(ann["attributes"] - 1)
                item["nuscenes_att"][k][att] = 1
                item["nuscenes_att_mask"][k][self.nuscenes_att_range[att]] = 1
            target["nuscenes_att"].append(item["nuscenes_att"][k])

        if "velocity" in self.config.heads:
            if ("velocity_cam" in ann) and min(ann["velocity_cam"]) > -1000:
                item["velocity"][k] = np.array(ann["velocity_cam"], np.float32)[:3]
                item["velocity_mask"][k] = 1
            target["velocity"].append(item["velocity"][k])

        if "rotation" in self.config.heads:
            self._add_rot(item, ann, k, target)

        if "depth" in self.config.heads and "depth" in ann:
            item["depth"][k] = ann["depth"] * scaleFactor
            item["depth_mask"][k] = 1
            target["depth"].append(item["depth"][k])

        if "dimension" in self.config.heads:
            if "dimension" in ann:
                item["dimension"][k] = ann["dimension"]
                item["dimension_mask"][k] = 1
                target["dimension"].append(item["dimension"][k])
            else:
                target["dimension"].append([1, 1, 1])

        if "amodal_offset" in self.config.heads:
            if "amodal_center" in ann:
                amodel_center = affineTransform(ann["amodal_center"], transMatOutput)
                item["amodal_offset"][k] = amodel_center - center_int
                item["amodal_offset_mask"][k] = 1
                target["amodal_offset"].append(item["amodal_offset"][k])
            else:
                target["amodal_offset"].append([0, 0])

        if self.config.DATASET.NUSCENES.RADAR_PC:
            if self.opt.disable_frustum:
                item["pc_hm"] = item["pc_dep"]
                # normalize depth
                item["pc_hm"][0] /= self.config.DATASET.NUSCENES.MAX_PC_DIST
            else:
                distanceThreshold = getDistanceThresh(
                    item["calib"], center, ann["dim"], ann["alpha"]
                )
                cvtPcDepthToHeatmap(
                    item["pc_hm"],
                    item["pc_dep"],
                    ann["depth"],
                    bbox,
                    distanceThreshold,
                    self.config.DATASET.NUSCENES.MAX_PC_DIST,
                )

    def _add_rot(self, ret, ann, k, gt_det):
        if "alpha" in ann:
            ret["rot_mask"][k] = 1
            alpha = ann["alpha"]
            if alpha < np.pi / 6.0 or alpha > 5 * np.pi / 6.0:
                ret["rotbin"][k, 0] = 1
                ret["rotres"][k, 0] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6.0 or alpha < -5 * np.pi / 6.0:
                ret["rotbin"][k, 1] = 1
                ret["rotres"][k, 1] = alpha - (0.5 * np.pi)
            gt_det["rot"].append(self._alpha_to_8(ann["alpha"]))
        else:
            gt_det["rot"].append(self._alpha_to_8(0))

    def _alpha_to_8(self, alpha):
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

    def _format_gt_det(self, gt_det):
        if len(gt_det["scores"]) == 0:
            gt_det = {
                "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
                "scores": np.array([1], dtype=np.float32),
                "classIds": np.array([0], dtype=np.float32),
                "centers": np.array([[0, 0]], dtype=np.float32),
                "pre_cts": np.array([[0, 0]], dtype=np.float32),
                "tracking": np.array([[0, 0]], dtype=np.float32),
                "bboxes_amodal": np.array([[0, 0]], dtype=np.float32),
                "hps": np.zeros((1, 17, 2), dtype=np.float32),
            }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset["videos"] = []
        for i in range(len(self.coco.dataset["images"])):
            img_id = self.coco.dataset["images"][i]["id"]
            self.coco.dataset["images"][i]["video_id"] = img_id
            self.coco.dataset["images"][i]["frame_id"] = 1
            self.coco.dataset["videos"].append({"id": img_id})

        if not ("annotations" in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset["annotations"])):
            self.coco.dataset["annotations"][i]["track_id"] = i + 1

    def loadRadarPointCloud(
        self, img, img_info, transMatInput, transMatOutput, isFlipped=False
    ):
        """
        Load the Radar point cloud data

        Args:
            img: the original image
            img_info: the image info
            transMatInput: the transformation matrix from the original image to the input image
            transMatOutput: the transformation matrix from the input image to the output image
            isFlipped: whether the image is flipped

        Returns: TODO
            pc_2d: the 2D point cloud data [x, y, d]
            pc_N: the amount of points in 2D image
            pc_dep: the depth feature map [d, vel_x, vel_z]
            pc_3d: the 3D point cloud data
        """
        # Load radar point cloud from files
        sensor_name = self.SENSOR_NAME[img_info["sensor_id"]]
        sample_token = img_info["sample_token"]
        sample = self.nusc.get("sample", sample_token)
        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
        for radar_channel in self.RADARS_FOR_CAMERA[sensor_name]:
            radar_pcs, _ = RadarPointCloud.from_file_multisweep(
                self.nusc, sample, radar_channel, sensor_name, self.NUM_SWEEPS
            )
            all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))
        radar_pc = all_radar_pcs.points
        if radar_pc is None:
            return None, None, None, None

        # get distance to points
        depth = radar_pc[2, :]
        maxDistance = self.config.DATASET.NUSCENES.MAX_PC_DIST

        # filter points by distance
        if maxDistance > 0:
            mask = depth <= maxDistance
            radar_pc = radar_pc[:, mask]
            depth = depth[mask]

        # add z offset to radar points / raise all Radar points in z direction
        if self.config.DATASET.NUSCENES.PC_Z_OFFSET != 0:
            radar_pc[1, :] -= self.config.DATASET.NUSCENES.PC_Z_OFFSET

        # map points to the image and filter ones outside
        pc_2d, mask = map_pointcloud_to_image(
            radar_pc,
            np.array(img_info["camera_intrinsic"]),
            img_shape=(img_info["width"], img_info["height"]),
        )
        pc_3d = radar_pc[:, mask]

        # sort points by distance
        index = np.argsort(pc_2d[2, :])
        pc_2d = pc_2d[:, index]
        pc_3d = pc_3d[:, index]

        # flip points if image is flipped
        if isFlipped:
            pc_2d[0, :] = img.shape[1] - 1 - pc_2d[0, :]
            pc_3d[0, :] *= -1  # flipping the x dimension
            pc_3d[8, :] *= -1  # flipping x velocity (x is right, z is front)

        pc_2d, pc_3d, pc_dep = self.processPointCloud(
            pc_2d, pc_3d, img, transMatInput, transMatOutput, img_info
        )
        pc_N = np.array(pc_2d.shape[1])

        # pad point clouds with zero to avoid size mismatch error in dataloader
        n_points = min(self.opt.max_pc, pc_2d.shape[1])
        pc_z = np.zeros((pc_2d.shape[0], self.opt.max_pc))
        pc_z[:, :n_points] = pc_2d[:, :n_points]
        pc_3dz = np.zeros((pc_3d.shape[0], self.opt.max_pc))
        pc_3dz[:, :n_points] = pc_3d[:, :n_points]

        return pc_z, pc_N, pc_dep, pc_3dz

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
        outputWidth, outputHeight = (
            self.config.MODEL.OUTPUT_SIZE[1],
            self.config.MODEL.OUTPUT_SIZE[0],
        )
        transformedPoints, mask = self.transformPointCloud(
            pc_2d, transMatOutput, outputWidth, outputHeight
        )
        depthMap = np.zeros((3, outputHeight, outputWidth), np.float32)

        if mask is not None:
            pc_N = sum(mask)
            pc_2d = pc_2d[:, mask]
            pc_3d = pc_3d[:, mask]
        else:
            pc_N = pc_2d.shape[1]

        # create point cloud pillars
        if self.config.DATASET.NUSCENES.PC_ROI_METHOD == "pillars":
            pillar_wh = self.getPcPillarsSize(
                img, img_info, pc_2d, pc_3d, transMatInput, transMatOutput
            )

        # generate point cloud channels
        for i in range(pc_N - 1, -1, -1):
            point = transformedPoints[:, i]
            depth = point[2]
            center = np.array([point[0], point[1]])
            method = self.config.DATASET.NUSCENES.PC_ROI_METHOD

            if method == "pillars":
                bbox = [
                    max(center[1] - pillar_wh[:, i][1], 0),
                    center[1],
                    max(center[0] - pillar_wh[:, i][0] / 2, 0),
                    min(center[0] + pillar_wh[:, i][0] / 2, outputWidth),
                ]

            elif method == "heatmap":
                radius = (1.0 / depth) * 250 + 5
                radius = getGaussianRadius((radius, radius))
                radius = max(0, int(radius))
                x, y = int(center[0]), int(center[1])
                height, width = depthMap.shape[1:3]
                left, right = min(x, radius), min(width - x, radius + 1)
                top, bottom = min(y, radius), min(height - y, radius + 1)
                bbox = np.array([y - top, y + bottom, x - left, x + right])

            bbox = np.round(bbox).astype(np.int32)
            # Add depth, x velocity, and z velocity to depth map
            depthMap[0, bbox[0] : bbox[1], bbox[2] : bbox[3]] = depth
            depthMap[1, bbox[0] : bbox[1], bbox[2] : bbox[3]] = pc_3d[8, i]
            depthMap[2, bbox[0] : bbox[1], bbox[2] : bbox[3]] = pc_3d[9, i]

        return pc_2d, pc_3d, depthMap

    def transformPointCloud(
        self, pc_2d, transformMat, img_width, img_height, filter_out=True
    ):
        """
        Transform 2D point cloud using transformation matrix

        Args:
            pc_2d: 2D point cloud [x, y, d]
            transformMat: transformation matrix
            img_width: output image width
            img_height: output image height
            filter_out: filter out points outside image

        Returns:
            out: transformed points [x, y, d]
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

    def getPcPillarsSize(
        self, img, img_info, pc_2d, pc_3d, transMatInput, transMatOutput
    ):
        """
        Get the size of the point cloud pillars for every point

        Args:
            img: original image
            img_info: image infomation
            pc_2d: 2D point cloud [x, y, d]
            pc_3d: 3D point cloud [x, y, z]
            transMatInput: transformation matrix from origin image to input size
            transMatOutput: transformation matrix from input to output size

        Returns:
            pillar_wh: width and height of the point cloud pillars [w, h]
        """
        pillar_wh = np.zeros((2, pc_3d.shape[1]))
        boxesInput2D = np.zeros((0, 8, 2))  # for debug plots
        pillar_dims = self.config.DATASET.NUSCENES.PILLAR_DIMS

        for i, center in enumerate(pc_3d[:3, :].T):
            # Create a 3D pillar at pc location for the full-size image
            boxOrigin3D = get3dBox(pillar_dims, center, 0.0)
            boxOrigin2D = project3DPoints(boxOrigin3D, img_info["calib"]).T  # [2x8]

            # save the box for debug plots
            if self.config.DEBUG:
                boxInput2D, _ = self.transformPointCloud(
                    boxOrigin2D,
                    transMatInput,
                    self.config.MODEL.INPUT_SIZE[1],
                    self.config.MODEL.INPUT_SIZE[0],
                    filter_out=False,
                )
                boxesInput2D = np.concatenate(
                    (boxesInput2D, np.expand_dims(boxInput2D.T, 0)), 0
                )

            # transform points
            boxOutput2D, _ = self.transformPointCloud(
                boxOrigin2D,
                transMatOutput,
                self.config.MODEL.OUTPUT_SIZE[1],
                self.config.MODEL.OUTPUT_SIZE[0],
            )

            if boxOutput2D.shape[1] <= 1:
                continue

            # get the bounding box in [xyxy] format
            bbox = [
                np.min(boxOutput2D[0, :]),
                np.min(boxOutput2D[1, :]),
                np.max(boxOutput2D[0, :]),
                np.max(boxOutput2D[1, :]),
            ]

            # store height and width of the 2D box
            pillar_wh[0, i] = bbox[2] - bbox[0]
            pillar_wh[1, i] = bbox[3] - bbox[1]

        ## DEBUG #################################################################
        if self.opt.debug:  # TODO
            img_2d = img.copy()
            img_2d_inp = cv2.warpAffine(
                img,
                transMatInput,
                (self.opt.input_w, self.opt.input_h),
                flags=cv2.INTER_LINEAR,
            )
            img_2d_out = cv2.warpAffine(
                img,
                transMatOutput,
                (self.opt.output_w, self.opt.output_h),
                flags=cv2.INTER_LINEAR,
            )
            img_3d = cv2.warpAffine(
                img,
                transMatInput,
                (self.opt.input_w, self.opt.input_h),
                flags=cv2.INTER_LINEAR,
            )
            blank_image = 255 * np.ones(
                (self.opt.input_h, self.opt.input_w, 3), np.uint8
            )
            mask = np.zeros(img_2d.shape[:2], np.uint8)
            overlay = img_2d_inp.copy()
            origin_overlay = img.copy()

            pc_inp, _ = self.transformPointCloud(
                pc_2d, transMatInput, self.opt.input_w, self.opt.input_h
            )
            pc_inp = pc_inp[:3, :].T
            pc_out, _ = self.transformPointCloud(
                pc_2d, transMatOutput, self.opt.output_w, self.opt.output_h
            )

            pill_wh_inp = pillar_wh * (self.opt.input_w / self.opt.output_w)
            pill_wh_out = pillar_wh
            pill_wh_ori = pill_wh_inp * 2

            # for i, p in reversed(list(enumerate(pc_inp[:3, :].T))):
            for i in range(len(pc_inp) - 1, -1, -1):
                p = pc_inp[i]
                color = int((p[2].tolist() / 60.0) * 255)
                color = (0, color, 0)

                rect_tl = (
                    np.min(int(p[0] - pill_wh_inp[0, i] / 2), 0),
                    np.min(int(p[1] - pill_wh_inp[1, i]), 0),
                )
                rect_br = (np.min(int(p[0] + pill_wh_inp[0, i] / 2), 0), int(p[1]))
                cv2.rectangle(
                    img_2d_inp, rect_tl, rect_br, (0, 0, 255), 1, lineType=cv2.LINE_AA
                )
                img_2d_inp = cv2.circle(
                    img_2d_inp, (int(p[0]), int(p[1])), 3, color, -1
                )

                # On original-sized image
                rect_tl_ori = (
                    np.min(int(pc_2d[0, i] - pill_wh_ori[0, i] / 2), 0),
                    np.min(int(pc_2d[1, i] - pill_wh_ori[1, i]), 0),
                )
                rect_br_ori = (
                    np.min(int(pc_2d[0, i] + pill_wh_ori[0, i] / 2), 0),
                    int(pc_2d[1, i]),
                )
                cv2.rectangle(
                    img_2d,
                    rect_tl_ori,
                    rect_br_ori,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                img_2d = cv2.circle(
                    img_2d, (int(pc_2d[0, i]), int(pc_2d[1, i])), 6, color, -1
                )

                p2 = pc_out[:3, i].T
                rect_tl2 = (
                    np.min(int(p2[0] - pill_wh_out[0, i] / 2), 0),
                    np.min(int(p2[1] - pill_wh_out[1, i]), 0),
                )
                rect_br2 = (np.min(int(p2[0] + pill_wh_out[0, i] / 2), 0), int(p2[1]))
                cv2.rectangle(
                    img_2d_out, rect_tl2, rect_br2, (0, 0, 255), 1, lineType=cv2.LINE_AA
                )
                img_2d_out = cv2.circle(
                    img_2d_out, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1
                )

                # on blank image
                cv2.rectangle(
                    blank_image, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA
                )
                mask[
                    rect_tl_ori[1] : rect_br_ori[1], rect_tl_ori[0] : rect_br_ori[0]
                ] = color[1]

                # plot 3d pillars
                img_3d = draw_box_3d(
                    img_3d,
                    boxesInput2D[i].astype(np.int32),
                    [114, 159, 207],
                    same_color=False,
                )

                # overlay
                cv2.rectangle(
                    overlay, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA
                )
                cv2.rectangle(
                    origin_overlay,
                    rect_tl_ori,
                    rect_br_ori,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

            cv2.imwrite(
                (
                    self.opt.debug_dir + "/{}pc_pillar_2d_inp." + self.opt.img_format
                ).format(self.img_ind),
                img_2d_inp,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir + "/{}pc_pillar_2d_ori." + self.opt.img_format
                ).format(self.img_ind),
                img_2d,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir + "/{}pc_pillar_2d_out." + self.opt.img_format
                ).format(self.img_ind),
                img_2d_out,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir + "/{}pc_pillar_2d_blank." + self.opt.img_format
                ).format(self.img_ind),
                blank_image,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir
                    + "/{}pc_pillar_2d_ori_overlay."
                    + self.opt.img_format
                ).format(self.img_ind),
                origin_overlay,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir
                    + "/{}pc_pillar_2d_overlay."
                    + self.opt.img_format
                ).format(self.img_ind),
                overlay,
            )
            cv2.imwrite(
                (self.opt.debug_dir + "/{}pc_pillar_3d." + self.opt.img_format).format(
                    self.img_ind
                ),
                img_3d,
            )
            cv2.imwrite(
                (
                    self.opt.debug_dir + "/{}pc_pillar_mask." + self.opt.img_format
                ).format(self.img_ind),
                mask,
            )
            cv2.imwrite(
                (self.opt.debug_dir + "/{}img." + self.opt.img_format).format(
                    self.img_ind
                ),
                img,
            )
            self.img_ind += 1
        ## DEBUG #################################################################

        return pillar_wh
