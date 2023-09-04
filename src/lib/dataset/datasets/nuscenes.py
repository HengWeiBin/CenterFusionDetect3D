# Copyright (c) Xingyi Zhou. All Rights Reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pyquaternion import Quaternion
import numpy as np
import torch
import json
import os
from tqdm import tqdm
import cv2

from ..generic_dataset import GenericDataset
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes
from itertools import compress

class nuScenes(GenericDataset):
    default_resolution = [900, 1600]
    num_categories = 10
    focal_length = 1200
    max_objs = 128

    class_name = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
    ]
    class_ids = {
        i + 1: i + 1 for i in range(num_categories)
    }

    vehicles = ["car", "truck", "bus", "trailer", "construction_vehicle"]
    cycles = ["motorcycle", "bicycle"]
    pedestrians = ["pedestrian"]

    attribute_to_id = {
        "": 0,
        "cycle.with_rider": 1,
        "cycle.without_rider": 2,
        "pedestrian.moving": 3,
        "pedestrian.standing": 4,
        "pedestrian.sitting_lying_down": 5,
        "vehicle.moving": 6,
        "vehicle.parked": 7,
        "vehicle.stopped": 8,
    }
    id_to_attribute = {v: k for k, v in attribute_to_id.items()}

    SENSOR_NAME = {
        1: "CAM_FRONT",
        2: "CAM_FRONT_RIGHT",
        3: "CAM_BACK_RIGHT",
        4: "CAM_BACK",
        5: "CAM_BACK_LEFT",
        6: "CAM_FRONT_LEFT",
        7: "RADAR_FRONT",
        8: "LIDAR_TOP",
        9: "RADAR_FRONT_LEFT",
        10: "RADAR_FRONT_RIGHT",
        11: "RADAR_BACK_LEFT",
        12: "RADAR_BACK_RIGHT",
    }
    RADARS_FOR_CAMERA = {
        "CAM_FRONT_LEFT": ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
        "CAM_FRONT_RIGHT": ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
        "CAM_FRONT": ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
        "CAM_BACK_LEFT": ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
        "CAM_BACK_RIGHT": ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
        "CAM_BACK": ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"],
    }
    SPLITS = {
        "mini_val": "v1.0-mini",
        "mini_train": "v1.0-mini",
        "train": "v1.0-trainval",
        "val": "v1.0-trainval",
        "test": "v1.0-test",
    }

    nuscenes_att_range = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3, 4],
        3: [2, 3, 4],
        4: [2, 3, 4],
        5: [5, 6, 7],
        6: [5, 6, 7],
        7: [5, 6, 7],
    }

    def __init__(self, config, split, device):
        data_dir = os.path.join(config.DATASET.ROOT, "nuscenes")

        ann_path = os.path.join(data_dir, "annotations", "{}.json").format(split)
        print(f"self.SPLITS[split]: {self.SPLITS[split]}")
        self.nusc = NuScenes(
            version=self.SPLITS[split], dataroot=data_dir, verbose=False
        )

        super(nuScenes, self).__init__(config, split, ann_path, data_dir, device)

        print("Loaded {} {} samples".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def _to_float(self, x):
        """
        Convert to float and accurate to 2 decimal places
        """
        return float("{:.2f}".format(x))

    def convert_coco_format(self, all_bboxes):
        """
        Convert 2D bbox to coco format
        """
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != dict:
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    category_id = item["class"]
                    bbox = item["bbox"]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(item["score"])),
                    }
                    detections.append(detection)
        return detections

    def convert_eval_format(self, results):
        ret = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": self.config.DATASET.NUSCENES.RADAR_PC,
                "use_map": False,
                "use_external": False,
            },
            "results": {},
        }
        print("Converting nuscenes format...")

        for image_id in self.images:
            if not (image_id in results):
                continue
            image_info = self.coco.loadImgs(ids=[image_id])[0]
            sample_token = image_info["sample_token"]
            trans_matrix = np.array(image_info["trans_matrix"], np.float32)
            velocity_mat = np.array(image_info["velocity_trans_matrix"], np.float32)
            sensor_id = image_info["sensor_id"]
            sample_results = []

            for item in results[image_id]:
                class_name = (
                    self.class_name[int(item["class"] - 1)]
                    if not ("detection_name" in item)
                    else item["detection_name"]
                )

                score = (
                    float(item["score"])
                    if not ("detection_score" in item)
                    else item["detection_score"]
                )

                if "size" in item:
                    size = item["size"]
                else:
                    size = [
                        float(item["dimension"][1]),
                        float(item["dimension"][2]),
                        float(item["dimension"][0]),
                    ]

                if "translation" in item:
                    translation = item["translation"]
                else:
                    translation = np.dot(
                        trans_matrix,
                        np.array(
                            [
                                item["loc"][0],
                                item["loc"][1] - size[2],
                                item["loc"][2],
                                1,
                            ],
                            np.float32,
                        ),
                    )

                det_id = item["det_id"] if "det_id" in item else -1
                tracking_id = item["tracking_id"] if "tracking_id" in item else 1

                if not ("rotation" in item):
                    rot_cam = Quaternion(axis=[0, 1, 0], angle=item["rot_y"])
                    loc = np.array(
                        [item["loc"][0], item["loc"][1], item["loc"][2]], np.float32
                    )
                    box = Box(loc, size, rot_cam, name="2", token="1")
                    box.translate(np.array([0, -box.wlh[2] / 2, 0]))
                    box.rotate(Quaternion(image_info["cs_record_rot"]))
                    box.translate(np.array(image_info["cs_record_trans"]))
                    box.rotate(Quaternion(image_info["pose_record_rot"]))
                    box.translate(np.array(image_info["pose_record_trans"]))
                    rotation = box.orientation
                    rotation = [
                        float(rotation.w),
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                    ]
                else:
                    rotation = item["rotation"]

                nuscenes_att = (
                    np.array(item["nuscenes_att"], np.float32)
                    if "nuscenes_att" in item
                    else np.zeros(8, np.float32)
                )
                att = ""
                if class_name in self.cycles:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[0:2]) + 1]
                elif class_name in self.pedestrians:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[2:5]) + 3]
                elif class_name in self.vehicles:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[5:8]) + 6]

                if "velocity" in item and len(item["velocity"]) == 2:
                    velocity = item["velocity"]
                else:
                    velocity = item["velocity"] if "velocity" in item else [0, 0, 0]
                    velocity = np.dot(
                        velocity_mat,
                        np.array(
                            [velocity[0], velocity[1], velocity[2], 0], np.float32
                        ),
                    )
                    velocity = [float(velocity[0]), float(velocity[1])]

                result = {
                    "sample_token": sample_token,
                    "translation": [
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2]),
                    ],
                    "size": size,
                    "rotation": rotation,
                    "velocity": velocity,
                    "detection_name": class_name,
                    "attribute_name": att
                    if not ("attribute_name" in item)
                    else item["attribute_name"],
                    "detection_score": score,
                    "tracking_name": class_name,
                    "tracking_score": score,
                    "tracking_id": tracking_id,
                    "sensor_id": sensor_id,
                    "det_id": det_id,
                }

                sample_results.append(result)

            if sample_token in ret["results"]:
                ret["results"][sample_token] = (
                    ret["results"][sample_token] + sample_results
                )
            else:
                ret["results"][sample_token] = sample_results

        for sample_token in ret["results"].keys():
            confs = sorted(
                [
                    (-d["detection_score"], ind)
                    for ind, d in enumerate(ret["results"][sample_token])
                ]
            )
            ret["results"][sample_token] = [
                ret["results"][sample_token][ind]
                for _, ind in confs[: min(500, len(confs))]
            ]

        return ret

    def run_eval(self, results, save_dir, n_plots=10):
        split = self.config.DATASET.VAL_SPLIT
        version = "v1.0-mini" if "mini" in split else "v1.0-trainval"
        json.dump(
            self.convert_eval_format(results),
            open(f"{save_dir}/results_nuscenes_det_{split}.json", "w"),
        )

        output_dir = f"{save_dir}/nuscenes_eval_det_output_{split}/"
        os.system(
            "python "
            + "src/nuscenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py "
            + f"{save_dir}/results_nuscenes_det_{split}.json "
            + f"--output_dir {output_dir} "
            + f"--eval_set {split} "
            + "--dataroot data/nuscenes/ "
            + f"--version {version} "
            + f"--plot_examples {n_plots} "
            + "--render_curves 1 "
        )

        return output_dir