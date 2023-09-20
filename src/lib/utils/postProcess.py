from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .image import affineTransform, getAffineTransform
from .ddd import cvtImgToCamCoord
import math


def get_alpha(rot):
    """
    Get the alpha angle from the rotation vector

    Args:
        rot: rotation vector (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                                      bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]

    Returns:
        alpha: alpha angle (B, 1)
    """
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def postProcess(
    config, detects, centers, scales, height, width, calibs=None, is_gt=False
):
    """
    This function post-processes the output of the network to get the final predictions

    Args:
        config: yacs config object
        detects: output of the network
        centers: centers of the image
        scales: scales of the image
        height: height of the output
        width: width of the output
        calibs: calibration matrices
        is_gt: whether the output is ground truth or not
    """
    if not ("scores" in detects):
        return [{}], [{}]
    ret = []

    for i in range(len(detects["scores"])):
        preds = []
        transMat = getAffineTransform(
            centers[i], scales[i], 0, (width, height), inverse=True
        ).astype(np.float32)
        for j in range(len(detects["scores"][i])):
            if detects["scores"][i][j] < -1:
                break
            item = {}
            item["score"] = detects["scores"][i][j]
            item["class"] = int(detects["classes"][i][j]) + 1
            item["center"] = affineTransform(
                (detects["centers"][i][j]).reshape(1, 2), transMat
            ).reshape(2)

            if "bboxes" in detects:
                bbox = affineTransform(
                    detects["bboxes"][i][j].reshape(2, 2), transMat
                ).reshape(4)
                item["bbox"] = bbox

            if "depth" in detects and len(detects["depth"][i]) > j:
                item["depth"] = detects["depth"][i][j]
                if len(item["depth"]) > 1:
                    item["depth"] = item["depth"][0]

            if "dimension" in detects and len(detects["dimension"][i]) > j:
                item["dimension"] = detects["dimension"][i][j]

            if "rotation" in detects and len(detects["rotation"][i]) > j:
                item["alpha"] = get_alpha(detects["rotation"][i][j : j + 1])[0]

            if (
                "rotation" in detects
                and "depth" in detects
                and "dimension" in detects
                and len(detects["depth"][i]) > j
            ):
                if "amodal_offset" in detects and len(detects["amodal_offset"][i]) > j:
                    centerOut = detects["bboxes"][i][j].reshape(2, 2).mean(axis=0)
                    amodalCenterOut = centerOut + detects["amodal_offset"][i][j]
                    center = (
                        affineTransform(amodalCenterOut.reshape(1, 2), transMat)
                        .reshape(2)
                        .tolist()
                    )
                else:
                    bbox = item["bbox"]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

                item["center"] = center
                item["location"], item["yaw"] = cvtImgToCamCoord(
                    center, item["alpha"], item["dimension"], item["depth"], calibs[i]
                )

            preds.append(item)

        if "nuscenes_att" in detects:
            for j in range(len(preds)):
                preds[j]["nuscenes_att"] = detects["nuscenes_att"][i][j]

        if "velocity" in detects:
            for j in range(len(preds)):
                vel = detects["velocity"][i][j]
                if config.DATASET.NUSCENES.RADAR_PC and not is_gt:
                    ## put velocity in the same direction as box orientation
                    V = math.sqrt(vel[0] ** 2 + vel[2] ** 2)
                    vel[0] = np.cos(preds[j]["yaw"]) * V
                    vel[2] = -np.sin(preds[j]["yaw"]) * V
                preds[j]["velocity"] = vel[:3]

        ret.append(preds)

    return ret
