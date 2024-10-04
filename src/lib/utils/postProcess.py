from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from .image import affineTransform, getAffineTransform
from utils.ddd import cvtImgToCamCoord, get3dBox
from .pointcloud import get_alpha


def postProcess(y, center, scale, height, width, calibs, isGt=False):
    """
    This function post-processes the output of the network to get the final predictions

    Args:
        y: output of the network
        center: center of the image
        scale: scale of the image
        height: height of the output
        width: width of the output
        calibs: calibration matrices
        isGt: whether the output is ground truth or not

    Returns:
        y: post-processed output
    """
    batch_size, K = y["scores"].shape

    transMat = getAffineTransform(
        center, scale, 0, (width, height), inverse=True
    ).astype(np.float32)

    y["classIds"] += 1
    y["centers"] = y["centers"] * torch.tensor(
        [width, height], device=y["centers"].device
    )
    
    if "bboxes" in y:
        y["bboxes"] = affineTransform(y["bboxes"].view(-1, 2), transMat).reshape(
            batch_size, K, 4
        )

    if "depth" in y:
        y["depth"] = y["depth"].view(batch_size, K)

    if "rotation" in y:
        y["alpha"] = get_alpha(y.pop("rotation").view(-1, 8)).view(batch_size, K)

    # Decode 3D bounding boxes
    if {"alpha", "depth", "dimension"} <= set(y):
        # if rotation, depth and dimension are available
        ## Apply amodal offset to center if available
        if not isGt and "amodal_offset" in y:
            amodalCenter = y["centers"] + y["amodal_offset"]
            centers = affineTransform(amodalCenter.view(-1, 2), transMat).view(
                batch_size, K, 2
            )
            y["centers"] = centers

        ## Use 2D bounding box center if amodal offset is not available
        elif not isGt and "bboxes" in y:
            centers = y["bboxes"].view(batch_size, K, 2, 2).mean(dim=2)
            y["centers"] = centers

        ## Else use the 2D detection center from above

        y["locations"], y["yaws"] = cvtImgToCamCoord(
            y["centers"], y["alpha"], y["dimension"], y["depth"], calibs
        )

    if not isGt and {"velocity", "yaws"} <= set(y):
        # if velocity and yaw are available
        ## put velocity in the same direction as box orientation
        V = torch.sqrt(y["velocity"][:, :, 0] ** 2 + y["velocity"][:, :, 2] ** 2)
        y["velocity"][:, :, 0] = torch.cos(y["yaws"]) * V
        y["velocity"][:, :, 2] = -torch.sin(y["yaws"]) * V

    if {"dimension", "locations", "yaws"} <= set(y):
        # if dimension, location and yaw are available
        y["bboxes3d"] = get3dBox(y["dimension"], y["locations"], y["yaws"])
        y["bboxes3d"][torch.any(y["dimension"] <= 0, dim=2)] = 0

    return y
