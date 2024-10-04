import numpy as np
import cv2
import torch

import utils.pointcloud as PC


def get3dBox(dim, location, yaw):
    """
    Compute 3D bounding box corners from its parameterization.

    Args:
        dim: array of h, w, l (B, K, 3)
        location: array of x, y, z (B, K, 3)
        yaw: rotation around y-axis in camera coordinates [-pi..pi] (B, K)

    Returns:
        corners_3d: 3D box corners (B, K, 8, 3)
    """
    B, K = dim.shape[:2]
    corners_3d = PC.get3DCorners(dim, yaw)
    corners_3d = corners_3d + location.reshape(B, K, 1, 3)
    return corners_3d


def project3DPoints(points_3D, calib):
    """
    Project 3D points into 2D image plane.

    Args:
        points_3D: 3D points (B, K, N, 3)
        calib: camera calibration matrix # (B, K, 3, 4)

    Return:
        points_2D: 2D points (B, K, N, 2)
    """
    lib = torch if isinstance(points_3D, torch.Tensor) else np
    transpose = torch.permute if lib is torch else np.transpose
    contiguous = lambda x: x.contiguous() if lib is torch else x
    cuda = lambda x: x.cuda() if lib is torch else x

    points_3D_homo = lib.concatenate(
        [points_3D, cuda(lib.ones((*points_3D.shape[:-1], 1), dtype=lib.float32))],
        axis=-1,
    )  # (B, K, 8, 3) -> (B, K, 8, 4)
    points_2D = lib.einsum(
        "lkij, lkjm->lkim", calib, contiguous(transpose(points_3D_homo, (0, 1, 3, 2)))
    )  # (B, K, 3, 4) @ (B, K, 4, 8) -> (B, K, 3, 8)
    points_2D = contiguous(
        transpose(points_2D, (0, 1, 3, 2))
    )  # (B, K, 3, 8) -> (B, K, 8, 3)
    points_2D = (
        points_2D[:, :, :, :2] / points_2D[:, :, :, 2:]
    )  # (B, K, 8, 3) -> (B, K, 8, 2)
    return points_2D


def draw3DBox(image, corners, color=(0, 255, 0), same_color=False):
    """
    Draw 3d box on image by 8 corners

    Args:
        image: numpy array of shape (H, W, 3)
        corners: 2D corners numpy array with shape of (8, 2)
        color: color of the box
        same_color: whether to use same color for all edges

    Returns:
        image: numpy array of shape (H, W, 3)
    """
    faceIndexes = [[0, 1, 5, 4], [1, 2, 6, 5], [3, 0, 4, 7], [2, 3, 7, 6]]
    rightCorners = [1, 2, 6, 5] if not same_color else []
    leftCorners = [0, 3, 7, 4] if not same_color else []
    thickness = 1
    corners = corners.astype(np.int32)
    for i in range(3, -1, -1):
        face = faceIndexes[i]
        for j in range(4):
            color2 = color if same_color else (255, 0, 255)
            if (face[j] in leftCorners) and (face[(j + 1) % 4] in leftCorners):
                color2 = (255, 0, 0)
            if (face[j] in rightCorners) and (face[(j + 1) % 4] in rightCorners):
                color2 = (0, 0, 255)
            try:
                cv2.line(
                    image,
                    (corners[face[j], 0], corners[face[j], 1]),
                    (
                        corners[face[(j + 1) % 4], 0],
                        corners[face[(j + 1) % 4], 1],
                    ),
                    color2,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            except:
                pass
        if i == 0:
            try:
                cv2.line(
                    image,
                    (corners[face[0], 0], corners[face[0], 1]),
                    (corners[face[2], 0], corners[face[2], 1]),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    image,
                    (corners[face[1], 0], corners[face[1], 1]),
                    (corners[face[3], 0], corners[face[3], 1]),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            except:
                pass
        # top_idx = [0, 1, 2, 3]
    return image


def alpha2rot_y(alpha, x, cx, focalLength):
    """
    Get rotation_y by alpha + theta - 180

    Args:
        alpha : Observation angle of object, ranging [-pi..pi] (B, K, 1)
        x : Object center x to the image center (x-W/2), in pixels (B, K)
        cx: Image center x, in pixels (B, K)
        focalLength: Camera focal length (B, K)

    Returns:
        rotation_y: Rotation ry around Y-axis in camera coordinates [-pi..pi] (B, K)
    """
    lib = torch if isinstance(alpha, torch.Tensor) else np
    rot_y = alpha + lib.arctan2(x - cx, focalLength)

    rot_y[rot_y > lib.pi] -= 2 * lib.pi
    rot_y[rot_y < -lib.pi] += 2 * lib.pi

    return rot_y


# unproject_2d_to_3d
def project2DTo3D(pt_2d, depth, calib):
    """
    Project 2D points into 3D space by depth and camera calibration matrix.

    Args:
        pt_2d: 2D points (B, K, 2)
        depth: depth of the point (B, K, 1)
        calib: Camera calibration matrix (B, k, 3, 4)

    Returns:
        pt_3d: 3D points (B, K, 3)
    """
    batch_size, K = pt_2d.shape[:2]
    z = depth[:, :, 0] - calib[:, :, 2, 3]  # (B, K)
    x = (
        pt_2d[:, :, 0] * depth[:, :, 0] - calib[:, :, 0, 3] - calib[:, :, 0, 2] * z
    ) / calib[:, :, 0, 0]
    y = (
        pt_2d[:, :, 1] * depth[:, :, 0] - calib[:, :, 1, 3] - calib[:, :, 1, 2] * z
    ) / calib[:, :, 1, 1]
    if isinstance(calib, torch.Tensor):
        pt_3d = torch.stack([x, y, z], dim=-1)
    else:
        pt_3d = np.array([x, y, z], dtype=np.float32).reshape(batch_size, K, 3)
    return pt_3d


# ddd2locrot
def cvtImgToCamCoord(center, alpha, dim, depth, calib):
    """
    Convert 3D detection from image coordinate to camera coordinate.

    Args:
        center: array of x, y in image coordinate (B, K, 2)
        alpha: angle around Y-axis in camera coordinates [-pi..pi] (B, K)
        dim: array of h, w, l (B, K, 3)
        depth: depth of the object (B, K)
        calib: camera calibration matrix (B, 3, 4)

    Returns:
        locations: array of x, y, z in camera coordinate (B, K, 3)
        yaw: rotation around y-axis in camera coordinates [-pi..pi] (B, K)
    """
    if len(calib.shape) == 2:
        calib = calib.unsqueeze(0)
        
    batch_size, K = center.shape[:2]
    depth = depth.view(batch_size, K, 1)  # (B, K) -> (B, K, 1)
    calib = calib.unsqueeze(1)  # (B, 3, 4) -> (B, 1, 3, 4)
    calib = calib.expand(batch_size, K, 3, 4)  # (B, 1, 3, 4) -> (B, K, 3, 4)

    locations = project2DTo3D(center, depth, calib)  # (B, K, 3)
    locations[:, :, 1] += dim[:, :, 0] / 2
    yaw = alpha2rot_y(alpha, center[:, :, 0], calib[:, :, 0, 2], calib[:, :, 0, 0])
    return locations, yaw
