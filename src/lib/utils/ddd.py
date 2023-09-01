import numpy as np
import cv2

from .pointcloud import get3DCorners


def get3dBox(dim, location, rotation_y):
    """
    Compute 3D bounding box corners from its parameterization.

    Args:
        dim: tuple of h, w, l
        location: tuple of x, y, z
        rotation_y: rotation around y-axis in camera coordinates [-pi..pi]

    Returns:
        corners_3d: numpy array of shape (8, 3) for 3D box corners
    """
    corners_3d = get3DCorners(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
    return corners_3d


def project3DPoints(points_3D, calib):
    """
    Project 3D points into 2D image plane.

    Args:
        points_3D: numpy array of shape (N, 3) for 3D points
        calib: numpy array of shape (3, 4) for camera calibration matrix

    Return:
        points_2D: numpy array of shape (N, 2) for 2D points
    """
    points_3D_homo = np.concatenate(
        [points_3D, np.ones((points_3D.shape[0], 1), dtype=np.float32)], axis=1
    )
    points_2D = np.dot(calib, points_3D_homo.transpose(1, 0)).transpose(1, 0)
    points_2D = points_2D[:, :2] / points_2D[:, 2:]
    return points_2D


def draw3DBox(image, corners, color=(255, 0, 255), same_color=False):
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
    thickness = 4 if same_color else 2
    corners = corners.astype(np.int32)
    for i in range(3, -1, -1):
        face = faceIndexes[i]
        for j in range(4):
            color2 = color
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
                    1,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    image,
                    (corners[face[1], 0], corners[face[1], 1]),
                    (corners[face[3], 0], corners[face[3], 1]),
                    color,
                    1,
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
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        cx: Camera center x, in pixels
        focalLength: Camera focal length

    Returns:
        rotation_y: Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, focalLength)
    if rot_y > np.pi:
        rot_y -= 2 * np.pi
    if rot_y < -np.pi:
        rot_y += 2 * np.pi
    return rot_y


# unproject_2d_to_3d
def project2DTo3D(pt_2d, depth, calib):
    """
    Project 2D points into 3D space by depth and camera calibration matrix.

    Args:
        pt_2d: array of shape (2,) for 2D points
        depth: depth of the point
        calib: array of shape (3, 4) for camera calibration matrix

    Returns:
        pt_3d: array of shape (3,) for 3D points
    """
    z = depth - calib[2, 3]
    x = (pt_2d[0] * depth - calib[0, 3] - calib[0, 2] * z) / calib[0, 0]
    y = (pt_2d[1] * depth - calib[1, 3] - calib[1, 2] * z) / calib[1, 1]
    pt_3d = np.array([x, y, z], dtype=np.float32).reshape(3)
    return pt_3d


# ddd2locrot
def cvtImgToCamCoord(center, alpha, dim, depth, calib):
    """
    Convert 3D detection from image coordinate to camera coordinate.

    Args:
        center: array of x, y in image coordinate
        alpha: angle around Y-axis in camera coordinates [-pi..pi]
        dim: array of h, w, l
        depth: depth of the object
        calib: array of shape (3, 4) for camera calibration matrix

    Returns:
        locations: array of x, y, z in camera coordinate
        rotation_y: rotation around y-axis in camera coordinates [-pi..pi]
    """
    locations = project2DTo3D(center, depth, calib)
    locations[1] += dim[0] / 2
    rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
    return locations, rotation_y
