from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce
from typing import Tuple, Dict
import os.path as osp
import torch
import numpy as np
from pyquaternion import Quaternion

from model.utils import topk, transposeAndGetFeature


def map_pointcloud_to_image(pc, cam_intrinsic, img_shape=(1600, 900)):
    """
    Map point cloud from camera coordinates to the image

    :param pc (PointCloud): point cloud in vehicle or global coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param img_shape: shape of the image (width, height)
    :param coordinates (str): Point cloud coordinates ('vehicle', 'global')
    :return points (nparray), depth, mask: Mapped and filtered points with depth and mask
    """

    if isinstance(pc, RadarPointCloud):
        points = pc.points[:3, :]
    else:
        points = pc

    (width, height) = img_shape
    depths = points[2, :]

    ## Take the actual picture
    points = view_points(points[:3, :], cam_intrinsic, normalize=True)

    ## Remove points that are either outside or behind the camera.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < height - 1)
    points = points[:, mask]
    points[2, :] = depths[mask]

    return points, mask


## A RadarPointCloud class where Radar velocity values are correctly
# transformed to the target coordinate system
class RadarPointCloudWithVelocity(RadarPointCloud):
    @classmethod
    def rotate_velocity(cls, pointcloud, transform_matrix):
        n_points = pointcloud.shape[1]
        third_dim = np.zeros(n_points)
        pc_velocity = np.vstack((pointcloud[[8, 9], :], third_dim, np.ones(n_points)))
        pc_velocity = transform_matrix.dot(pc_velocity)

        ## in camera coordinates, x is right, z is front
        pointcloud[[8, 9], :] = pc_velocity[[0, 2], :]

        return pointcloud

    @classmethod
    def from_file_multisweep(
        cls,
        nusc: "NuScenes",
        sample_rec: Dict,
        chan: str,
        ref_chan: str,
        nsweeps: int = 5,
        min_distance: float = 1.0,
    ) -> Tuple["PointCloud", np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )
        ref_from_car_rot = transform_matrix(
            [0.0, 0.0, 0.0], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )
        car_from_global_rot = transform_matrix(
            [0.0, 0.0, 0.0], Quaternion(ref_pose_rec["rotation"]), inverse=True
        )

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec["data"][chan]
        current_sd_rec = nusc.get("sample_data", sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(
                osp.join(nusc.dataroot, current_sd_rec["filename"])
            )
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
            global_from_car = transform_matrix(
                current_pose_rec["translation"],
                Quaternion(current_pose_rec["rotation"]),
                inverse=False,
            )
            global_from_car_rot = transform_matrix(
                [0.0, 0.0, 0.0], Quaternion(current_pose_rec["rotation"]), inverse=False
            )

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get(
                "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
            )
            car_from_current = transform_matrix(
                current_cs_rec["translation"],
                Quaternion(current_cs_rec["rotation"]),
                inverse=False,
            )
            car_from_current_rot = transform_matrix(
                [0.0, 0.0, 0.0], Quaternion(current_cs_rec["rotation"]), inverse=False
            )

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(
                np.dot,
                [ref_from_car, car_from_global, global_from_car, car_from_current],
            )
            velocity_trans_matrix = reduce(
                np.dot,
                [
                    ref_from_car_rot,
                    car_from_global_rot,
                    global_from_car_rot,
                    car_from_current_rot,
                ],
            )
            current_pc.transform(trans_matrix)

            # Do the required rotations to the Radar velocity values
            current_pc.points = cls.rotate_velocity(
                current_pc.points, velocity_trans_matrix
            )

            # Add time vector which can be used as a temporal feature.
            time_lag = (
                ref_time - 1e-6 * current_sd_rec["timestamp"]
            )  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec["prev"] == "":
                break
            else:
                current_sd_rec = nusc.get("sample_data", current_sd_rec["prev"])

        return all_pc, all_times


def get_alpha(rotation):
    """
    Convert rotation to observation angle

    Args:
        rotation: rotation tensor (B, K, 8)
            [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
            bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]

    Returns:
        alpha: observation angle (B, K)
    """
    idx = rotation[..., 1] > rotation[..., 5]
    alpha1 = torch.atan2(rotation[..., 2], rotation[..., 3]) + (-0.5 * torch.pi)
    alpha2 = torch.atan2(rotation[..., 6], rotation[..., 7]) + (0.5 * torch.pi)
    alpha = alpha1 * idx.float() + alpha2 * (~idx).float()
    return alpha


def cvtAlphaToYaw(alpha, objCenterX, imgCenterX, focalLength):
    """
    Get rotation yaw by alpha + theta - 180

    Args:
        alpha : Observation angle of object, ranging [-pi..pi] (B, K)
        objCenterX : Object center x to the camera center (x-W/2), in pixels (B, K)
        imgCenterX: image center x, in pixels (B, K)
        focalLength: Camera focal length x, in pixels (K, B)

    Return:
        yaw : Rotation around Y-axis in camera coordinates [-pi..pi]
    """
    if all(
        isinstance(arg, torch.Tensor) for arg in [objCenterX, imgCenterX, focalLength]
    ):
        yaw = alpha + torch.atan2(objCenterX - imgCenterX, focalLength)
    else:
        yaw = alpha + np.arctan2(objCenterX - imgCenterX, focalLength)

    yaw[yaw > torch.pi] -= 2 * torch.pi
    yaw[yaw < -torch.pi] += 2 * torch.pi
    return yaw


def get3DCorners(dim, yaw):
    """
    Get 3D bounding box corners from its parameterization.

    Args:
        dim: 3D object dimensions [h, w, l] (B, K, 3)
        yaw: Rotation ry around Y-axis in camera coordinates [-pi..pi] (B, K)

    Return:
        corners_3d: 3D bounding box corners (B, K, 8, 3)
    """
    if isinstance(yaw, torch.Tensor):
        lib, unsqueeze, full, zeros, transpose = (
            torch,
            torch.unsqueeze,
            lambda *args, **kwargs: torch.full(*args, **kwargs, device=yaw.device),
            lambda *args, **kwargs: torch.zeros(*args, **kwargs, device=yaw.device),
            torch.permute,
        )
    else:
        lib, unsqueeze, full, zeros, transpose = (
            np,
            np.expand_dims,
            np.full,
            np.zeros,
            np.transpose,
        )

    batch, K = dim.shape[:2]
    c, s = lib.cos(yaw), lib.sin(yaw)  # (B, K)
    R = zeros((batch, K, 3, 3), dtype=lib.float32)  # (B, K, 3, 3)
    R[:, :, 0, 0] = c
    R[:, :, 0, 2] = s
    R[:, :, 1, 1] = 1
    R[:, :, 2, 0] = -s
    R[:, :, 2, 2] = c

    l, w, h = dim[:, :, 2], dim[:, :, 1], dim[:, :, 0]  # (B, K)
    x_corners = full((batch, K, 8), 0.5, dtype=lib.float32)  # (B, K, 8)
    x_corners[:, :, 2:4] *= -1
    x_corners[:, :, 6:8] *= -1
    x_corners *= unsqueeze(l, 2)  # (B, K, 8)

    y_corners = zeros((batch, K, 8), dtype=lib.float32)  # (B, K, 8)
    y_corners[:, :, 4:] = unsqueeze(h, 2) * -1

    z_corners = full((batch, K, 8), 0.5, dtype=lib.float32)  # (B, K, 8)
    z_corners[:, :, 1:3] *= -1
    z_corners[:, :, 5:7] *= -1
    z_corners *= unsqueeze(w, 2)  # (B, K, 8)

    corners = lib.stack([x_corners, y_corners, z_corners], -2)  # (B, K, 3, 8)

    corners_3d = lib.einsum(
        "lkij, lkjm->lkim", R, corners
    )  # (B, K, 3, 3) @ (B, K, 3, 8) = (B, K, 3, 8)
    corners_3d = transpose(corners_3d, (0, 1, 3, 2))  # (B, K, 8, 3)
    return corners_3d


def getDistanceThresh(calib, center, dim, alpha):
    """
    Get the distance threshold for the object.

    Args:
        calib: The calibration matrix. (B, 3, 4)
        center: The center of the object. (B, K, 2)
        dim: The dimensions of the object. (B, K, 3)
        alpha(float): The rotation angle of the object. (B, K, 1)

    Returns:
        The distance threshold for the object. (B, K)
    """
    batch_size, K, _ = center.shape
    # calib (B, 3, 4) -> (B, K, 3, 4)
    if isinstance(calib, torch.Tensor):
        calib = calib.view(-1, 1, 3, 4).expand(batch_size, K, 3, 4)
        lib_min = lambda x, dim: x.min(dim=dim).values
        lib_max = lambda x, dim: x.max(dim=dim).values
    else:
        calib = np.broadcast_to(calib, (batch_size, K, 3, 4))
        lib_min = lambda x, dim: x.min(axis=dim)
        lib_max = lambda x, dim: x.max(axis=dim)

    yaw = cvtAlphaToYaw(alpha, center[..., 0], calib[..., 0, 2], calib[..., 0, 0])
    corners_3d = get3DCorners(dim, yaw)
    dist_thresh = (
        lib_max(corners_3d[..., 2], -1) - lib_min(corners_3d[..., 2], -1) / 2.0
    )
    return dist_thresh


def getPcFrustumHeatmap(output, pc_dep, calib, config):
    """
    Generate point cloud heatmap from the frustum association

    Args:
        output: model output
        pc_dep: point cloud depth feature map [depth, vel_x, vel_z]
        calib: calibration matrix
        config: config / options
    """
    K = config.MODEL.K
    heatmap = output["heatmap"]
    widthHeight = output["widthHeight"]
    pc_hm = torch.zeros_like(pc_dep)

    batch, nClass, _, _ = heatmap.size()
    _, indices, classes, ys, xs = topk(heatmap, K=K)
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    calib = calib.view(batch, 3, 4)

    ## get estimated depths
    depth = transposeAndGetFeature(output["depth"], indices)  # B x K x (C)

    # get topk bounding boxes
    widthHeight = transposeAndGetFeature(widthHeight, indices)  # B x K x 2
    widthHeight = widthHeight.view(batch, K, 2)
    widthHeight[widthHeight < 0] = 0
    bboxes = torch.cat(
        [
            xs - widthHeight[..., 0:1] / 2,
            ys - widthHeight[..., 1:2] / 2,
            xs + widthHeight[..., 0:1] / 2,
            ys + widthHeight[..., 1:2] / 2,
        ],
        dim=2,
    )  # B x K x 4

    # get dimensions and rotation
    dimension = transposeAndGetFeature(output["dimension"], indices).view(batch, K, -1)
    rotation = transposeAndGetFeature(output["rotation"], indices).view(batch, K, -1)
    alpha = get_alpha(rotation)  # B x K
    center = torch.stack(
        [
            (bboxes[..., 0] + bboxes[..., 2]) / 2,
            (bboxes[..., 1] + bboxes[..., 3]) / 2,
        ],
        dim=2,
    )  # B x K x 2
    distanceThreshold = getDistanceThresh(calib, center, dimension, alpha)  # B x K

    # Draw heatmap
    for batch in range(len(bboxes)):
        for i in range(len(bboxes[batch])):
            cvtPcDepthToHeatmap(
                pc_hm[batch],
                pc_dep[batch],
                depth[batch, i],
                bboxes[batch, i],
                distanceThreshold[batch, i],
                config.DATASET.MAX_PC_DIST,
            )

    return pc_hm


def cvtPcDepthToHeatmap(pc_hm, pc_dep, depth, bbox, distanceThreshold, max_pc_dist):
    """
    Frustum Association: Filter point cloud depth feature map and convert to heatmap

    Args:
        pc_hm: draw a single-depth-value rectangle that is smaller than the bounding box (output)
        pc_dep: point cloud depth feature map [depth, vel_x, vel_z]
        depth: depth annotation
        bbox: bounding box [x1, y1, x2, y2]
        distanceThreshold: distance threshold
        config: config / options

    Returns:
        None
    """
    if isinstance(pc_hm, torch.Tensor):
        lib, toArray, nonzero = (
            torch,
            torch.tensor,
            lambda x: torch.nonzero(x, as_tuple=True),
        )
    else:
        lib, toArray, nonzero = (np, np.array, np.nonzero)

    if isinstance(depth, list) and len(depth) > 0:
        depth = depth[0]

    center = toArray(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=lib.float32
    )
    bbox_int = toArray(
        [
            int(lib.floor(bbox[0])),
            int(lib.floor(bbox[1])),
            int(lib.ceil(bbox[2])),
            int(lib.ceil(bbox[3])),
        ],
        dtype=lib.int32,
    )

    roi = pc_dep[:, bbox_int[1] : bbox_int[3] + 1, bbox_int[0] : bbox_int[2] + 1]
    enableVel = len(roi) == 3
    if enableVel:
        pc_dep, vel_x, vel_z = roi[0], roi[1], roi[2]
    else:
        pc_dep = roi[0]

    nonZeroIndex = nonzero(pc_dep)
    if len(nonZeroIndex[0]) > 0:
        nonZero_pc_dep = pc_dep[nonZeroIndex]
        if enableVel:
            nonZero_vel_x = vel_x[nonZeroIndex]
            nonZero_vel_z = vel_z[nonZeroIndex]

        # Get points within distance threshold
        within_thresh = (nonZero_pc_dep < depth + distanceThreshold) & (
            nonZero_pc_dep > max(0, depth - distanceThreshold)
        )
        pc_dep_match = nonZero_pc_dep[within_thresh]
        if enableVel:
            vel_x_match = nonZero_vel_x[within_thresh]
            vel_z_match = nonZero_vel_z[within_thresh]

        if len(pc_dep_match) > 0:
            arg_min = lib.argmin(pc_dep_match)
            dist = pc_dep_match[arg_min]
            if enableVel:
                vx = vel_x_match[arg_min]
                vz = vel_z_match[arg_min]
            dist /= max_pc_dist  # normalize depth

            w = bbox[2] - bbox[0]
            w_interval = 0.3 * w  # Heatmap to box ratio is 0.3
            w_min = int(center[0] - w_interval / 2.0)
            w_max = int(center[0] + w_interval / 2.0)

            h = bbox[3] - bbox[1]
            h_interval = 0.3 * h  # Heatmap to box ratio is 0.3
            h_min = int(center[1] - h_interval / 2.0)
            h_max = int(center[1] + h_interval / 2.0)

            pc_hm[0, h_min : h_max + 1, w_min : w_max + 1 + 1] = dist
            if enableVel:
                pc_hm[1, h_min : h_max + 1, w_min : w_max + 1 + 1] = vx
                pc_hm[2, h_min : h_max + 1, w_min : w_max + 1 + 1] = vz
