from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud
from functools import reduce
from typing import Tuple, Dict
import os.path as osp
import torch
import timeit
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


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = torch.atan2(rot[:, 2], rot[:, 3]) + (-0.5 * 3.14159)
    alpha2 = torch.atan2(rot[:, 6], rot[:, 7]) + (0.5 * 3.14159)
    # return alpha1 * idx + alpha2 * (~idx)
    alpha = alpha1 * idx.float() + alpha2 * (~idx).float()
    return alpha


def cvtAlphaToRotateY(alpha, objCenterX, imgCenterX, focalLength):
    """
    Get rotation_y by alpha + theta - 180

    Args:
        alpha : Observation angle of object, ranging [-pi..pi]
        objCenterX : Object center x to the camera center (x-W/2), in pixels
        imgCenterX: image center x, in pixels
        focalLength: Camera focal length x, in pixels

    Return:
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    if all(
        isinstance(arg, torch.Tensor) for arg in [objCenterX, imgCenterX, focalLength]
    ):
        rot_y = alpha + torch.atan2(objCenterX - imgCenterX, focalLength)
    else:
        rot_y = alpha + np.arctan2(objCenterX - imgCenterX, focalLength)

    if rot_y > 3.14159:
        rot_y -= 2 * 3.14159
    if rot_y < -3.14159:
        rot_y += 2 * 3.14159
    return rot_y


def get3DCorners(dim, rotation_y):
    """
    Get 3D bounding box corners from its parameterization.

    Args:
        dim: 3D object dimensions [l, w, h]
        rotation_y: Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Return:
        corners_3d: 3D bounding box corners [8, 3]
    """
    if isinstance(rotation_y, torch.Tensor):
        lib, matmul, toArray = torch, torch.mm, torch.tensor
    else:
        lib, matmul, toArray = np, np.dot, np.array

    c, s = lib.cos(rotation_y), lib.sin(rotation_y)
    R = toArray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=lib.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = toArray([x_corners, y_corners, z_corners], dtype=lib.float32)
    corners_3d = matmul(R, corners).transpose(1, 0)
    return corners_3d


def getDistanceThresh(calib, center, dim, alpha):
    """
    Get the distance threshold for the object.

    Args:
        calib: The calibration matrix.
        center: The center of the object.
        dim: The dimensions of the object.
        alpha: The rotation angle of the object.

    Returns:
        The distance threshold for the object.
    """
    rotation_y = cvtAlphaToRotateY(alpha, center[0], calib[0, 2], calib[0, 0])
    corners_3d = get3DCorners(dim, rotation_y)
    dist_thresh = max(corners_3d[:, 2]) - min(corners_3d[:, 2]) / 2.0
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
    K = config.TEST.K
    heatmap = output["heatmap"]
    widthHeight = output["widthHeight"]
    pc_hm = torch.zeros_like(pc_dep)

    batch, nClass, _, _ = heatmap.size()
    _, indices, classes, ys, xs = topk(heatmap, K=K)
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5

    ## get estimated depths
    out_dep = 1.0 / (output["depth"].sigmoid() + 1e-6) - 1.0
    dep = transposeAndGetFeature(out_dep, indices)  # B x K x (C)
    if dep.size(2) == nClass:
        classes_ = classes.view(batch, K, 1, 1)
        dep = dep.view(batch, K, -1, 1)  # B x K x C x 1
        dep = dep.gather(2, classes_.long()).squeeze(2)  # B x K x 1

    # get topk bounding boxes
    widthHeight = transposeAndGetFeature(widthHeight, indices)  # B x K x 2
    widthHeight = widthHeight.view(batch, K, 2)
    widthHeight[widthHeight < 0] = 0
    if widthHeight.size(2) == 2 * nClass:
        widthHeight = widthHeight.view(batch, K, -1, 2)
        classes_ = classes.view(batch, K, 1, 1).expand(batch, K, 1, 2)
        widthHeight = widthHeight.gather(2, classes_.long()).squeeze(2)  # B x K x 2
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
    dims = transposeAndGetFeature(output["dim"], indices).view(batch, K, -1)
    rot = transposeAndGetFeature(output["rot"], indices).view(batch, K, -1)

    # Draw heatmap
    for i, [pc_dep_b, bboxes_b, depth_b, dim_b, rot_b] in enumerate(
        zip(pc_dep, bboxes, dep, dims, rot)
    ):
        alpha_b = get_alpha(rot_b).unsqueeze(1)

        for bbox, depth, dim, alpha in zip(bboxes_b, depth_b, dim_b, alpha_b):
            center = torch.tensor(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                device=pc_dep_b.device,
            )
            distanceThreshold = getDistanceThresh(calib, center, dim, alpha)
            cvtPcDepthToHeatmap(
                pc_hm[i],
                pc_dep_b,
                depth,
                bbox,
                distanceThreshold,
                config.DATASET.NUSCENES.MAX_PC_DIST,
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
        [lib.floor(bbox[0]), lib.floor(bbox[1]), lib.ceil(bbox[2]), lib.ceil(bbox[3])],
        lib.int32,
    )

    roi = pc_dep[:, bbox_int[1] : bbox_int[3] + 1, bbox_int[0] : bbox_int[2] + 1]
    pc_dep, vel_x, vel_z = roi[0], roi[1], roi[2]

    nonZeroIndex = nonzero(pc_dep)
    if len(nonZeroIndex[0]) > 0:
        nonZero_pc_dep = pc_dep[nonZeroIndex]
        nonZero_vel_x = vel_x[nonZeroIndex]
        nonZero_vel_z = vel_z[nonZeroIndex]

        # Get points within distance threshold
        within_thresh = (nonZero_pc_dep < depth + distanceThreshold) & (
            nonZero_pc_dep > max(0, depth - distanceThreshold)
        )
        pc_dep_match = nonZero_pc_dep[within_thresh]
        vel_x_match = nonZero_vel_x[within_thresh]
        vel_z_match = nonZero_vel_z[within_thresh]

        if len(pc_dep_match) > 0:
            arg_min = lib.argmin(pc_dep_match)
            dist = pc_dep_match[arg_min]
            vx = vel_x_match[arg_min]
            vz = vel_z_match[arg_min]
            dist /= max_pc_dist  # normalize depth

            w = bbox[2] - bbox[0]
            w_interval = 0.3 * w
            w_min = int(center[0] - w_interval / 2.0)
            w_max = int(center[0] + w_interval / 2.0)

            h = bbox[3] - bbox[1]
            h_interval = 0.3 * h
            h_min = int(center[1] - h_interval / 2.0)
            h_max = int(center[1] + h_interval / 2.0)

            pc_hm[0, h_min : h_max + 1, w_min : w_max + 1 + 1] = dist
            pc_hm[1, h_min : h_max + 1, w_min : w_max + 1 + 1] = vx
            pc_hm[2, h_min : h_max + 1, w_min : w_max + 1 + 1] = vz
