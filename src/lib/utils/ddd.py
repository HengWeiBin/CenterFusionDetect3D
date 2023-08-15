import numpy as np
from .pointcloud import comput_corners_3d

def get3dBox(dim, location, rotation_y):
  '''
  Compute 3D bounding box corners from its parameterization.

  Args:
      dim: tuple of h, w, l
      location: tuple of x, y, z
      rotation_y: rotation around y-axis in camera coordinates [-pi..pi]

  Returns:
      corners_3d: numpy array of shape (8, 3) for 3D box corners
  '''
  corners_3d = comput_corners_3d(dim, rotation_y)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
  return corners_3d

def project3DPoints(points_3D, calib):
  '''
  Project 3D points into 2D image plane.

  Args:
      points_3D: numpy array of shape (N, 3) for 3D points
      calib: numpy array of shape (3, 4) for camera calibration matrix
      
  Return:
      points_2D: numpy array of shape (N, 2) for 2D points
  '''
  points_3D_homo = np.concatenate(
    [points_3D, np.ones((points_3D.shape[0], 1), dtype=np.float32)], axis=1)
  points_2D = np.dot(calib, points_3D_homo.transpose(1, 0)).transpose(1, 0)
  points_2D = points_2D[:, :2] / points_2D[:, 2:]
  return points_2D