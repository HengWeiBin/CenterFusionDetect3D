from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .datasets.kitti import KITTI
from .datasets.nuscenes import nuScenes

dataset_factory = {
  # 'kitti': KITTI,
  'nuscenes': nuScenes,
}

def getDataset(dataset):
  return dataset_factory[dataset]