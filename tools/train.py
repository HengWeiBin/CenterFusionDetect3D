from __future__ import absolute_import

import _init_paths # initialize paths

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Center Fusion 3D Object Detection network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)

    return args

from config import config