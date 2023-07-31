from __future__ import absolute_import

import _init_paths # initialize paths
import argparse
from config import config, updateConfig
import numpy as np
import torch
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train Center Fusion 3D Object Detection network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=None,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    updateConfig(config, args)

def main():
    print(config)

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)


if __name__ == '__main__':
    parse_args()
    main()