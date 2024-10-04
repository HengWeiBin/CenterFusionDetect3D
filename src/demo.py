from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths  # initialize paths
import argparse
import warnings

from dataset import getDataset
from dataset.datasets.nuscenes import nuScenes
from config import config, updateConfig
from utils.utils import createLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo in Radar-Camera Fusion 3D Object Detection network with nuScenes sample"
    )

    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument(
        "--split",
        help="nuScenes dataset split: [train, val, mini_train, mini_val]",
        default="mini_val",
        type=str,
    )
    parser.add_argument("--save", help="save results", action="store_true")
    parser.add_argument(
        "--min", help="ignore number of scenes from front", default=0, type=int
    )
    parser.add_argument("--max", help="max number of scenes demo", default=1, type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--sample",
        help="Enter nuScenes sample token, or sample id, predict only on specific sample.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--single",
        help="Run demo on single camera view only.",
        action="store_true",
    )
    parser.add_argument(
        "--not-show",
        help="Do not show the result.",
        action="store_true",
    )
    parser.add_argument(
        "--show-attention",
        help="Show attention map.",
        action="store_true",
    )

    args = parser.parse_args()
    updateConfig(config, args)
    return args


def main(args):
    if args.sample is not None and args.max > 0:
        warnings.warn("Scene demo will be ignored when sample is specified.")

    config.defrost()
    config.NAME = "Demo"
    config.freeze()

    _, output_dir = createLogger(config)
    dataset = getDataset(config.DATASET.DATASET)
    demo = dataset.getDemoInstance(args)
    demo.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
