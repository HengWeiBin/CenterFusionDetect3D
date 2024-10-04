from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths  # initialize paths
import argparse
import os
import cv2
import json
import copy
import numpy as np
import time
from pathlib import Path

from detector import Detector
from dataset import getDataset
from config import config, updateConfig, updateDatasetAndModelConfig
from utils.utils import createFolder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference in Center Fusion 3D Object Detection network"
    )

    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument("--input", help="input file/folder/cam", default=None, type=str)
    parser.add_argument("--save", help="save results", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    updateConfig(config, args)
    return args


def main(args):
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    output_dir = Path("output") / "Demo" / time_str
    createFolder(output_dir, parents=True, exist_ok=True)
    dataset = getDataset(config.DATASET.DATASET)
    updateDatasetAndModelConfig(config, dataset, str(output_dir))
    detector = Detector(config, show=True)
    
    if (
        args.input == "webcam"
        or args.input[args.input.rfind(".") + 1 :].lower() in video_ext
    ):
        cam = cv2.VideoCapture(0 if args.input == "webcam" else args.input)
        inputHeight, inputWidth = config.MODEL.INPUT_SIZE
        out = None
        out_name = args.input[args.input.rfind("/") + 1 :]
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                f"./{out_name}_output.mp4",
                fourcc,
                cam.get(cv2.CAP_PROP_FPS),
                (inputWidth, inputHeight),
            )
        detector.pause = False
        nDetects = 0
        results = {}

        while True:
            _, img = cam.read()
            if img is None:
                break

            nDetects += 1
            ret = detector.run(img)
            time_str = "frame {} |".format(nDetects)
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            results[nDetects] = ret["detects"]
            print(time_str)

            if args.save:
                out.write(ret["result"])

            if cv2.waitKey(1) == 27:
                print("EXIT!")
                save_and_exit(args, out, results, out_name)
                return  # esc to quit
    else:
        # Demo on images, currently does not support tracking
        if os.path.isdir(args.input):
            image_names = []
            ls = os.listdir(args.input)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind(".") + 1 :].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(args.input, file_name))
        else:
            image_names = [args.input]

        for image_name in image_names:
            ret, inferenceTime = detector.run(image_name)
            ret["total"] = inferenceTime
            time_str = ""
            for stat in time_stats:
                time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
            print(time_str)

            if args.save:
                head, tail = image_name.split(".")
                cv2.imwrite(
                    f"{head}_result.{tail}",
                    ret["results"],
                )

            if cv2.waitKey(1) == 27:
                print("EXIT!")
                return  # esc to quit


def save_and_exit(args, out=None, results=None, out_name=""):
    if args.save and (results is not None):
        save_dir = f"./{out_name}_results.json"
        print("saving results to", save_dir)
        json.dump(_to_list(copy.deepcopy(results)), open(save_dir, "w"))

    if args.save and out is not None:
        out.release()

    exit()


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == "__main__":
    """
    inference.py
    This script is used to run inference on a video or image folder.
    Currently, it only supports inference on a video or image with a single camera.

    Usage:
        python src/inference.py --save --cfg configs/centerfusion_debug.yaml --input ./testNuScenes_0.mp4
    """
    image_ext = ["jpg", "jpeg", "png", "webp"]
    video_ext = ["mp4", "mov", "avi", "mkv"]
    time_stats = ["total", "load", "preprocess", "net", "decode", "postprocess", "merge", "display"]
    args = parse_args()
    main(args)
