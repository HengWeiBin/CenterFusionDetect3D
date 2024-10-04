# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.
# Modified by Heng Wei Bin, 2024

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any, List

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    load_prediction,
    load_gt,
    add_center_dist,
)
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionConfig,
    DetectionMetrics,
    DetectionBox,
    DetectionMetricDataList,
)
from nuscenes.eval.detection.render import (
    summary_plot,
    class_pr_curve,
    class_tp_curve,
    dist_pr_curve,
    visualize_sample,
)

from algo import accumulate
from loaders import filter_eval_boxes


class DetectionEval:
    """
    This is not the official nuScenes detection evaluation code.
    We add multiple range thresholds for distance metrics. (Heng Wei Bin, 2024)
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(
        self,
        nusc: NuScenes,
        config: DetectionConfig,
        result_path: str,
        eval_set: str,
        output_dir: str = None,
        verbose: bool = True,
        class_range: Dict[str, float] = None,
        specific_scenes: List[str] = None,
    ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        :param class_range: A dictionary that maps class names to distance thresholds.
        :param specific_scenes: A list of scene names to evaluate on.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Set class range.
        if class_range is None:
            class_range = self.cfg.class_range
            min_class_range = {k: 0 for k in self.cfg.class_names}
        else:
            max_range = max(class_range.values())
            min_class_range = {
                k: max(max_range - 20, 0) for k, v in class_range.items()
            }

        # Set scenes keywords
        keywords = None
        if specific_scenes is not None:
            key_dict = {
                "night": ["dark", "very dark", "Night"],
                "rain": ["Rain", "heavy rain"],
            }
            if not len(set(specific_scenes) & set(key_dict.keys())) == len(key_dict):
                raise ValueError("Invalid scene keyword.")
            keywords = [k for key in specific_scenes for k in key_dict[key]]

        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose,
        )
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print("Filtering predictions")
        self.pred_boxes = filter_eval_boxes(
            nusc,
            self.pred_boxes,
            max_dist=class_range,
            min_dist=min_class_range,
            keywords=keywords,
            verbose=verbose,
        )
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = filter_eval_boxes(
            nusc,
            self.gt_boxes,
            max_dist=class_range,
            min_dist=min_class_range,
            keywords=keywords,
            verbose=verbose,
        )

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print("Accumulating metric data...")
        metric_data_list = DetectionMetricDataList()
        mARBufferPath = os.path.join(self.output_dir, "mARBuffer.txt")
        with open(mARBufferPath, "w") as f:
            for class_name in self.cfg.class_names:
                for dist_th in self.cfg.dist_ths:
                    md, addRet = accumulate(
                        self.gt_boxes,
                        self.pred_boxes,
                        class_name,
                        self.cfg.dist_fcn_callable,
                        dist_th,
                    )
                    metric_data_list.set(class_name, dist_th, md)
                    f.write(
                        f"Class: {addRet['Class']}, Dist: {addRet['Dist']}, Recall: {addRet['Recall']}\n"
                    )

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print("Calculating metrics...")
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ["traffic_cone"] and metric_name in [
                    "attr_err",
                    "vel_err",
                    "orient_err",
                ]:
                    tp = np.nan
                elif class_name in ["barrier"] and metric_name in [
                    "attr_err",
                    "vel_err",
                ]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        with open(mARBufferPath, "r") as f:
            recalls = f.readlines()
        os.remove(mARBufferPath)

        # Parse all inputs
        allData = []
        for recall in recalls:
            inpSplit = recall[:-1].replace(" ", "").split(",")
            singleData = dict([kv.split(":") for kv in inpSplit])
            allData.append(singleData)

        # Accumulate all ARs
        ARs = dict([(cls, []) for cls in self.cfg.class_names])
        dist_ARs = dict([(str(dist), []) for dist in self.cfg.dist_ths])
        for data in allData:
            ARs[data["Class"]].append(float(data["Recall"]))
            dist_ARs[data["Dist"]].append(float(data["Recall"]))

        # Calculate and print mAR
        ARs = {cls: np.mean(ARs[cls]) for cls in ARs}
        dist_ARs = {dist: np.mean(dist_ARs[dist]) for dist in dist_ARs}
        mAR = sum(ARs.values())

        outputStr = f"{','.join([str(v) for v in ARs.values()])},"
        outputStr += f"{','.join([str(v) for v in dist_ARs.values()])},"
        outputStr += f"{mAR/len(ARs):.6f}"
        titleStr = f"{','.join(ARs.keys())},{','.join(dist_ARs.keys())},mAR\n"

        mAR_result_path = os.path.join(self.output_dir, "mAR.csv")
        with open(mAR_result_path, "w") as f:
            f.write(titleStr)
            f.write(outputStr)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(
        self, metrics: DetectionMetrics, md_list: DetectionMetricDataList
    ) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print("Rendering PR and TP curves")

        def savepath(name):
            return os.path.join(self.plot_dir, name + ".pdf")

        summary_plot(
            md_list,
            metrics,
            min_precision=self.cfg.min_precision,
            min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp,
            savepath=savepath("summary"),
        )

        for detection_name in self.cfg.class_names:
            class_pr_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath(detection_name + "_pr"),
            )

            class_tp_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_recall,
                self.cfg.dist_th_tp,
                savepath=savepath(detection_name + "_tp"),
            )

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(
                md_list,
                metrics,
                dist_th,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath("dist_pr_" + str(dist_th)),
            )

    def main(
        self, plot_examples: int = 0, render_curves: bool = True
    ) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, "examples")
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(
                    self.nusc,
                    sample_token,
                    self.gt_boxes if self.eval_set != "test" else EvalBoxes(),
                    # Don't render test GT.
                    self.pred_boxes,
                    eval_range=max(self.cfg.class_range.values()),
                    savepath=os.path.join(example_dir, "{}.png".format(sample_token)),
                )

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print("Saving metrics to: %s" % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary["meta"] = self.meta.copy()
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, "metrics_details.json"), "w") as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print("mAP: %.4f" % (metrics_summary["mean_ap"]))
        err_name_mapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            print("%s: %.4f" % (err_name_mapping[tp_name], tp_val))
        print("NDS: %.4f" % (metrics_summary["nd_score"]))
        print("Eval time: %.1fs" % metrics_summary["eval_time"])

        # Print per-class metrics.
        print()
        print("Per-class results:")
        print(
            "%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s"
            % ("Object Class", "AP", "ATE", "ASE", "AOE", "AVE", "AAE")
        )
        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        for class_name in class_aps.keys():
            print(
                "%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f"
                % (
                    class_name,
                    class_aps[class_name],
                    class_tps[class_name]["trans_err"],
                    class_tps[class_name]["scale_err"],
                    class_tps[class_name]["orient_err"],
                    class_tps[class_name]["vel_err"],
                    class_tps[class_name]["attr_err"],
                )
            )

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(
        description="Evaluate nuScenes detection results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("result_path", type=str, help="The submission as a JSON file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/nuscenes-metrics",
        help="Folder to store result metrics, graphs and example visualizations.",
    )
    parser.add_argument(
        "--eval_set",
        type=str,
        default="val",
        help="Which dataset split to evaluate on, train, val or test.",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/data/sets/nuscenes",
        help="Default nuScenes data directory.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to the configuration file."
        "If no path given, the CVPR 2019 configuration will be used.",
    )
    parser.add_argument(
        "--plot_examples",
        type=int,
        default=10,
        help="How many example visualizations to write to disk.",
    )
    parser.add_argument(
        "--render_curves",
        type=int,
        default=1,
        help="Whether to render PR and TP curves to disk.",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Whether to print to stdout."
    )
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == "":
        cfg_ = config_factory("detection_cvpr_2019")
    else:
        with open(config_path, "r") as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    # Generate multiple range thresholds.
    class_ranges = []
    default_ranges = cfg_.class_range
    for range_ in [10, 30, 50]:
        class_ranges_ = {}
        for k, v in default_ranges.items():
            class_ranges_[k] = min(range_, v)
        class_ranges.append(class_ranges_)
    class_ranges.append(None)

    # Create output directory.
    if not os.path.isdir(output_dir_):
        os.makedirs(output_dir_)
    with open(os.path.join(output_dir_, "mAR.csv"), "w") as f:
        f.write(
            f"{','.join(cfg_.class_names)},{','.join([str(v) for v in cfg_.dist_ths])},mAR,range,extreme\n"
        )
    # Run evaluation for each range.
    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    for specific_scenes in [None, ["night", "rain"]]:
        for class_range in class_ranges:
            rangeStr = (
                f"{max(class_range.values())}" if class_range is not None else "all"
            )
            output_dir = os.path.join(
                output_dir_,
                f"range_{rangeStr}{'_extreme' if specific_scenes is not None else ''}",
            )
            plot_examples = plot_examples_ if rangeStr == "all" else 0

            nusc_eval = DetectionEval(
                nusc_,
                config=cfg_,
                result_path=result_path_,
                eval_set=eval_set_,
                output_dir=output_dir,
                verbose=verbose_,
                class_range=class_range,
                specific_scenes=specific_scenes,
            )
            nusc_eval.main(plot_examples=plot_examples, render_curves=render_curves_)

            with open(os.path.join(output_dir, "mAR.csv"), "r") as fRead, open(
                os.path.join(output_dir_, "mAR.csv"), "a"
            ) as fWrite:
                fWrite.write(
                    f"{fRead.readlines()[-1][:-1]},{rangeStr},{specific_scenes is not None}\n"
                )
