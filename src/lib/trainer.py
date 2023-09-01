from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import DataParallel
import numpy as np
from tqdm import tqdm
import psutil

from utils import AverageMeter
from model import fusionDecode
from model.losses import (
    FastFocalLoss,
    RegWeightedL1Loss,
    BinRotLoss,
    WeightedBCELoss,
    DepthLoss,
)
from model.utils import sigmoid
from utils.postProcess import postProcess

# from utils.debugger import Debugger


class GenericLoss(torch.nn.Module):
    def __init__(self, config):
        super(GenericLoss, self).__init__()
        self.focalLoss = FastFocalLoss()
        self.L1Loss = RegWeightedL1Loss()
        if "rotation" in config.heads:
            self.binRotLoss = BinRotLoss()
        if "nuscenes_att" in config.heads:
            self.bceLoss = WeightedBCELoss()
        self.config = config
        self.depthLoss = DepthLoss()

    def _sigmoid_output(self, output):
        if "heatmap" in output:
            output["heatmap"] = sigmoid(output["heatmap"])
        if "depth" in output:
            output["depth"] = 1.0 / (output["depth"].sigmoid() + 1e-6) - 1.0
        if "depth2" in output:
            output["depth2"] = 1.0 / (output["depth2"].sigmoid() + 1e-6) - 1.0
        return output

    def forward(self, outputs, batch):
        losses = {head: 0 for head in self.config.heads}

        output = outputs[0]
        output = self._sigmoid_output(output)

        if "heatmap" in output:
            losses["heatmap"] += self.focalLoss(
                output["heatmap"],
                batch["heatmap"],
                batch["indices"],
                batch["mask"],
                batch["classIds"],
            )

        if "depth" in output:
            losses["depth"] += self.depthLoss(
                output["depth"],
                batch["depth"],
                batch["indices"],
                batch["depth_mask"],
                batch["classIds"],
            )

        regression_heads = [
            "reg",
            "widthHeight",
            "dimension",
            "amodal_offset",
            "velocity",
        ]

        for head in regression_heads:
            if head in output:
                losses[head] += self.L1Loss(
                    output[head],
                    batch[head + "_mask"],
                    batch["indices"],
                    batch[head],
                )

        if "rotation" in output:
            losses["rotation"] += self.binRotLoss(
                output["rotation"],
                batch["rotation_mask"],
                batch["indices"],
                batch["rotbin"],
                batch["rotres"],
            )

        if "nuscenes_att" in output:
            losses["nuscenes_att"] += self.bceLoss(
                output["nuscenes_att"],
                batch["nuscenes_att_mask"],
                batch["indices"],
                batch["nuscenes_att"],
            )

        if "depth2" in output:
            losses["depth2"] += self.depthLoss(
                output["depth2"],
                batch["depth"],
                batch["indices"],
                batch["depth_mask"],
                batch["classIds"],
            )

        if "rotation2" in output:
            losses["rotation2"] += self.binRotLoss(
                output["rotation2"],
                batch["rotation_mask"],
                batch["indices"],
                batch["rotbin"],
                batch["rotres"],
            )

        losses["total"] = 0
        for head in self.config.heads:
            losses["total"] += self.config.weights[head] * losses[head]

        return losses["total"], losses


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, config):
        super(ModelWithLoss, self).__init__()
        self.config = config
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # run the first stage
        outputs = self.model(
            batch["image"],
            pc_hm=batch.get("pc_hm", None),
            pc_dep=batch.get("pc_dep", None),
            calib=batch["calib"].squeeze(0),
        )

        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class Trainer(object):
    def __init__(self, config, model, optimizer=None):
        self.config = config
        self.optimizer = optimizer
        self.loss_stats, self.loss = self.getLossFunction(config)
        self.model = ModelWithLoss(model, self.loss, config)

    def setDevice(self, config):
        """
        Set device for model and optimizer.

        Args:
            config: yacs config object.

        Returns:
            None.
        """
        self.device = torch.device("cuda" if config.GPUS[0] != -1 else "cpu")
        if len(config.GPUS) > 1:
            self.model = DataParallel(self.model, device_ids=config.GPUS).to(
                self.device
            )
        else:
            self.model = self.model.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device, non_blocking=True)

    def runEpoch(self, phase, epoch, dataloader, logger, log):
        """
        Run one epoch of training or evaluation.

        Args:
            phase: "train" or "val".
            epoch: epoch id.
            dataloader: torch dataloader.
            logger: logger object.
        """
        model = self.model
        if phase == "train":
            model.train()
        else:
            if len(self.config.GPUS) > 1:
                model = self.model.module
            model.eval()
            torch.cuda.empty_cache()

        results = {}
        avgLossStats = {
            loss: AverageMeter()
            for loss in self.loss_stats
            if loss == "total" or self.config.weights[loss] > 0
        }

        # progress bar
        pbar_title_indent = " " * (16 if phase == "train" else 13)
        print(
            pbar_title_indent
            + "\t".join([loss.split("_")[-1] for loss in avgLossStats.keys()])
            + "    RAM"
        )
        pbar = tqdm(dataloader, desc=phase)

        # Start iterating over batches
        for iter_id, batch in enumerate(pbar):
            if iter_id >= len(dataloader):
                break
            for k in batch:
                if k != "meta":
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)

            # run one iteration
            output, loss, loss_stats = model(batch)
            for k in loss_stats.keys():
                loss_stats[k] = loss_stats[k].mean().item()

            # backpropagate and step optimizer
            if phase == "train":
                self.optimizer.zero_grad(set_to_none=True)
                loss.mean().backward()
                self.optimizer.step()

            pbar_msg = f"{phase} epoch {epoch}: "
            for loss in avgLossStats:
                avgLossStats[loss].update(loss_stats[loss], batch["image"].size(0))
                pbar_msg += f"{avgLossStats[loss].avg:.2f}" + " " * 4
            mem_used = psutil.virtual_memory()[3] / 1e9
            pbar_msg += f"{mem_used:.2f}    "
            pbar.set_description(pbar_msg)

            # if self.config.DEBUG > 0: # TODO
            #     self.debug(batch, output, iter_id, dataset=dataloader.dataset)

            # generate detections for evaluation
            if phase == "val" and (self.config.TEST.OFFICIAL_EVAL or self.config.EVAL):
                meta = batch["meta"]
                detects = fusionDecode(output, K=self.config.K)

                for k in detects:
                    detects[k] = detects[k].detach().cpu().numpy()

                calib = meta["calib"].detach().numpy() if "calib" in meta else None
                detects = postProcess(
                    self.config,
                    detects,
                    meta["center"].cpu().numpy(),
                    meta["scale"].cpu().numpy(),
                    output["heatmap"].shape[2],
                    output["heatmap"].shape[3],
                    calib,
                )

                # merge results
                result = []
                for i in range(len(detects[0])):
                    if detects[0][i]["score"] > self.config.CONF_THRESH and all(
                        detects[0][i]["dimension"] > 0
                    ):
                        result.append(detects[0][i])

                img_id = batch["meta"]["img_id"].numpy().astype(np.int32)[0]
                results[img_id] = result

        # Log epoch results
        for loss in avgLossStats:
            if loss not in log:
                log[loss] = []
            log[loss].append(avgLossStats[loss].avg)
        log["memory"].append(mem_used)
        logger.info(pbar_msg)

        ret = {k: v.avg for k, v in avgLossStats.items()}
        ret["time"] = pbar.format_dict["elapsed"]
        return ret, results

    def getLossFunction(self, config):
        """
        Create generic loss functions with heads from config.

        Args:
            config: yacs config object.

        Returns:
            loss_states: list of loss names.
            loss: GenericLoss object.
        """
        loss_order = [
            "heatmap",
            "widthHeight",
            "reg",
            "depth",
            "depth2",
            "dimension",
            "rotation",
            "rotation2",
            "amodal_offset",
            "nuscenes_att",
            "velocity",
        ]
        loss_states = ["total"] + [k for k in loss_order if k in config.heads]
        loss = GenericLoss(config)
        return loss_states, loss

    # def debug(self, batch, output, iter_id, dataset):
        opt = self.opt
        if "pre_hm" in batch:
            output.update({"pre_hm": batch["pre_hm"]})
        dets = fusionDecode(output, K=opt.K, opt=opt)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
        dets_gt = batch["meta"]["gt_det"]
        for i in range(1):
            debugger = Debugger(opt=opt, dataset=dataset)
            img = batch["image"][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * dataset.std + dataset.mean) * 255.0), 0, 255).astype(
                np.uint8
            )
            pred = debugger.gen_colormap(output["hm"][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch["hm"][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, "pred_hm", trans=self.opt.hm_transparency)
            debugger.add_blend_img(img, gt, "gt_hm", trans=self.opt.hm_transparency)

            debugger.add_img(img, img_id="img")

            # show point clouds
            if opt.pointcloud:
                pc_2d = batch["pc_2d"][i].detach().cpu().numpy()
                pc_3d = None
                pc_N = batch["pc_N"][i].detach().cpu().numpy()
                debugger.add_img(img, img_id="pc")
                debugger.add_pointcloud(pc_2d, pc_N, img_id="pc")

                if "pc_hm" in opt.pc_feat_lvl:
                    channel = opt.pc_feat_channels["pc_hm"]
                    pc_hm = debugger.gen_colormap(
                        batch["pc_hm"][i][channel].unsqueeze(0).detach().cpu().numpy()
                    )
                    debugger.add_blend_img(
                        img, pc_hm, "pc_hm", trans=self.opt.hm_transparency
                    )
                if "pc_dep" in opt.pc_feat_lvl:
                    channel = opt.pc_feat_channels["pc_dep"]
                    pc_hm = (
                        batch["pc_hm"][i][channel].unsqueeze(0).detach().cpu().numpy()
                    )
                    pc_dep = debugger.add_overlay_img(img, pc_hm, "pc_dep")

            if "pre_img" in batch:
                pre_img = batch["pre_img"][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(
                    ((pre_img * dataset.std + dataset.mean) * 255), 0, 255
                ).astype(np.uint8)
                debugger.add_img(pre_img, "pre_img_pred")
                debugger.add_img(pre_img, "pre_img_gt")
                if "pre_hm" in batch:
                    pre_hm = debugger.gen_colormap(
                        batch["pre_hm"][i].detach().cpu().numpy()
                    )
                    debugger.add_blend_img(
                        pre_img, pre_hm, "pre_hm", trans=self.opt.hm_transparency
                    )

            debugger.add_img(img, img_id="out_pred")

            # Predictions
            for k in range(len(dets["scores"][i])):
                if dets["scores"][i, k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets["bboxes"][i, k] * opt.down_ratio,
                        dets["clses"][i, k],
                        dets["scores"][i, k],
                        img_id="out_pred",
                    )

            # Ground truth
            debugger.add_img(img, img_id="out_gt")
            for k in range(len(dets_gt["scores"][i])):
                if dets_gt["scores"][i][k] > opt.vis_thresh:
                    if "depth" in dets_gt.keys():
                        dist = dets_gt["depth"][i][k]
                        if len(dist) > 1:
                            dist = dist[0]
                    else:
                        dist = -1
                    debugger.add_coco_bbox(
                        dets_gt["bboxes"][i][k] * opt.down_ratio,
                        dets_gt["clses"][i][k],
                        dets_gt["scores"][i][k],
                        img_id="out_gt",
                        dist=dist,
                    )

                    if "ltrb_amodal" in opt.heads:
                        debugger.add_coco_bbox(
                            dets_gt["bboxes_amodal"][i, k] * opt.down_ratio,
                            dets_gt["clses"][i, k],
                            dets_gt["scores"][i, k],
                            img_id="out_gt_amodal",
                        )

                    if "hps" in opt.heads and (int(dets["clses"][i, k]) == 0):
                        debugger.add_coco_hp(
                            dets_gt["hps"][i][k] * opt.down_ratio, img_id="out_gt"
                        )

                    if "tracking" in opt.heads:
                        debugger.add_arrow(
                            dets_gt["cts"][i][k] * opt.down_ratio,
                            dets_gt["tracking"][i][k] * opt.down_ratio,
                            img_id="out_gt",
                        )
                        debugger.add_arrow(
                            dets_gt["cts"][i][k] * opt.down_ratio,
                            dets_gt["tracking"][i][k] * opt.down_ratio,
                            img_id="pre_img_gt",
                        )

            if "hm_hp" in opt.heads:
                pred = debugger.gen_colormap_hp(
                    output["hm_hp"][i].detach().cpu().numpy()
                )
                gt = debugger.gen_colormap_hp(batch["hm_hp"][i].detach().cpu().numpy())
                debugger.add_blend_img(
                    img, pred, "pred_hmhp", trans=self.opt.hm_transparency
                )
                debugger.add_blend_img(
                    img, gt, "gt_hmhp", trans=self.opt.hm_transparency
                )

            if (
                "rotation" in opt.heads
                and "dimension" in opt.heads
                and "depth" in opt.heads
            ):
                dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
                calib = (
                    batch["meta"]["calib"].detach().numpy()
                    if "calib" in batch["meta"]
                    else None
                )
                det_pred = postProcess(
                    opt,
                    dets,
                    batch["meta"]["c"].cpu().numpy(),
                    batch["meta"]["s"].cpu().numpy(),
                    output["hm"].shape[2],
                    output["hm"].shape[3],
                    calib,
                )
                det_gt = postProcess(
                    opt,
                    dets_gt,
                    batch["meta"]["c"].cpu().numpy(),
                    batch["meta"]["s"].cpu().numpy(),
                    output["hm"].shape[2],
                    output["hm"].shape[3],
                    calib,
                    is_gt=True,
                )

                debugger.add_3d_detection(
                    batch["meta"]["img_path"][i],
                    batch["meta"]["flipped"][i],
                    det_pred[i],
                    calib[i],
                    vis_thresh=opt.vis_thresh,
                    img_id="add_pred",
                )
                debugger.add_3d_detection(
                    batch["meta"]["img_path"][i],
                    batch["meta"]["flipped"][i],
                    det_gt[i],
                    calib[i],
                    vis_thresh=opt.vis_thresh,
                    img_id="add_gt",
                )

                pc_3d = None
                if opt.pointcloud:
                    pc_3d = batch["pc_3d"].cpu().numpy()

                debugger.add_bird_views(
                    det_pred[i],
                    det_gt[i],
                    vis_thresh=opt.vis_thresh,
                    img_id="bird_pred_gt",
                    pc_3d=pc_3d,
                    show_velocity=opt.show_velocity,
                )
                debugger.add_bird_views(
                    [],
                    det_gt[i],
                    vis_thresh=opt.vis_thresh,
                    img_id="bird_gt",
                    pc_3d=pc_3d,
                    show_velocity=opt.show_velocity,
                )

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix="{}".format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    @torch.no_grad()
    def val(self, epoch, dataloader, logger, log):
        return self.runEpoch("val", epoch, dataloader, logger, log)

    def train(self, epoch, dataloader, logger, log):
        return self.runEpoch("train", epoch, dataloader, logger, log)
