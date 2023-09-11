from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import DataParallel
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import psutil

from utils import AverageMeter, saveModel
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
from utils.wandb import WandbLogger

try:
    import wandb
except ImportError:
    wandb = None


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
        self.scaler = GradScaler(enabled=config.MIXED_PRECISION)

    def setDevice(self, config):
        """
        Set device for model and optimizer.

        Args:
            config: yacs config object.

        Returns:
            None.
        """
        if config.GPUS[0] != -1:
            torch.cuda.set_device(config.GPUS[0])
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

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
        wandbLog = {}

        # progress bar
        pbar_title_indent = " " * (16 if phase == "train" else 13)
        print(
            pbar_title_indent
            + "".join([f"{loss[:6]+'.':>9}" for loss in avgLossStats.keys()])
            + f"{'RAM ':>9}"
        )
        pbar = tqdm(dataloader, desc=phase)

        # Start iterating over batches
        for step, batch in enumerate(pbar):
            for k in batch:
                if k != "meta":
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)

            # run one iteration
            with autocast(enabled=self.config.MIXED_PRECISION):
                output, loss, loss_stats = model(batch)
            for k in loss_stats.keys():
                loss_stats[k] = loss_stats[k].mean().item()

            # backpropagate and step optimizer
            if phase == "train":
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss.mean()).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            pbar_msg = f"{phase} epoch {epoch}: "
            for loss in avgLossStats:
                avgLossStats[loss].update(loss_stats[loss], batch["image"].size(0))
                pbar_msg += f"{avgLossStats[loss].avg:9.2f}"
            mem_used = psutil.virtual_memory()[3] / 1e9
            pbar_msg += f"{mem_used:9.2f}"
            pbar.set_description(pbar_msg)

            # generate detections for evaluation
            if phase == "val" and (self.config.TEST.OFFICIAL_EVAL or self.config.EVAL):
                meta = batch["meta"]
                detects = fusionDecode(output, K=self.config.TEST.K)

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

                img_id = meta["img_id"].numpy().astype(np.int32)[0]
                results[img_id] = result

                # Log visualize results to wandb
                WandbLogger.addGroundTruth(
                    dataloader.dataset.coco,
                    dataloader.dataset.img_dir,
                    img_id,
                    batch["pc_hm"][0],
                    config=self.config,
                )
                WandbLogger.addPredict(result, output["pc_hm"][0], calib[0])

            # Log to wandb
            elif phase == "train" and wandb and wandb.run:
                wandbLog = {
                    f"train/{loss}": avgLossStats[loss].avg for loss in avgLossStats
                }
                if step != len(dataloader) - 1:
                    wandb.log(wandbLog, step=(step + 1) + (epoch - 1) * len(dataloader))

        # Log epoch results
        for loss in avgLossStats:
            if phase not in log:
                log[phase] = {}
            if loss not in log[phase]:
                log[phase][loss] = []
            log[phase][loss].append(avgLossStats[loss].avg)

            if phase == "val":
                wandbLog[f"val/{loss}"] = avgLossStats[loss].avg

        if wandb and wandb.run:
            wandb.log(wandbLog, step=epoch * len(dataloader))
        log["memory"].append(mem_used)
        logger.info(pbar_msg)

        return results

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

    @torch.no_grad()
    def val(self, epoch, dataloader, logger, log):
        predicts = self.runEpoch("val", epoch, dataloader, logger, log)
        WandbLogger.syncVisualizeResult()  # Log images to wandb
        return predicts

    def train(self, epoch, dataloader, logger, log):
        return self.runEpoch("train", epoch, dataloader, logger, log)
