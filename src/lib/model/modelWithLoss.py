from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
from torch.optim.lr_scheduler import (
    CyclicLR,
    SequentialLR,
    LambdaLR,
    MultiStepLR,
    ConstantLR,
)
from tqdm import tqdm as Tqdm
import lightning as L
import logging

from utils import AverageMeter, saveModel
from utils.logger import WandbLogger
from model.genericLoss import GenericLoss


class ModelWithLoss(L.LightningModule):
    def __init__(self, model, config, logger, ckpt, output_dir, dataset, start_epoch):
        super(ModelWithLoss, self).__init__()
        self.config = config
        self.model = model
        self.ckpt = ckpt
        self.output_dir = output_dir
        self._logger = logger
        self.dataset = dataset  # Dataset object used for evaluation and visualization
        self.start_epoch = start_epoch
        self.defrozen = config.MODEL.DEFREEZE == -1
        self.lr = config.TRAIN.LR
        self.loss_stats, self.loss = self.configure_losses(config)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None and param.requires_grad:
                print(name)

    def forward(self, batch, test=False):
        args = {
            "pc_hm": batch.get("pc_hm", None),
            "pc_dep": batch.get("pc_dep", None),
            "calib": batch["calib"].squeeze(0),
        }
        outputs = self.model(batch["image"], **args)

        if test:
            return outputs

        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats

    def configure_optimizers(self):
        # Configure learning rate
        default_lr = {"adam": 1e-3, "sgd": 1e-2}
        start_lr = self.lr if self.lr else default_lr[self.config.TRAIN.OPTIMIZER]
        for step in self.config.TRAIN.LR_STEP:
            if self.start_epoch >= step:
                start_lr *= 0.1

        # Configure optimizer
        if self.config.TRAIN.OPTIMIZER == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), start_lr, weight_decay=0.0005
            )
        elif self.config.TRAIN.OPTIMIZER == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                start_lr,
                momentum=0.9,
                weight_decay=0.0005,
            )
        else:
            assert 0, self.config.TRAIN.OPTIMIZER

        # Configure learning rate scheduler
        if self.config.TRAIN.LR_SCHEDULER == "CLR":
            # scheduler before defreeze
            scheduler1 = CyclicLR(
                self.optimizer,
                base_lr=self.config.TRAIN.LR / 15,
                max_lr=self.config.TRAIN.LR,
                step_size_up=5,
                cycle_momentum=False,
                mode="triangular",
            )

            # scheduler after defreeze
            scheduler2 = CyclicLR(
                self.optimizer,
                base_lr=self.config.TRAIN.LR / 15,
                max_lr=self.config.TRAIN.LR,
                step_size_up=5,
                cycle_momentum=False,
                mode="triangular2",
            )

            schedulers = [scheduler1, scheduler2]
            milestones = [self.config.MODEL.DEFREEZE]
            skip = 0
            STEP = self.config.TRAIN.LR_STEP
            for i, step in enumerate(STEP):
                if step > self.config.MODEL.DEFREEZE:
                    nextStep = (
                        STEP[i + 1] if i + 1 < len(STEP) else self.config.TRAIN.EPOCHS
                    )
                    milestones.append(step)
                    schedulers.append(
                        ConstantLR(
                            self.optimizer,
                            factor=0.1 ** (i + 1 - skip),
                            last_epoch=-1,
                            total_iters=nextStep - step + 2,
                        )
                    )
                else:
                    skip += 1

            scheduler = SequentialLR(
                self.optimizer,
                schedulers,
                milestones=milestones,
            )

        elif self.config.TRAIN.LR_SCHEDULER == "StepLR":
            # Gradual Warmup in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
            # ref: https://arxiv.org/pdf/1706.02677.pdf
            WARM_EPOCHS = self.config.TRAIN.WARM_EPOCHS
            DEFREEZE = self.config.MODEL.DEFREEZE
            warmupFactor = lambda step: 0.5 ** (WARM_EPOCHS - step)
            schedulers = []
            milestones = []

            # Warmup on start
            if WARM_EPOCHS:
                schedulers.append(
                    LambdaLR(
                        self.optimizer,
                        lr_lambda=warmupFactor,
                    )
                )
                milestones.append(WARM_EPOCHS)

            # StepLR before defreeze
            if DEFREEZE > self.start_epoch:
                milestones2 = [
                    step - self.start_epoch - WARM_EPOCHS
                    for step in self.config.TRAIN.LR_STEP
                    if step < DEFREEZE
                ]
                schedulers.append(
                    MultiStepLR(
                        self.optimizer,
                        milestones=milestones2,
                        gamma=0.1,
                    )
                )
                milestones.append(DEFREEZE - self.start_epoch)

                # Warmup after defreeze
                if WARM_EPOCHS:
                    schedulers.append(
                        LambdaLR(
                            self.optimizer,
                            lr_lambda=warmupFactor,
                        )
                    )
                    milestones.append(DEFREEZE + WARM_EPOCHS - self.start_epoch)

            # StepLR after defreeze
            s4Milestones = [
                step - WARM_EPOCHS - max(DEFREEZE, self.start_epoch)
                for step in self.config.TRAIN.LR_STEP
                if step >= DEFREEZE
            ]
            schedulers.append(
                MultiStepLR(
                    self.optimizer,
                    milestones=s4Milestones,
                    gamma=0.1,
                )
            )

            # Combine all schedulers
            scheduler = SequentialLR(
                self.optimizer,
                schedulers,
                milestones=milestones,
            )

        else:
            assert 0, self.config.TRAIN.LR_SCHEDULER

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def configure_losses(self, config):
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
            "lidar_depth",
            "radar_depth",
            "dimension",
            "rotation",
            "rotation2",
            "amodal_offset",
            "nuscenes_att",
            "velocity",
            "bbox2d",
            "bbox3d",
        ]
        loss_states = ["total"]
        for k in loss_order:
            if (
                k in config.weights
                and config.weights[k] > 0
                and (
                    k in config.heads
                    or k in ["lidar_depth", "radar_depth", "bbox2d", "bbox3d"]
                )
            ) or (k == "depth2" and "depthOffset" in config.heads):
                loss_states.append(k)
        loss = GenericLoss(config, self.dataset.num_categories)
        return loss_states, loss

    def resetAvgLossStats(self, mode: str) -> None:
        avgLossStats = {
            loss: AverageMeter()
            for loss in self.loss_stats
            if loss == "total" or self.config.weights[loss] > 0
        }

        if mode == "train":
            self.trainAvgLossStats = avgLossStats
        elif mode == "val":
            self.valAvgLossStats = avgLossStats

    def init_train_tqdm(self) -> Tqdm:
        return Tqdm(
            disable=self.is_disabled, dynamic_ncols=True, file=sys.stdout, smoothing=0
        )

    def init_validation_tqdm(self) -> Tqdm:
        return Tqdm(disable=self.is_disabled, dynamic_ncols=True, file=sys.stdout)

    def on_train_epoch_start(self):
        self.resetAvgLossStats("train")

        if self.config.MODEL.NORM_EVAL:
            for module in self.model.base.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

        if (
            not self.defrozen
            and self.config.MODEL.FREEZE_BACKBONE
            and self.current_epoch > self.config.MODEL.DEFREEZE
        ):
            logging.info("Defrosting model...")
            for param in self.model.parameters():
                param.requires_grad = True
            self.defrozen = True

            # Calculate number of parameters
            total_params = {"frozen": 0, "trainable": 0}
            for param in self.parameters():
                if param.requires_grad:
                    total_params["trainable"] += param.numel()
                else:
                    total_params["frozen"] += param.numel()
            logging.info(
                f"Number of parameters: {sum(total_params.values()) / 1e6:.2f}M"
            )
            logging.info(
                f"Number of trainable parameters: {total_params['trainable'] / 1e6:.2f}M"
            )

    def training_step(self, batch, batch_idx):
        output, loss, loss_stats = self(batch)

        for k in loss_stats.keys():
            loss_stats[k] = loss_stats[k].mean().item()

        return {"output": output, "loss": loss, "loss_stats": loss_stats}

    def on_train_epoch_end(self):
        # Save model according to save intervals
        if (
            self.config.TRAIN.SAVE_INTERVALS > 0
            and self.current_epoch % self.config.TRAIN.SAVE_INTERVALS == 0
        ):
            saveModel(
                self.ckpt,
                self.model,
                self.current_epoch,
                os.path.join(self.output_dir, f"model_{self.current_epoch}.pt"),
            )

        # Save model every epoch
        saveModel(
            self.ckpt,
            self.model,
            self.current_epoch,
            os.path.join(self.output_dir, "model_last.pt"),
        )
        WandbLogger.commit()

    def on_validation_epoch_start(self):
        self.resetAvgLossStats("val")

        if self.config.EVAL:
            return

        # Save model for prevent validation crash
        saveModel(
            self.ckpt,
            self.model,
            self.current_epoch,
            os.path.join(self.output_dir, "model_last.pt"),
        )

    def validation_step(self, batch, batch_idx):
        outputs, loss, loss_stats = self(batch)

        for output in outputs:
            if self.config.TEST.OFFICIAL_EVAL or self.config.EVAL:
                output["meta"] = batch["meta"]

        for k in loss_stats.keys():
            loss_stats[k] = loss_stats[k].mean().item()

        return {"output": outputs, "loss": loss, "loss_stats": loss_stats}

    def test_step(self, batch, batch_idx):
        outputs = self(batch, test=True)

        for output in outputs:
            output["meta"] = batch["meta"]

        return {"output": outputs}


def learningRateTest():
    import matplotlib.pyplot as plt
    from torch.optim.lr_scheduler import CyclicLR, SequentialLR, ConstantLR

    EPOCH = 230
    START_EPOCH = 1
    learningRate = 1e-3
    DEFREEZE = 100
    STEP = (90, 170, 210)

    dummyModel = torch.nn.Linear(1, 1)
    dummyOptimizer = torch.optim.Adam(dummyModel.parameters(), learningRate)

    scheduler1 = CyclicLR(
        dummyOptimizer,
        base_lr=learningRate * 0.1,
        max_lr=learningRate,
        step_size_up=5,
        cycle_momentum=False,
        mode="triangular",
    )

    scheduler2 = CyclicLR(
        dummyOptimizer,
        base_lr=learningRate * 0.1,
        max_lr=learningRate,
        step_size_up=5,
        cycle_momentum=False,
        mode="triangular2",
    )

    schedulers = [scheduler1, scheduler2]
    milestones = [DEFREEZE]
    skip = 0
    for i, step in enumerate(STEP):
        if step > DEFREEZE:
            nextStep = STEP[i + 1] if i + 1 < len(STEP) else EPOCH
            milestones.append(step)
            schedulers.append(
                ConstantLR(
                    dummyOptimizer,
                    factor=0.5 ** (i + 1 - skip),
                    last_epoch=-1,
                    total_iters=nextStep - step + 2,
                )
            )
        else:
            skip += 1

    print(f"milestones: {milestones}")
    print(f"schedulers: {schedulers}")
    scheduler = SequentialLR(
        dummyOptimizer,
        schedulers,
        milestones=milestones,
    )

    for _ in range(0, START_EPOCH):
        scheduler.step()

    lrs = []
    for _ in range(START_EPOCH, EPOCH + 1):
        scheduler.step()
        lrs.append(dummyOptimizer.param_groups[0]["lr"])

    print(f"last lr: {dummyOptimizer.param_groups[0]['lr']}")
    plt.plot(range(START_EPOCH, EPOCH + 1), lrs)
    plt.show()


if __name__ == "__main__":
    learningRateTest()
