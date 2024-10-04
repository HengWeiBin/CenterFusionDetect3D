import torch
from typing import Any, Optional, Union
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import TQDMProgressBar
import math
from tqdm import tqdm as Tqdm
import psutil
import numpy as np

from utils import getProgressBarMessage, stackDictionary
from utils.postProcess import postProcess
from model import fusionDecode
from utils.logger import WandbLogger


class ProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        self.train_desc = self.val_desc = ""
        self.loss_stats = kwargs.pop("loss_stats")
        self.logger = kwargs.pop("logger")
        self.validation_step_outputs = {}
        self.test_step_outputs = {}
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if trainer.global_rank == 0:
            Tqdm.write("\n")
            self.logger.info(
                " " * 17
                + "".join([f"{loss[:6]+'.':>9}" for loss in self.loss_stats])
                + f"{'RAM ':>9}"
            )

        super().on_train_epoch_start(trainer)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        WandbLogger.checkGPUTemperature()

        self.train_desc = getProgressBarMessage(
            "Train",
            outputs["loss_stats"],
            pl_module.trainAvgLossStats,
            batch["image"].size(0),
            trainer.current_epoch,
        )

        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_description(self.train_desc)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_start(trainer, pl_module)

        if pl_module.config.EVAL and trainer.global_rank == 0:
            Tqdm.write("\n")
            self.logger.info(
                " " * 17
                + "".join([f"{loss[:6]+'.':>9}" for loss in self.loss_stats])
                + f"{'RAM ':>9}"
            )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        WandbLogger.checkGPUTemperature()

        # ================= Save outputs for evaluation =================
        if pl_module.config.TEST.OFFICIAL_EVAL or pl_module.config.EVAL:
            if torch.distributed.is_initialized():
                output = pl_module.all_gather(outputs["output"])
                batch = pl_module.all_gather(batch)

                for i in range(len(output)):
                    output[i] = stackDictionary(output[i])
                batch = stackDictionary(batch)
            else:
                output = outputs["output"]

            # Decode and post process
            detects = fusionDecode(
                output,
                outputSize=pl_module.config.MODEL.OUTPUT_SIZE,
                K=pl_module.config.MODEL.K,
                norm2d=pl_module.config.MODEL.NORM_2D,
            )

            meta = batch["meta"]
            detects = postProcess(
                detects,
                meta["center"].cpu().detach().numpy()[0],
                float(meta["scale"][0]),
                pl_module.config.MODEL.OUTPUT_SIZE[0],
                pl_module.config.MODEL.OUTPUT_SIZE[1],
                batch["calib"],
            )
            for k in detects:
                detects[k] = detects[k].detach().cpu()

            # merge results
            optional_outputs = ["bboxes", "bboxes3d", "nuscenes_att", "velocity"]
            keep = (detects["scores"] > -1) & torch.all(detects["dimension"] > 0, dim=2)
            for i, img_id in enumerate(meta["img_id"].tolist()):
                self.validation_step_outputs[img_id] = []
                for j in range(len(detects["scores"][i])):
                    if not keep[i, j]:
                        continue

                    self.validation_step_outputs[img_id].append(
                        {
                            "class": detects["classIds"][i, j],
                            "score": detects["scores"][i, j],
                            "dimension": detects["dimension"][i, j],
                            "location": detects["locations"][i, j],
                            "yaw": detects["yaws"][i, j],
                            "alpha": detects["alpha"][i, j],
                        }
                    )

                    for option in optional_outputs:
                        if option in detects:
                            self.validation_step_outputs[img_id][-1].update(
                                {option: detects[option][i, j]}
                            )

            # ================= Update WandbLogger =================
            WandbLogger.addGroundTruth(
                pl_module.dataset,
                batch["meta"]["img_id"].cpu().numpy().astype(np.int32)[-1],
                batch["pc_hm"][-1][0] if "pc_hm" in batch else None,
                config=pl_module.config,
            )
            WandbLogger.addPredict(
                self.validation_step_outputs[img_id],
                output["pc_hm"][-1][0] if "pc_hm" in output else None,
                batch["calib"][-1].cpu().numpy(),
            )

        # ================ Update progress bar ================
        self.val_desc = getProgressBarMessage(
            "Val",
            outputs["loss_stats"],
            pl_module.valAvgLossStats,
            batch["image"].size(0),
            trainer.current_epoch,
        )
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)
            self.val_progress_bar.set_description(self.val_desc)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # ================= Save outputs for evaluation =================
        if torch.distributed.is_initialized():
            output = pl_module.all_gather(outputs["output"])
            batch = pl_module.all_gather(batch)

            for i in range(len(output)):
                output[i] = stackDictionary(output[i])
            batch = stackDictionary(batch)
        else:
            output = outputs["output"]

        # Decode and post process
        detects = fusionDecode(
            output,
            outputSize=pl_module.config.MODEL.OUTPUT_SIZE,
            K=pl_module.config.MODEL.K,
            norm2d=pl_module.config.MODEL.NORM_2D,
        )

        meta = batch["meta"]
        detects = postProcess(
            detects,
            meta["center"].cpu().detach().numpy()[0],
            float(meta["scale"][0]),
            pl_module.config.MODEL.OUTPUT_SIZE[0],
            pl_module.config.MODEL.OUTPUT_SIZE[1],
            batch["calib"],
        )
        for k in detects:
            detects[k] = detects[k].detach().cpu()

        # merge results
        optional_outputs = ["bboxes", "bboxes3d", "nuscenes_att", "velocity"]
        keep = (detects["scores"] > -1) & torch.all(detects["dimension"] > 0, dim=2)
        for i, img_id in enumerate(meta["img_id"].tolist()):
            self.test_step_outputs[img_id] = []
            for j in range(len(detects["scores"][i])):
                if not keep[i, j]:
                    continue

                self.test_step_outputs[img_id].append(
                    {
                        "class": detects["classIds"][i, j],
                        "score": detects["scores"][i, j],
                        "dimension": detects["dimension"][i, j],
                        "location": detects["locations"][i, j],
                        "yaw": detects["yaws"][i, j],
                        "alpha": detects["alpha"][i, j],
                    }
                )

                for option in optional_outputs:
                    if option in detects:
                        self.test_step_outputs[img_id][-1].update(
                            {option: detects[option][i, j]}
                        )

        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.global_rank != 0:
            return

        if getattr(self, "train_desc", "") == "":
            # Skip if training is not performed
            return
        self.logger.info(self.train_desc)

        # Save epoch result to checkpoint dictionary and update wandb log
        wandbLog = {}
        for loss in pl_module.trainAvgLossStats:
            if "train" not in pl_module.ckpt:
                pl_module.ckpt["train"] = {}
            if loss not in pl_module.ckpt["train"]:
                pl_module.ckpt["train"][loss] = {}

            pl_module.ckpt["train"][loss][trainer.current_epoch] = (
                pl_module.trainAvgLossStats[loss].avg
            )
            wandbLog[f"train/{loss}"] = pl_module.trainAvgLossStats[loss].avg

        pl_module.ckpt["memory"].append(psutil.virtual_memory()[3] / 1e9)
        WandbLogger.log.update(wandbLog)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_end(trainer, pl_module)

        if trainer.global_rank != 0:
            return

        if getattr(self, "val_desc", "") == "":
            # Skip if validation is not performed
            return
        self.logger.info(self.val_desc)

        # Save epoch result to checkpoint dictionary
        # And update wandb log
        wandbLog = {}
        for loss in pl_module.valAvgLossStats:
            if "val" not in pl_module.ckpt:
                pl_module.ckpt["val"] = {}
            if loss not in pl_module.ckpt["val"]:
                pl_module.ckpt["val"][loss] = {}
            pl_module.ckpt["val"][loss][trainer.current_epoch] = (
                pl_module.valAvgLossStats[loss].avg
            )
            wandbLog[f"val/{loss}"] = pl_module.valAvgLossStats[loss].avg
        WandbLogger.log.update(wandbLog)

        # Run official evaluation
        if pl_module.config.TEST.OFFICIAL_EVAL or pl_module.config.EVAL:
            WandbLogger.renderVisualizeResult()
            pl_module.dataset.run_eval(
                self.validation_step_outputs, pl_module.output_dir
            )
            self.validation_step_outputs.clear()  # free memory
            pl_module.dataset.logValidResult(self.logger, pl_module.output_dir)
            
    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_test_end(trainer, pl_module)

        if trainer.global_rank != 0:
            return

        # Run official evaluation (output detect file)
        pl_module.dataset.run_eval(
            self.test_step_outputs, pl_module.output_dir
        )
        self.test_step_outputs.clear()  # free memory


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def _update_n(bar: Tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()
