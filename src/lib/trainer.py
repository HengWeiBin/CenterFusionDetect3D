from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything
from thop import profile, clever_format
import logging
import torch
from torch.utils.data import DataLoader

from model.progressBar import ProgressBar
from utils.logger import initWandb
from model.modelWithLoss import ModelWithLoss
from model.utils import dictToCuda


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        logger,
        log,
        output_dir,
        dataset,
        start_epoch=1,
    ):
        # Define objects used for training
        self.config = config
        self.model = ModelWithLoss(
            model, config, logger, log, output_dir, dataset, start_epoch
        )
        devices = config.GPUS if config.GPUS[0] != -1 else "auto"

        # Debug
        if config.DEBUG > 0:
            self.debugLearningRate(start_epoch)

        # Define callback functions
        callbacks = []
        callbacks.append(ProgressBar(loss_stats=self.model.loss_stats, logger=logger))
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Initialize wandb logger
        wandblogger = initWandb(config, log, output_dir)
        if wandblogger is not None:
            wandblogger.watch(model)

        # Define trainer
        seed_everything(config.RANDOM_SEED)
        self.trainer = L.Trainer(
            deterministic=config.CUDNN.DETERMINISTIC,
            max_epochs=config.TRAIN.EPOCHS + 1,
            check_val_every_n_epoch=config.TRAIN.VAL_INTERVALS,
            precision="16-mixed" if config.MIXED_PRECISION else 32,
            log_every_n_steps=0,
            callbacks=callbacks,
            default_root_dir=output_dir,
            num_sanity_val_steps=0,
            devices=devices,
            logger=wandblogger,
            strategy=DDPStrategy(
                # find_unused_parameters=True, # Set static graph to True will automatically set this to True
                gradient_as_bucket_view=True,
                static_graph=True,  # This may increase the training speed, turn off to save memory
            ),
        )
        self.trainer.fit_loop.epoch_progress.current.processed = start_epoch

    def debugLearningRate(self, start_epoch):
        import matplotlib.pyplot as plt

        EPOCH = self.config.TRAIN.EPOCHS
        opt = self.model.configure_optimizers()
        dummyOptimizer = opt["optimizer"]
        scheduler = opt["lr_scheduler"]["scheduler"]
        lrs = []
        for _ in range(start_epoch, EPOCH + 1):
            dummyOptimizer.step()
            scheduler.step()
            lrs.append(dummyOptimizer.param_groups[0]["lr"])

        print(f"last lr: {dummyOptimizer.param_groups[0]['lr']}")
        plt.plot(range(start_epoch, EPOCH + 1), lrs)
        plt.show()
        breakpoint()

    def train(self, train_dataloader, val_dataloader):
        self.trainer.fit(
            self.model,
            train_dataloader,
            val_dataloader,
        )
        if self.config.TEST.OFFICIAL_EVAL:
            self.val(val_dataloader)

    def val(self, dataloader):
        # Prepare dummy input for profiling
        dummyDataloader = DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False, num_workers=0
        )
        dummyInput = next(iter(dummyDataloader))
        dummyInput = dictToCuda(dummyInput)

        # Inference
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            self.model.eval().cuda()
            macs, params = profile(self.model, inputs=(dummyInput,))

        # Log model complexity
        GFlops = macs / 1e9 * 2
        macs, params = clever_format([macs, params], "%.3f")
        logging.info(f"Model MACs(G): {macs}, GFLOPs: {GFlops:.3f}, params: {params}")

        # Log memory usage
        vram_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        logging.info(f"Max CUDA memory used: {vram_max:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

        self.trainer.validate(self.model, dataloader)

    def test(self, dataloader):
        self.trainer.test(self.model, dataloader)
