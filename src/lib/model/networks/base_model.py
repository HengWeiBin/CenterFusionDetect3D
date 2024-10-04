# ------------------------------------------------------------------------------
# Portions of this code are from
# CenterFusion (https://github.com/mrnabati/CenterFusion)
# Copyright (c) 2020 Ramin Nabati
# Licensed under the MIT License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, fx

import tune_mode_convbn
from .fusionModules import ConcateCombiner
from .detectHeads import (
    DetectHead,
    CenterFusionHead,
)

_head_factory = {
    # Camera-Radar Fusion Heads
    "early": DetectHead,
    "middle": CenterFusionHead,
    None: DetectHead,
}
EARLY_FUSION = ["early"]


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        Args:
            in_channels_head(int or list of int): input channels of the head [[img_channels, pc_channels], anothor_layers...]
            config: yacs config
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.heads = config.heads

        if isinstance(self.in_channels_head, int):
            self.in_channels_head = [[self.in_channels_head]]

        detectHead = _head_factory[self.fusionStrategy]
        for i, in_channels in enumerate(self.in_channels_head):
            self.__setattr__(
                f"detectHead{'_'+str(i)}",
                detectHead(in_channels, config),
            )

        # Define fusion modules
        if self.isRadarEnabled and self.fusionStrategy in EARLY_FUSION:
            self.combiner = ConcateCombiner()

    def transform(self, m: nn.Module) -> torch.nn.Module:
        """
        Code from https://pytorch.org/docs/stable/fx.html
        """
        graphModel: fx.GraphModule = fx.symbolic_trace(m)
        tune_mode_convbn.turn_on_efficient_conv_bn_eval_for_single_model(graphModel)
        graphModel.recompile()
        return graphModel

    def img2feats(self, x):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # Prepare pointcloud features
        if (
            not self.training
            and self.isRadarEnabled
            and not (self.config.MODEL.FRUSTUM and self.fusionStrategy == "middle")
        ):
            pc_hm = kwargs["pc_dep"]
            maxDistance = self.config.DATASET.MAX_PC_DIST
            slice_ = int(maxDistance) if self.config.DATASET.ONE_HOT_PC else 1
            pc_hm[:, :slice_] /= maxDistance
            pc_hm[:, :slice_] = 1 - pc_hm[:, :slice_]
            kwargs["pc_hm"] = pc_hm

        return self.forward_(*args, **kwargs)

    def forward_(self, x, pc_hm=None, pc_dep=None, calib=None):
        x = self.forwardBackbone(x, pc_hm, pc_dep, calib)
        y = self.forwardHead(x, pc_hm, pc_dep, calib)

        return y

    def forwardBackbone(self, x, pc_hm, pc_dep, calib):
        ## Early fusion strategy
        if self.isRadarEnabled and self.fusionStrategy in EARLY_FUSION:
            x, _ = self.combiner(x, pc_hm)

        ## Backbone forward
        x = self.img2feats(x)
        if isinstance(x, torch.Tensor):
            return [x]
        return x

    def forwardHead(self, x, pc_hm, pc_dep, calib):
        y = []
        for i in range(len(x)):
            y.append(
                self.__getattr__(f"detectHead{'_'+str(i)}")(x[i], pc_hm, pc_dep, calib)
            )
        return y
