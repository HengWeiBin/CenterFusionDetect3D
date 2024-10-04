from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from utils import pointcloud as pc
from model.utils import initConv2dWeights, sigmoidDepth
from .fusionModules import ConcateCombiner


# ------------------------------------------------------------------------------
#                         Sub Modules for Heads
# ------------------------------------------------------------------------------
class SigmoidDepth(nn.Module):
    def forward(self, x):
        return sigmoidDepth(x)


class SigmoidHeatmap(nn.Module):
    def forward(self, x):
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
# ------------------------------------------------------------------------------
#                         Sub Modules for Heads End
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                        Heads for 3D Object Detection
# ------------------------------------------------------------------------------

class DetectHead(nn.Module): 
    # Base class for all fusion strategies
    # By default, the fusion strategy is None, performing only image detection
    def __init__(self, in_channels_head, config):
        self.isRadarEnabled = config.DATASET.RADAR_PC
        self.fusion_strategy = (
            config.MODEL.FUSION_STRATEGY if self.isRadarEnabled else None
        )
        self.isOneHotPC = config.DATASET.ONE_HOT_PC if self.isRadarEnabled else False
        self.pc_max_dist = config.DATASET.MAX_PC_DIST
        self.isFrustumEnabled = config.MODEL.FRUSTUM
        self.dataset = config.DATASET.DATASET
        self.config = config
        self.heads = config.heads
        self.secondary_heads = []

        assert not (self.isRadarEnabled and self.isOneHotPC and self.isFrustumEnabled)
        assert not (self.isRadarEnabled and self.fusion_strategy is None)
        assert not (self.isOneHotPC and self.fusion_strategy in ["middle2", "deep2"])

        super(DetectHead, self).__init__()

        self.depthSigmoid = SigmoidDepth()
        in_channels_heads = {head: sum(in_channels_head) for head in self.heads}
        self.head_activation = nn.ReLU
        self.configure_heads(in_channels_heads)

    def configure_heads(self, in_channels_heads):
        for head in self.heads:
            nOutputs = self.heads[head]
            head_conv = self.config.head_conv[head]
            if len(head_conv) > 0:
                head_out = nn.Conv2d(
                    head_conv[-1],
                    nOutputs,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                head_in = nn.Conv2d(
                    in_channels_heads[head],
                    head_conv[0],
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )

                convLayer = lambda i, o: nn.Conv2d(i, o, kernel_size=1, bias=True)
                actLayer = lambda *_: self.head_activation(inplace=True)
                headSequence = [
                    head_in,
                    actLayer(0, 0),
                    *(
                        layer(head_conv[i - 1], head_conv[i])
                        for i in range(1, len(head_conv))
                        for layer in [convLayer, actLayer]
                    ),
                    head_out,
                ]
                if "heatmap" in head:
                    head_out.bias.data.fill_(-4.6)
                    headSequence.append(SigmoidHeatmap())
                    headSequence = nn.Sequential(*headSequence)
                else:
                    headSequence = nn.Sequential(*headSequence)
                    initConv2dWeights(headSequence)
            else:
                headSequence = nn.Sequential(
                    nn.Conv2d(
                        in_channels_heads[head],
                        nOutputs,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                )
                if "heatmap" in head:
                    headSequence[-1].bias.data.fill_(-4.6)
                else:
                    initConv2dWeights(headSequence)

            self.__setattr__(head, headSequence)

    def forward(self, imgFeatures, pc_hm=None, pc_dep=None, calib=None):
        return self.fowardFirstStage(imgFeatures, calib)

    def fowardFirstStage(self, firstStageFeats, calib):
        y = {}
        ## Run the first stage heads
        for head in self.config.heads:
            if head not in self.secondary_heads:
                y[head] = getattr(self, head)(firstStageFeats)

        if "depth" in y:
            y["depthMap"] = y["depth"]
            y["depth"] = self.depthSigmoid(y["depth"])

        y["calib"] = calib
        return y


class CenterFusionHead(DetectHead): 
    # Middle fusion refers to the fusion strategy in CenterFusion
    # Ref: "Centerfusion: Center-based radar and camera fusion for 3d object detection"

    def __init__(self, in_channels_head, config):
        super(CenterFusionHead, self).__init__(in_channels_head, config)

        # Define fusion modules
        self.combiner = ConcateCombiner()

        # Middle fusion heads in CenterFusion
        for head in [
            "velocity",
            "nuscenes_att",
            "depth2",
            "rotation2",
        ]:
            if head in self.heads:
                self.secondary_heads.append(head)

        in_channels_heads = {head: sum(in_channels_head) for head in self.heads}
        nDepthChannels = int(self.pc_max_dist) if self.isOneHotPC else 1
        nVelChannels = 0 if self.dataset == "dualradar" else 2
        for head in self.secondary_heads:
            in_channels_heads[head] = (
                sum(in_channels_head) + nDepthChannels + nVelChannels
            )

        self.configure_heads(in_channels_heads)

    def forward(self, imgFeatures, pc_hm=None, pc_dep=None, calib=None):
        y = super(CenterFusionHead, self).forward(imgFeatures, pc_hm, pc_dep, calib)

        if self.config.DATASET.ONE_HOT_PC:
            pc_slice = int(self.config.DATASET.MAX_PC_DIST)
        else:
            pc_slice = 1
        y["pc_hm_in"] = pc_dep[:, :pc_slice]

        ## Run the second stage heads
        ## get pointcloud heatmap
        if not self.training and self.isFrustumEnabled:
            pc_hm = pc.getPcFrustumHeatmap(y, pc_dep, calib, self.config)
            # Frustum association only available after first stage
            # because we need 2d bbox to perform it

        y["pc_hm"] = pc_hm[:, 0, :, :].unsqueeze(1)
        sec_feats, pc_hm_out = self.combiner(imgFeatures, pc_hm)
        for head in self.secondary_heads:
            y[head] = getattr(self, head)(sec_feats)
        y["pc_hm_out"] = pc_hm_out[:, :pc_slice]

        # Overwrite depthmap by depth2 prediction
        if "depth2" in y:
            y["depthMap"] = y["depth2"]
            y["depth2"] = self.depthSigmoid(y["depth2"])
        return y
