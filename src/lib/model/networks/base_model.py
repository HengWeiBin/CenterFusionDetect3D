from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from utils import pointcloud as pc
from model.utils import initConv2dWeights

class BaseModel(nn.Module):
    def __init__(self, num_stacks, last_channel, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.num_stacks = num_stacks
        self.heads = config.heads
        self.secondary_heads = ["velocity", "nuscenes_att", "depth2", "rotation2"]

        last_channels = {head: last_channel for head in config.heads}
        for head in self.secondary_heads:
            last_channels[head] = last_channel + 3  # 3 = pc_dep channel

        for head in self.heads:
            classes = self.heads[head]
            head_conv = config.head_conv[head]
            if len(head_conv) > 0:
                head_out = nn.Conv2d(
                    head_conv[-1],
                    classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                head_in = nn.Conv2d(
                    last_channels[head],
                    head_conv[0],
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )

                convLayer = lambda i, o: nn.Conv2d(i, o, kernel_size=1, bias=True)
                reluLayer = lambda i, o: nn.ReLU(inplace=True)
                headSequence = nn.Sequential(
                    head_in,
                    reluLayer(0, 0),
                    *(
                        layer(head_conv[i - 1], head_conv[i])
                        for i in range(1, len(head_conv))
                        for layer in [convLayer, reluLayer]
                    ),
                    head_out
                )
                if "heatmap" in head:
                    headSequence[-1].bias.data.fill_(-4.6)
                else:
                    initConv2dWeights(headSequence)
            else:
                headSequence = nn.Conv2d(
                    last_channels[head],
                    classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                if "heatmap" in head:
                    headSequence.bias.data.fill_(-4.6)
                else:
                    initConv2dWeights(headSequence)

            self.__setattr__(head, headSequence)

    def img2feats(self, x):
        raise NotImplementedError

    def forward(self, x, pc_hm=None, pc_dep=None, calib=None):
        ## extract features from image
        feats = self.img2feats(x)
        out = []
        for stack in range(self.num_stacks):
            z = {}

            ## Run the first stage heads
            for head in self.heads:
                if head not in self.secondary_heads:
                    z[head] = self.__getattr__(head)(feats[stack])

            if self.config.DATASET.NUSCENES.RADAR_PC:
                ## get pointcloud heatmap
                if not self.training:
                    if self.config.MODEL.FRUSTUM:
                        pc_hm = pc.getPcFrustumHeatmap(z, pc_dep, calib, self.config)
                    else:
                        pc_hm = pc_dep
                        pc_hm[0] /= self.config.DATASET.NUSCENES.MAX_PC_DIST

                z["pc_hm"] = pc_hm[:, 0, :, :].unsqueeze(1)  # pc_dep

                ## Run the second stage heads
                sec_feats = [feats[stack], pc_hm]
                sec_feats = torch.cat(sec_feats, 1)
                for head in self.secondary_heads:
                    z[head] = self.__getattr__(head)(sec_feats)

            out.append(z)

        return out
