from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class ConcateCombiner(nn.Module):
    """
    Concatenate image features and radar features
    """

    def __init__(self):
        super(ConcateCombiner, self).__init__()

    def forward(self, x1, x2):
        """
        Args:
            x1: torch.Tensor, image features
            x2: torch.Tensor, radar features

        Returns:
            y: combined features
            x2_: enhanced radar features
        """
        if 0 in x2.shape:
            return x1

        resize_dim = len(x1.shape) - 2
        x2_ = F.interpolate(x2, size=x1.shape[-resize_dim:], mode="nearest")
        y = torch.cat([x1, x2_], dim=1)

        return y, x2_
