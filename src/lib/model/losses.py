# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import transposeAndGetFeature
import torch.nn.functional as F


def _only_neg_loss(pred, gt):
    gt = torch.pow(1 - gt, 4)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
    return neg_loss.sum()


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, ind, mask, cat):
        """
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        """
        neg_loss = self.only_neg_loss(out, target)
        pos_pred_pix = transposeAndGetFeature(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = transposeAndGetFeature(output, ind)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, output, mask, ind, target):
        # output: B x F x H x W
        # ind: B x M
        # mask: B x M x F
        # target: B x M x F
        pred = transposeAndGetFeature(output, ind)  # B x M x F
        loss = mask * self.bceloss(pred, target)
        loss = loss.sum() / (mask.sum() + 1e-4)
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = transposeAndGetFeature(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction="mean")


def compute_bin_loss(output, target, mask):
    """
    Compute loss from classifying if angle is in this bin or in the other bin with cross entropy.
    Use the prediction if its in this bin (=1) AND if its in the other bin (=0) because bins overlap
    and the angle can be in both bins.
    Mask predictions by wether or not the annotations even have a gt angle labeled or not.
    Don't learn from making a prediction when there is no target.
    """
    nonzero_idx = mask.nonzero()[:, 0].long()
    if nonzero_idx.shape[0] > 0:  # if there are any annotations with a labeled angle
        output_nz = output.index_select(0, nonzero_idx)
        target_nz = target.index_select(0, nonzero_idx)
        return F.cross_entropy(output_nz, target_nz, reduction="mean")
    else:
        # loss would be NaN if computed normally when no annotation is given
        # set to different grad_fn but not relevant since loss is zero
        return torch.tensor(0.0).cuda()


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0])
        )
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0])
        )
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1])
        )
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1])
        )
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


## Loss function for depth with support for class-based depth
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, output, target, ind, mask, cat):
        """
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        """
        pred = transposeAndGetFeature(output, ind)  # B x M x (C)
        if pred.shape[2] > 1:
            pred = pred.gather(2, cat.unsqueeze(2))  # B x M
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss
