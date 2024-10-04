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
import torch.nn.functional as F

from .utils import transposeAndGetFeature
from utils.image import affineTransform
from utils.ddd import cvtImgToCamCoord, get3dBox
from utils.pointcloud import get_alpha



class GIoU2DLoss(nn.Module):
    """
    Inplement of Generalized Intersection over Union Loss

    ref: https://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf
    """

    def forward(self, pred, target, mask):
        """
        Args:
            pred: (tensor) predictions from model (B x K x 4) (x1, y1, x2, y2)
            target: (tensor) targets (B x K x 4) (x1, y1, x2, y2)
            mask: (tensor) mask of targets (B x K)
        """
        eps = 1e-7
        if (totalTarget := mask.sum()) == 0:
            return totalTarget + eps

        # Intersection area
        pred_x1, pred_y1, pred_x2, pred_y2 = (
            pred[..., 0],
            pred[..., 1],
            pred[..., 2],
            pred[..., 3],
        )
        target_x1, target_y1, target_x2, target_y2 = (
            target[..., 0],
            target[..., 1],
            target[..., 2],
            target[..., 3],
        )
        interX = (torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1)).clamp(0)
        interY = (torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1)).clamp(0)
        inter = interX * interY

        # Union Area
        # Don't trust prediction value will be always positive
        pred_w, pred_h = (pred_x2 - pred_x1).clamp(0), (pred_y2 - pred_y1).clamp(0)
        target_w, target_h = target_x2 - target_x1, target_y2 - target_y1
        union = pred_w * pred_h + target_w * target_h - inter
        # Target area should be a positive and non-zero value if it is a valid target
        # So, union no need to add eps

        # IoU
        iou = inter / union

        # Smallest enclosing box
        convexWidth = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        convexHeight = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        convexArea = convexWidth * convexHeight

        giou = iou - (convexArea - union) / convexArea
        giou = giou.clamp(min=-1.0, max=1.0)

        # Mask should filter out invalid target(nan),
        # so it should be safe to use mean directly
        return 1.0 - giou[mask.bool()].mean()


class Bbox2DLoss(nn.Module):
    """
    This loss decode 2D bounding box from local offset and dimension
    and calculate the loss with GIoU2DLoss
    """

    def __init__(self):
        super(Bbox2DLoss, self).__init__()
        self.iouloss = GIoU2DLoss()

    def forward(self, localOffset, dimension2D, centerInt, target, indices, mask):
        """
        Args:
            localOffset: (tensor) local offset (B x 2 x H x W)
            dimension2D: (tensor) 2D dimension (B x 2 x H x W)
            centerInt: (tensor) center (B x M x 2)
            target: (tensor) targets (B x M x 4)
            indices: (tensor) indices of targets (B x K)
            mask: (tensor) mask of targets (B x M)
        """
        # Decode (B x 2 x H x W) -> (B x M x 2)
        localOffset = transposeAndGetFeature(localOffset, indices)
        dimension2D = transposeAndGetFeature(dimension2D, indices)
        
        # Calculate predict bounding box
        center = centerInt + localOffset
        pred = torch.cat(
            [
                center[..., 0:1] - dimension2D[..., 0:1] / 2,
                center[..., 1:2] - dimension2D[..., 1:2] / 2,
                center[..., 0:1] + dimension2D[..., 0:1] / 2,
                center[..., 1:2] + dimension2D[..., 1:2] / 2,
            ],
            dim=2,
        )  # (B x M x 4)
        bbox2dLoss = self.iouloss(pred, target, mask)
        return bbox2dLoss


class Bbox3DLoss(nn.Module):
    def forward(
        self,
        output,
        centerInt,
        transMat,
        calib,
        target,
        indices,
        mask,
        depthHead="depth",
    ):
        """
        Args:
            output: (dict) output from model with {rotation, depth, dimension, amodalOffset}
            centerInt: (tensor) center (B x M x 2)
            transMat: (array) transformation matrix (2 x 3)
            calib: (tensor) calibration matrix (B x 3 x 4)
            target: (tensor) targets (B x M x 8 x 3)
            indices: (tensor) indices of targets (B x K)
            mask: (tensor) mask of targets (B x M)
            depthHead: (str) name of depth head in output
        """
        if mask.sum() == 0:
            return 0

        # Decode (B x N x H x W) -> (B x M x N)
        rotation = transposeAndGetFeature(output["rotation"], indices)
        depth = transposeAndGetFeature(output[depthHead], indices)
        dimension = transposeAndGetFeature(output["dimension"], indices)
        if "amodal_offset" in output:
            amodalOffset = transposeAndGetFeature(output["amodal_offset"], indices)
        else:
            amodalOffset = torch.zeros_like(centerInt)

        ## Decode 3D bounding boxes
        batch_size, maxObj = indices.shape

        # calculate 3D bounding box center
        center3D = centerInt + amodalOffset
        center3D = affineTransform(center3D.view(-1, 2), transMat).view(
            batch_size, maxObj, 2
        )

        # calculate 3D dimension, location, and yaw(rotation_y)
        alpha = get_alpha(rotation.view(-1, 8)).view(batch_size, maxObj)
        locations, yaws = cvtImgToCamCoord(center3D, alpha, dimension, depth, calib)

        # calculate 3D bounding box and project to 2D
        bbox3D_pred = get3dBox(dimension, locations, yaws)

        bbox3dLoss = F.l1_loss(
            bbox3D_pred[..., ::2],
            target[..., ::2],  # bbox3D_target,
            reduction="none",
        )

        bbox3dLoss = bbox3dLoss[mask.bool()].mean()
        return bbox3dLoss


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def only_neg_loss(self, pred, gt):
        gt = torch.pow(1 - gt, 4)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
        return neg_loss.sum()

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

    def forward(self, output, mask, ind, target, reduction="mean"):
        nSamples = mask.sum()
        if nSamples == 0:
            nSamples = torch.tensor(1e7).to(mask.device)

        pred = transposeAndGetFeature(output, ind)
        loss = F.l1_loss(pred * mask, target * mask, reduction="none")

        if reduction == "mean":
            return loss.sum() / nSamples
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, output, mask, ind, target):
        # output: B x F x H x W
        # ind: B x M
        # mask: B x M x F
        # target: B x M x F

        # if mask.sum() == 0:
        #     return 0
        nSamples = mask.sum()
        if nSamples == 0:
            nSamples = torch.tensor(1e7).to(mask.device)

        pred = transposeAndGetFeature(output, ind)  # B x M x F
        loss = mask * self.bceloss(pred, target)
        loss = loss.sum() / nSamples
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = transposeAndGetFeature(output, ind)
        if mask.sum() == 0:
            return (pred * mask.view(*mask.shape, 1)).mean()

        loss = self.compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

    def compute_res_loss(self, output, target):
        return F.smooth_l1_loss(output, target, reduction="mean")

    def compute_bin_loss(self, output, target, mask):
        """
        Compute loss from classifying if angle is in this bin or in the other bin with cross entropy.
        Use the prediction if its in this bin (=1) AND if its in the other bin (=0) because bins overlap
        and the angle can be in both bins.
        Mask predictions by wether or not the annotations even have a gt angle labeled or not.
        Don't learn from making a prediction when there is no target.
        """
        nonzero_idx = mask.nonzero()[:, 0].long()
        if (
            nonzero_idx.shape[0] > 0
        ):  # if there are any annotations with a labeled angle
            output_nz = output.index_select(0, nonzero_idx)
            target_nz = target.index_select(0, nonzero_idx)
            return F.cross_entropy(output_nz, target_nz, reduction="mean")
        else:
            # loss would be NaN if computed normally when no annotation is given
            # set to different grad_fn but not relevant since loss is zero
            return torch.tensor(0.0, requires_grad=True).to(output.device)

    def compute_rot_loss(self, output, target_bin, target_res, mask):
        # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
        #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
        # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
        # target_res: (B, 128, 2) [bin1_res, bin2_res]
        # mask: (B, 128, 1)
        output = output.view(-1, 8)
        target_bin = target_bin.view(-1, 2)
        target_res = target_res.view(-1, 2)
        mask = mask.view(-1, 1)
        loss_bin1 = self.compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
        loss_bin2 = self.compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
        loss_res = torch.zeros_like(loss_bin1)
        if target_bin[:, 0].nonzero().shape[0] > 0:
            idx1 = target_bin[:, 0].nonzero()[:, 0]
            valid_output1 = torch.index_select(output, 0, idx1.long())
            valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
            loss_sin1 = self.compute_res_loss(
                valid_output1[:, 2], torch.sin(valid_target_res1[:, 0])
            )
            loss_cos1 = self.compute_res_loss(
                valid_output1[:, 3], torch.cos(valid_target_res1[:, 0])
            )
            loss_res += loss_sin1 + loss_cos1
        if target_bin[:, 1].nonzero().shape[0] > 0:
            idx2 = target_bin[:, 1].nonzero()[:, 0]
            valid_output2 = torch.index_select(output, 0, idx2.long())
            valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
            loss_sin2 = self.compute_res_loss(
                valid_output2[:, 6], torch.sin(valid_target_res2[:, 1])
            )
            loss_cos2 = self.compute_res_loss(
                valid_output2[:, 7], torch.cos(valid_target_res2[:, 1])
            )
            loss_res += loss_sin2 + loss_cos2
        return loss_bin1 + loss_bin2 + loss_res


class UncertaintyDepthLoss(nn.Module):
    """
    Implement uncertainty-attenuated L1 Loss from Cluster Fusion (eq.14)

    ClusterFusion: Leveraging Radar Spatial Features for Radar-Camera 3D Object Detection in Autonomous Vehicles
    ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10302296
    """

    def __init__(self):
        super(UncertaintyDepthLoss, self).__init__()
        self.l1_loss = RegWeightedL1Loss()

    def forward(self, output, mask, ind, target, uncertainty):
        """
        Args:
            output: (tensor) predictions from model (B x 1 x H x W)
            target: (tensor) targets (B x M x 1)
            ind: (tensor) indices of targets (B x M)
            mask: (tensor) mask of targets (B x M x 1)
            uncertainty: (tensor) uncertainty of targets (B x K x 1)
        """
        loss = self.l1_loss(output, mask, ind, target, reduction="none")
        sigmaLog = transposeAndGetFeature(uncertainty, ind)
        sigma = torch.exp(-sigmaLog)
        if mask.sum() == 0:
            uncertaintyLoss = (loss * sigma + sigmaLog).mean()
            loss = loss.mean()
        else:
            uncertaintyLoss = (loss * sigma + sigmaLog)[mask.bool()].mean()
            loss = loss[mask.bool()].mean()
        return loss, uncertaintyLoss


class DecoupledLoss(nn.Module):
    """
    Implement decoupled loss from MonoFlex (eq. 4)

    Objects are Different: Flexible Monocular 3D Object Detection
    ref: https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Objects_Are_Different_Flexible_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf
    """

    def __init__(self):
        super(DecoupledLoss, self).__init__()
        self.l1_loss = RegWeightedL1Loss()

    def forward(self, output, mask, ind, target, truncMask):
        loss = self.l1_loss(output, mask, ind, target, reduction="none")
        insideLoss = loss * ~truncMask.bool()
        outsideLoss = ((loss * truncMask) + 1).log()
        loss = (
            (insideLoss + outsideLoss).mean()
            if mask.sum() == 0
            else (insideLoss + outsideLoss)[mask.bool()].mean()
        )

        return loss
