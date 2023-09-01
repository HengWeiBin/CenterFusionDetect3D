from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .utils import topk, nms, transposeAndGetFeature

# from .utils import _gather_feat, transposeAndGetFeature
# from .utils import _nms, _topk_channel


# def _update_kps_with_hm(kps, output, batch, num_joints, K, bboxes=None, scores=None):
#     if "hm_hp" in output:
#         hm_hp = output["hm_hp"]
#         hm_hp = _nms(hm_hp)
#         thresh = 0.2
#         kps = (
#             kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous()
#         )  # b x J x K x 2
#         reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
#         hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
#         if "hp_offset" in output or "reg" in output:
#             hp_offset = output["hp_offset"] if "hp_offset" in output else output["reg"]
#             hp_offset = transposeAndGetFeature(hp_offset, hm_inds.view(batch, -1))
#             hp_offset = hp_offset.view(batch, num_joints, K, 2)
#             hm_xs = hm_xs + hp_offset[:, :, :, 0]
#             hm_ys = hm_ys + hp_offset[:, :, :, 1]
#         else:
#             hm_xs = hm_xs + 0.5
#             hm_ys = hm_ys + 0.5

#         mask = (hm_score > thresh).float()
#         hm_score = (1 - mask) * -1 + mask * hm_score
#         hm_ys = (1 - mask) * (-10000) + mask * hm_ys
#         hm_xs = (1 - mask) * (-10000) + mask * hm_xs
#         hm_kps = (
#             torch.stack([hm_xs, hm_ys], dim=-1)
#             .unsqueeze(2)
#             .expand(batch, num_joints, K, K, 2)
#         )
#         dist = ((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5
#         min_dist, min_ind = dist.min(dim=3)  # b x J x K
#         hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
#         min_dist = min_dist.unsqueeze(-1)
#         min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
#             batch, num_joints, K, 1, 2
#         )
#         hm_kps = hm_kps.gather(3, min_ind)
#         hm_kps = hm_kps.view(batch, num_joints, K, 2)
#         mask = hm_score < thresh

#         if bboxes is not None:
#             l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#             t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#             r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#             b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
#             mask = (
#                 (hm_kps[..., 0:1] < l)
#                 + (hm_kps[..., 0:1] > r)
#                 + (hm_kps[..., 1:2] < t)
#                 + (hm_kps[..., 1:2] > b)
#                 + mask
#             )
#         else:
#             l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
#             t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
#             r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
#             b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
#             margin = 0.25
#             l = l - (r - l) * margin
#             r = r + (r - l) * margin
#             t = t - (b - t) * margin
#             b = b + (b - t) * margin
#             mask = (
#                 (hm_kps[..., 0:1] < l)
#                 + (hm_kps[..., 0:1] > r)
#                 + (hm_kps[..., 1:2] < t)
#                 + (hm_kps[..., 1:2] > b)
#                 + mask
#             )
#             # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
#         # mask = mask + (min_dist > 10)
#         mask = (mask > 0).float()
#         kps_score = (1 - mask) * hm_score + mask * scores.unsqueeze(-1).expand(
#             batch, num_joints, K, 1
#         )  # bJK1
#         kps_score = scores * kps_score.mean(dim=1).view(batch, K)
#         # kps_score[scores < 0.1] = 0
#         mask = mask.expand(batch, num_joints, K, 2)
#         kps = (1 - mask) * hm_kps + mask * kps
#         kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)
#         return kps, kps_score
#     else:
#         return kps, kps


def fusionDecode(output, K=100):
    """
    Decoder with Radar point cloud fusion support

    Args:
        output: model output
        K: number of top-k objects to be selected

    Returns:
        ret (dict): dictionary of the decoded output
    """
    if not ("heatmap" in output):
        return {}

    heat = output["heatmap"]
    batch, nclass, height, width = heat.size()

    heat = nms(heat)
    scores, indices, classes, ys0, xs0 = topk(heat, K=K)

    classes = classes.view(batch, K)
    scores = scores.view(batch, K)
    bboxes = None
    centers = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {
        "scores": scores,
        "classes": classes.float(),
        "xs": xs0,
        "ys": ys0,
        "centers": centers,
    }
    if "reg" in output:
        reg = output["reg"]
        reg = transposeAndGetFeature(reg, indices)
        reg = reg.view(batch, K, 2)
        xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs0.view(batch, K, 1) + 0.5
        ys = ys0.view(batch, K, 1) + 0.5

    if "widthHeight" in output:
        widthHeight = output["widthHeight"]
        widthHeight = transposeAndGetFeature(widthHeight, indices)  # B x K x (F)
        widthHeight = widthHeight.view(batch, K, 2)
        widthHeight[widthHeight < 0] = 0
        if widthHeight.size(2) == 2 * nclass:
            widthHeight = widthHeight.view(batch, K, -1, 2)
            cats = classes.view(batch, K, 1, 1).expand(batch, K, 1, 2)
            widthHeight = widthHeight.gather(2, cats.long()).squeeze(2)  # B x K x 2

        bboxes = torch.cat(
            [
                xs - widthHeight[..., 0:1] / 2,
                ys - widthHeight[..., 1:2] / 2,
                xs + widthHeight[..., 0:1] / 2,
                ys + widthHeight[..., 1:2] / 2,
            ],
            dim=2,
        )
        ret["bboxes"] = bboxes

    ## Decode depth with depth residual support
    if "depth" in output:
        depth = output["depth"]
        depth = transposeAndGetFeature(depth, indices)  # B x K x (C)
        cats = classes.view(batch, K, 1, 1)
        if depth.size(2) == nclass:
            depth = depth.view(batch, K, -1, 1)  # B x K x C x 1
            depth = depth.gather(2, cats.long()).squeeze(2)  # B x K x 1

        # add depth residuals to estimated depth values
        if "depth2" in output:
            depth2 = output["depth2"]
            depth2 = transposeAndGetFeature(depth2, indices)  # B x K x [C]
            if depth2.size(2) == nclass:
                depth2 = depth2.view(batch, K, -1, 1)  # B x K x C x 1
                depth2 = depth2.gather(2, cats.long()).squeeze(2)  # B x K x 1
                depth2Mask = (
                    torch.tensor(depth2Mask, device=depth2.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(3)
                )
            depth = depth2

        ret["depth"] = depth

    regressionHeads = [
        "rotation",
        "dimention",
        "amodal_offset",
        "nuscenes_att",
        "velocity",
        "rotation2",
    ]
    for head in regressionHeads:
        if head in output:
            ret[head] = transposeAndGetFeature(output[head], indices).view(batch, K, -1)

    if "rotation2" in output:
        ret["rotation"] = ret["rotation2"]

    return ret
