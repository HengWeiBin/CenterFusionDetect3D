from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .utils import topk, nms, transposeAndGetFeature

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
        "dimension",
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
