from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .utils import topk, nms, transposeAndGetFeature


def fusionDecode(outputs, outputSize=(112, 200), K=100, norm2d=False):
    """
    Decoder with Radar point cloud fusion support

    Args:
        output: model output
        outputSize: size of the output feature map
        K: number of top-k objects to be selected
        norm2d: normalize the 2D output

    Returns:
        ret (dict): dictionary of the decoded output
    """
    assert isinstance(outputs, list), "output must be a list of dictionaries"

    checkHeatmap = ["heatmap" in output for output in outputs]
    if not any(checkHeatmap):
        return {}

    # get topk from all layers of outputs
    layerTopKs = []
    for i, output in enumerate(outputs):
        if not checkHeatmap[i]:
            continue

        heat = output["heatmap"]
        batch, _, height, width = heat.size()

        heat = nms(heat)
        scores_layer, indices_layer, classes_layer, ys_layer, xs_layer = topk(heat, K=K)
        ys_layer = ys_layer / height
        xs_layer = xs_layer / width
        layerTopKs.append(
            [scores_layer, indices_layer, classes_layer, ys_layer, xs_layer]
        )

    # combine topk from all layers
    layerTopK = [*zip(*layerTopKs)]
    scores_all = torch.cat(layerTopK[0], dim=-1)  # (B, K * nLayer)
    scores, indices = torch.topk(scores_all, K, dim=-1)  # (B, K)

    classes_all = torch.cat(layerTopK[2], dim=-1)
    ys_all = torch.cat(layerTopK[3], dim=-1)
    xs_all = torch.cat(layerTopK[4], dim=-1)

    classes_final = classes_all.gather(1, indices)
    ys_final = ys_all.gather(1, indices)
    xs_final = xs_all.gather(1, indices)

    centers = torch.cat([xs_final.unsqueeze(2), ys_final.unsqueeze(2)], dim=2)
    ret = {
        "scores": scores,
        "classIds": classes_final.float(),
        "centers": centers,
    }

    batch_size, _, height, width = outputs[0]["heatmap"].shape
    decode = {
        "reg": torch.zeros((batch_size, 0, 2), device=scores.device),
        "widthHeight": torch.zeros((batch_size, 0, 2), device=scores.device),
        "depth": torch.zeros((batch_size, 0, 1), device=scores.device),
        "rotation": torch.zeros((batch_size, 0, 8), device=scores.device),
        "dimension": torch.zeros((batch_size, 0, 3), device=scores.device),
        "amodal_offset": torch.zeros((batch_size, 0, 2), device=scores.device),
        "nuscenes_att": torch.zeros((batch_size, 0, 8), device=scores.device),
        "velocity": torch.zeros((batch_size, 0, 3), device=scores.device),
    }

    for i, output in enumerate(outputs):
        indices_layer = layerTopKs[i][1]
        if "uncertainty" in output:
            depthConfLayer = output["uncertainty"]
            depthConfLayer = transposeAndGetFeature(depthConfLayer, indices_layer)
            depthConfLayer = depthConfLayer.view(batch_size, K, 1)
            depthConfLayer = torch.exp(-torch.exp(depthConfLayer)).squeeze()
            ret["scores"] = ret["scores"] * depthConfLayer

        if "reg" in output:
            reg_layer = output["reg"]
            reg_layer = transposeAndGetFeature(reg_layer, indices_layer)
            reg_layer = reg_layer.view(batch_size, K, 2)
            decode["reg"] = torch.cat([decode["reg"], reg_layer], dim=1)

        if "widthHeight" in output:
            widthHeight_layer = output["widthHeight"]
            widthHeight_layer = transposeAndGetFeature(widthHeight_layer, indices_layer)
            widthHeight_layer = widthHeight_layer.view(batch_size, K, 2)
            decode["widthHeight"] = torch.cat(
                [decode["widthHeight"], widthHeight_layer], dim=1
            )

        if "depth2" in output:
            depth_layer = output["depth2"]
            depth_layer = transposeAndGetFeature(depth_layer, indices_layer)
            depth_layer = depth_layer.view(batch_size, K, 1)
            decode["depth"] = torch.cat([decode["depth"], depth_layer], dim=1)
        elif "depth" in output:
            depth_layer = output["depth"]
            depth_layer = transposeAndGetFeature(depth_layer, indices_layer)
            depth_layer = depth_layer.view(batch_size, K, 1)
            decode["depth"] = torch.cat([decode["depth"], depth_layer], dim=1)

        regressionHeads = [
            "rotation",
            "dimension",
            "amodal_offset",
            "nuscenes_att",
            "velocity",
        ]

        if "rotation2" in output:
            output["rotation"] = output.pop("rotation2")

        for head in regressionHeads:
            if head in output:
                regression_layer = output[head]
                regression_layer = transposeAndGetFeature(
                    regression_layer, indices_layer
                )
                regression_layer = regression_layer.view(batch_size, K, -1)
                decode[head] = torch.cat([decode[head], regression_layer], dim=1)

    xs_final *= outputSize[1]
    ys_final *= outputSize[0]
    if decode["reg"].size(1) > 0:
        reg = decode["reg"]
        reg = reg.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        xs = xs_final.view(batch_size, K, 1) + reg[:, :, 0:1]
        ys = ys_final.view(batch_size, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs_final.view(batch_size, K, 1) + 0.5
        ys = ys_final.view(batch_size, K, 1) + 0.5

    outputWidthHeight = (
        torch.tensor(outputSize[::-1], device=outputs[0]["heatmap"].device)
        if norm2d
        else 1
    )
    if decode["widthHeight"].size(1) > 0:
        widthHeight = decode["widthHeight"]
        widthHeight = widthHeight.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        widthHeight[widthHeight < 0] = 0
        widthHeight *= outputWidthHeight  # denormalize

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

    decode["amodal_offset"] *= outputWidthHeight  # denormalize
    for head in regressionHeads + ["depth"]:
        if decode[head].size(1) > 0:
            regression = decode[head]
            regression = regression.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, regression.shape[-1])
            )
            ret[head] = regression

    return ret
