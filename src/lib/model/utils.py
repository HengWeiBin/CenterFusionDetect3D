import torch
from torch import nn
import math


def topk(heatmap, K=100):
    """
    Returns the k largest elements of the heatmap

    Args:
        heatmap: heatmap
        K: number of elements to return

    Returns:
        topk_heat: heatmap of the top k elements
        topk_heats: heatmap of the top k elements in onehot format
        topk_inds: indices of the top k elements in input heatmap
        topk_classes: classes of the top k elements
        topk_ys: y coordinates of the top k elements
        topk_xs: x coordinates of the top k elements
    """
    batch, nclass, height, width = heatmap.size()

    topk_heats, topk_inds = torch.topk(heatmap.view(batch, nclass, -1), K)

    # get every topk heat xy coordinates
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = topk_inds % width

    # get every topk class and their xy coordinates
    topk_heat, topk_ind = torch.topk(topk_heats.view(batch, -1), K)
    topk_classes = (topk_ind / K).int()
    topk_inds = getFeature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = getFeature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = getFeature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_heat, topk_inds, topk_classes, topk_ys, topk_xs


def getFeature(feature, indices):
    """
    Reshapes the feature map and return the feature of the specified indices

    Args:
        feature: features
        indices: indices of the elements

    Returns:
        feature of the specified indices
    """
    dim = feature.size(2)
    indices = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dim)
    feature = feature.gather(1, indices)
    return feature


def transposeAndGetFeature(feature, indices):
    """
    Transposes the feature map and return the feature of the specified indices

    Args:
        feature: features
        indices: indices of the elements

    Returns:
        feature of the specified indices
    """
    feature = feature.permute(0, 2, 3, 1).contiguous()
    feature = feature.view(feature.size(0), -1, feature.size(3))
    feature = getFeature(feature, indices)
    return feature


def initConv2dWeights(layers):
    """
    This function initializes the weights of the conv2d layers to 0

    Args:
        layers: pytorch module

    Returns:
        None
    """
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initUpModuleWeights(up):
    """
    This function initializes the weights of the up module

    Args:
        up: pytorch module

    Returns:
        None
    """
    weights = up.weight.data
    floor = math.ceil(weights.size(2) / 2)
    ceil = (2 * floor - 1 - floor % 2) / (2.0 * floor)
    for i in range(weights.size(2)):
        for j in range(weights.size(3)):
            weights[0, 0, i, j] = (1 - math.fabs(i / floor - ceil)) * (
                1 - math.fabs(j / floor - ceil)
            )
    for ceil in range(1, weights.size(0)):
        weights[ceil, 0, :, :] = weights[0, 0, :, :]


def nms(heatmap, kernel=3):
    """
    Performs non-maximum suppression on the heatmap
    Retains the local maxima of the heatmap

    Args:
        heatmap: heatmap
        kernel: kernel size

    Returns:
        heatmap after non-maximum suppression
    """
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def sigmoidDepth(x):
    """
    Sigmoid function for depth

    Args:
        x: input tensor

    Returns:
        sigmoid of x
    """
    return 1.0 / (x.sigmoid() + 1e-6) - 1.0


def dictToCuda(dict_):
    for key in dict_:
        if isinstance(dict_[key], dict):
            dict_[key] = dictToCuda(dict_[key])
        elif isinstance(dict_[key], torch.Tensor):
            dict_[key] = dict_[key].cuda()
        else:
            continue
    return dict_
