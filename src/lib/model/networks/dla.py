from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
from torch.utils import model_zoo
import numpy as np
from torchvision.ops import deform_conv2d

from .base_model import BaseModel
from model.utils import initConv2dWeights, initUpModuleWeights


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels, momentum=0.1),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0], momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    # TODO delete pre
    def forward(self, x):#, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        # if pre_img is not None:
        #     x = x + self.pre_img_layer(pre_img)
        # if pre_hm is not None:
        #     x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            y.append(x)

        return y

    def load_pretrained_model(self, data="imagenet", name="dla34", hash="ba72cf86"):
        if name.endswith(".pth"):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.load_state_dict(model_weights, strict=False)


def get_model_url(data="imagenet", name="dla34", hash="ba72cf86"):
    return os.path.join(
        "http://dl.yf.io/dla/models", data, "{}-{}.pth".format(name, hash)
    )


def getModel(model="dla34", pretrained=True, **kwargs):
    if model == "dla34":
        model = DLA(
            [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs
        )
        if pretrained:
            model.load_pretrained_model(data="imagenet", name="dla34", hash="ba72cf86")
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    return model


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1):
        super(GlobalConv, self).__init__()
        convLeft = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                bias=False,
                dilation=dilation,
                padding=(dilation * (kernel_size // 2), 0),
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                bias=False,
                dilation=dilation,
                padding=(0, dilation * (kernel_size // 2)),
            ),
        )
        convRight = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                bias=False,
                dilation=dilation,
                padding=(0, dilation * (kernel_size // 2)),
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                bias=False,
                dilation=dilation,
                padding=(dilation * (kernel_size // 2), 0),
            ),
        )
        initConv2dWeights(convLeft)
        initConv2dWeights(convRight)
        self.convLeft = convLeft
        self.convRight = convRight
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=0.1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convLeft(x) + self.convRight(x)
        x = self.activation(x)
        return x


class DeformConv(nn.Module):
    """
    Deformable ConvNets v2: More Deformable, Better Results
    link: https://arxiv.org/abs/1811.11168
    """

    def __init__(self, in_channels, out_channels):
        super(DeformConv, self).__init__()
        # Original conv layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Declare offset layer and initialize to 0
        self.conv_offset = nn.Conv2d(
            in_channels, 18, kernel_size=3, stride=1, padding=1
        )
        nn.init.constant_(self.conv_offset.weight, 0)

        # Declare mask layer and initialize to 0.5
        self.conv_mask = nn.Conv2d(in_channels, 9, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.conv_mask.weight, 0.5)

        # Declare activation function
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=0.1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        x = deform_conv2d(
            input=x, offset=offset, mask=mask, weight=self.conv.weight, padding=(1, 1)
        )
        x = self.activation(x)
        return x


class IDAUp(nn.Module):
    def __init__(
        self, out_channels, in_channels, up_f, node_type=(DeformConv, DeformConv)
    ):
        super(IDAUp, self).__init__()
        for i in range(1, len(in_channels)):
            f = int(up_f[i])
            proj = node_type[0](in_channels[i], out_channels)
            node = node_type[1](out_channels, out_channels)

            up = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=out_channels,
                bias=False,
            )
            initUpModuleWeights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(
        self, startp, channels, scales, in_channels=None, node_type=DeformConv
    ):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(
                    channels[j],
                    in_channels[j:],
                    scales[j:] // scales[j],
                    node_type=node_type,
                ),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_NODE = {
    "DeformConv": (DeformConv, DeformConv),
    "GlobalConv": (Conv, GlobalConv),
    "Conv": (Conv, Conv),
}


class DLASeg(BaseModel):
    def __init__(self, num_layers, config):
        super(DLASeg, self).__init__(1, 64 if num_layers == 34 else 128, config=config)
        down_ratio = 4
        self.config = config
        self.node_type = DLA_NODE[config.MODEL.DLA.NODE]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = getModel("dla34", pretrained=(config.MODEL.LOAD_DIR == ""))

        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(
            self.first_level,
            channels[self.first_level :],
            scales,
            node_type=self.node_type,
        )
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel,
            channels[self.first_level : self.last_level],
            [2**i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type,
        )

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.base(x, pre_img, pre_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]
