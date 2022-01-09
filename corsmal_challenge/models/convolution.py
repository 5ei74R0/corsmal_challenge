"""contains custom convolution layers"""
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class DepthWiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        bias: bool = True,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(DepthWiseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=in_channels * expansion,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.expansion: int = expansion

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, self.weight, self.bias)


class PointWiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(PointWiseConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, self.weight, self.bias)


class InvertedResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
        expansion: int = 6,
    ):
        super(InvertedResBlock, self).__init__()
        padding_size = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.pconv1 = PointWiseConv2d(
            in_channels=channels,
            out_channels=channels * expansion,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(self.pconv1.out_channels)
        self.dconv = DepthWiseConv2d(
            in_channels=self.pconv1.out_channels,
            expansion=1,
            kernel_size=kernel_size,
            stride=(1, 1),
            bias=bias,
            padding=padding_size,
        )
        self.bn3 = nn.BatchNorm2d(self.dconv.out_channels)
        self.pconv2 = PointWiseConv2d(
            self.dconv.out_channels,
            channels,
            bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.bn1(inputs)
        x = F.relu6(x, inplace=True)
        x = self.pconv1(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.dconv(x)
        x = self.bn3(x)
        x = F.relu6(x, inplace=True)
        x = self.pconv2(x)
        res = x + inputs
        return res
