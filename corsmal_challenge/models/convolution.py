"""contains custom convolution layers"""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        bias: bool = True,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(DepthWiseConv2d, self).__init__()
        self.in_channels: int = in_channels
        self.conv_groups: int = in_channels
        self.expansion: int = expansion
        self.out_channels: int = in_channels * expansion
        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: Tuple[int, int] = stride
        self.bias: Optional[torch.Tensor] = torch.randn(self.out_channels) if bias else None
        self.padding: Union[int, Tuple[int, int]] = padding
        self.weight = torch.randn(  # kernel
            self.out_channels,
            self.in_channels // self.conv_groups,
            kernel_size[0],
            kernel_size[1],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            groups=self.conv_groups,
            padding=self.padding,
        )


class PointWiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(PointWiseConv2d, self).__init__()
        self.in_channels: int = in_channels
        self.conv_groups: int = 1
        self.out_channels: int = out_channels
        self.kernel_size: Tuple[int, int] = (1, 1)
        self.stride: Tuple[int, int] = (1, 1)
        self.bias: Optional[torch.Tensor] = torch.randn(self.out_channels) if bias else None
        self.padding: Union[int, Tuple[int, int]] = padding
        self.weight = torch.randn(
            self.out_channels,
            self.in_channels // self.conv_groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            groups=self.conv_groups,
            padding=self.padding,
        )
