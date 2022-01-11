"""contains audio encoder"""
import torch
import torch.nn.functional as F
from torch import nn

from corsmal_challenge.models.convolution import DepthWiseConv2d, PointWiseConv2d
from corsmal_challenge.models.linear_transformer import LinearTransformerEncoder as LiT


class LogMelEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_encoder_blocks: int = 6,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.05,
    ):
        super(LogMelEncoder, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.dconv = DepthWiseConv2d(in_channels, expansion=2, kernel_size=(5, 5), stride=(2, 1), padding=2)
        self.pconv = PointWiseConv2d(self.dconv.out_channels, out_channels=1, bias=True)

        self.transformer = LiT(
            num_encoder_blocks,
            embed_dim,
            num_heads,
            dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """processes log mel spectrogram data

        Args:
            inputs (torch.Tensor): (batches, channels, sequence_len, embed_dim)

        Returns:
            torch.Tensor: (batches, sequence_len, embed_dim)
        """
        x: torch.Tensor = self.norm(inputs)
        x = F.relu(x)
        x = self.dconv(x)
        x = self.pconv(x)
        x = F.relu(x)
        x = x.squeeze(dim=1)  # batches, sequence_len, embed_dim
        x = self.transformer(x)
        return x
