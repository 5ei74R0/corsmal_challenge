"""
contains linear transformer
- https://arxiv.org/pdf/2006.16236.pdf
"""
import math
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import nn

from corsmal_challenge.models.activation import SquaredReLU


class MultiheadedLinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        attn_dropout: float = 0.05,
        proj_dropout: float = 0.05,
    ):
        super(MultiheadedLinearSelfAttention, self).__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dim % num heads != 0"
        self.head_dim: int = embed_dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        self.qkv: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout: Callable[..., torch.Tensor] = nn.Dropout(attn_dropout)
        self.projection: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout: Callable[..., torch.Tensor] = nn.Dropout(proj_dropout)

    @torch.jit.script
    def _kernel_fn(x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return F.elu(x) + 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batches, sequence_len, _ = inputs.shape
        qkv: torch.Tensor = (
            self.qkv(inputs).reshape(batches, sequence_len, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        )
        q, k, v = qkv  # batches, sequence_len, num_heads, head_dim

        kv: torch.Tensor = torch.einsum("bshd,bshm->bhmd", self._kernel_fn(k), v)
        z: torch.Tensor = 1 / (torch.einsum("bshd,bhd->bsh", self._kernel_fn(q), self._kernel_fn(k).sum(dim=1)) + 1e-6)
        attn: torch.Tensor = torch.einsum("bshd,bhmd,bsh->bshm", self._kernel_fn(q), kv, z)

        attn = attn.reshape(batches, sequence_len, self.head_dim * self.num_heads)
        attn = self.attn_dropout(attn)
        proj = self.projection(attn)
        proj = self.proj_dropout(proj)

        return attn.contiguous()


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 8000,
        freq: float = 16000.0,
    ):
        super(PositionalEncoding, self).__init__()
        pos_e = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(freq) / embed_dim))
        pos_e[:, 0::2] = torch.sin(position * div)
        pos_e[:, 1::2] = torch.cos(position * div)
        pos_e = pos_e.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_e", pos_e)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs + self.pos_e[: inputs.size(0), :]  # type: ignore
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.05):
        super(FFN, self).__init__()
        self.embed_dim: int = embed_dim
        self.expansion: int = expansion
        self.fc1: Callable[..., torch.Tensor] = nn.Linear(embed_dim, embed_dim * expansion)
        self.squared_relu = SquaredReLU()
        self.fc2: Callable[..., torch.Tensor] = nn.Linear(embed_dim * expansion, embed_dim)
        self.dropout: Callable[..., torch.Tensor] = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(inputs)
        x = self.squared_relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LinearTransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.05,
        proj_dropout: float = 0.05,
    ):
        super(LinearTransformerEncoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mhla = MultiheadedLinearSelfAttention(embed_dim, num_heads, attn_dropout, proj_dropout)
        self.ffn = FFN(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.mhla(x)
        internal = x + inputs
        x = self.norm(internal)
        x = self.ffn(x)
        out = x + internal
        return out


class LinearTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.05,
        proj_dropout: float = 0.05,
    ):
        super(LinearTransformerEncoder, self).__init__()
        self.layer_stack = self._make_encoder_block_stack(
            num_layers,
            embed_dim,
            num_heads,
            attn_dropout,
            proj_dropout,
        )

    def _make_encoder_block_stack(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.05,
        proj_dropout: float = 0.05,
    ):
        layer_stack: List[LinearTransformerEncoderBlock] = []

        for _ in range(num_layers):
            layer_stack.append(LinearTransformerEncoderBlock(embed_dim, num_heads, attn_dropout, proj_dropout))

        return nn.Sequential(*layer_stack)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer_stack(inputs)
        return x
