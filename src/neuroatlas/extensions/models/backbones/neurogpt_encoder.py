"""Self-contained NeuroGPT EEG Conformer encoder.

Extracted from https://github.com/wenhui0206/NeuroGPT (Cui et al., IEEE ISBI 2024).
The encoder is the convolutional + transformer portion of the NeuroGPT pipeline
that processes raw EEG chunks into 1080-dimensional embeddings.

Architecture:
    PatchEmbedding (temporal conv + spatial conv + pool) -> TransformerEncoder (6 layers)

Input:  (B, n_chunks, 22, 500)  -- 22 channels, 500 samples (2s at 250 Hz)
Output: (B*n_chunks, 27, 40)    -- 27 temporal tokens, 40 features each
        Flattened: (B*n_chunks, 1080)

Original code: BSD-3 license, Song et al. (EEG Conformer).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor

try:
    from einops.layers.torch import Rearrange
except ImportError:
    raise ImportError("einops is required for NeuroGPT encoder: pip install einops")

try:
    from einops import rearrange
except ImportError:
    raise ImportError("einops is required for NeuroGPT encoder: pip install einops")

NEUROGPT_CHANNELS: tuple[str, ...] = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T1", "T3", "C3", "CZ", "C4", "T4", "T2",
    "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
)

N_CHANNELS = 22
CHUNK_SIZE = 500
N_FILTERS_TIME = 40
FILTER_TIME_LENGTH = 25
POOL_TIME_LENGTH = 75
POOL_TIME_STRIDE = 15
ATT_DEPTH = 6
ATT_HEADS = 10
ATT_DROP = 0.5
DROP_PROB = 0.5
FORWARD_EXPANSION = 4

# Output token count: ((500 - 25 + 1 - 75) // 15 + 1) = 27
N_TEMPORAL_TOKENS = (
    (CHUNK_SIZE - FILTER_TIME_LENGTH + 1 - POOL_TIME_LENGTH) // POOL_TIME_STRIDE + 1
)
EMBEDDING_DIM = N_TEMPORAL_TOKENS * N_FILTERS_TIME  # 27 * 40 = 1080


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        n_filters_time: int = N_FILTERS_TIME,
        filter_time_length: int = FILTER_TIME_LENGTH,
        n_channels: int = N_CHANNELS,
        pool_time_length: int = POOL_TIME_LENGTH,
        stride_avg_pool: int = POOL_TIME_STRIDE,
        drop_prob: float = DROP_PROB,
    ):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool),
            ),
            nn.Dropout(p=drop_prob),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        att_heads: int,
        att_drop: float,
        forward_expansion: int = FORWARD_EXPANSION,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=att_drop
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        att_depth: int,
        emb_size: int,
        att_heads: int,
        att_drop: float,
    ):
        super().__init__(
            *[
                _TransformerEncoderBlock(emb_size, att_heads, att_drop)
                for _ in range(att_depth)
            ]
        )


class EEGConformerEncoder(nn.Module):
    """NeuroGPT's EEG Conformer encoder (encoder-only, no classification head).

    Parameters
    ----------
    n_chans : int
        Number of EEG channels (must be 22 for pretrained weights).
    n_filters_time : int
        Number of temporal convolution filters (= embedding dimension per token).
    filter_time_length : int
        Temporal convolution kernel width.
    pool_time_length : int
        Average pooling kernel width.
    pool_time_stride : int
        Average pooling stride.
    drop_prob : float
        Dropout probability for conv layers.
    att_depth : int
        Number of transformer encoder blocks.
    att_heads : int
        Number of attention heads.
    att_drop_prob : float
        Dropout probability for attention layers.
    """

    def __init__(
        self,
        n_chans: int = N_CHANNELS,
        n_filters_time: int = N_FILTERS_TIME,
        filter_time_length: int = FILTER_TIME_LENGTH,
        pool_time_length: int = POOL_TIME_LENGTH,
        pool_time_stride: int = POOL_TIME_STRIDE,
        drop_prob: float = DROP_PROB,
        att_depth: int = ATT_DEPTH,
        att_heads: int = ATT_HEADS,
        att_drop_prob: float = ATT_DROP,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_filters_time = n_filters_time

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, n_chans, T)`` where T is the chunk length (500).

        Returns
        -------
        Tensor
            Shape ``(B, n_temporal_tokens, n_filters_time)`` = ``(B, 27, 40)``
            for the default chunk size of 500.
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.patch_embedding(x)  # (B, seq, d_model)
        x = self.transformer(x)  # (B, seq, d_model)
        return x
