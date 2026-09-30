"""Vendored ST-EEGFormer encoder architecture.

Adapted from LiuyinYang1101/STEEGFormer (ICLR 2026).
Original: benchmark/neural_networks/models/models_vit_eeg.py

Stripped: MAE decoder, masking, attention hooks, classification head.
Kept: PatchEmbedEEG, ChannelPositionalEmbed, TemporalPositionalEncoding,
      VisionTransformerEEG (encoder-only forward_features).

The model inherits from timm's VisionTransformer for Block/Attention reuse
but replaces patch_embed and positional encoding with EEG-specific modules.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import timm.models.vision_transformer


# ---------------------------------------------------------------------------
# Channel-index vocabulary (142 entries, indices 0-141).
# Extracted from the official pretrain/senloc_file/sen_chan_idx.pkl
# via data["channels_mapping"].  Order matters — the pretrained
# enc_channel_emd weights are keyed to these exact indices.
# ---------------------------------------------------------------------------
STEEGFORMER_CHANNEL_MAP: Dict[str, int] = {
    "C1": 0, "Pz": 1, "C4": 2, "F6": 3, "FTT8h": 4, "Oz": 5, "Fp1": 6,
    "FCC5h": 7, "TPP8h": 8, "CPP6h": 9, "C2": 10, "F4": 11, "OI2h": 12,
    "AF4": 13, "FCz": 14, "CCP6h": 15, "TP8": 16, "POO10h": 17, "FC1": 18,
    "FC6": 19, "C5": 20, "P8": 21, "FT8": 22, "P6": 23, "P9": 24, "Fz": 25,
    "AFF1": 26, "TPP10h": 27, "AFF2": 28, "P10": 29, "CPP2h": 30, "M1": 31,
    "FCC6h": 32, "FTT7h": 33, "FC2": 34, "PPO2": 35, "AFp3h": 36, "AF7": 37,
    "PO10": 38, "AF8": 39, "CPP1h": 40, "P7": 41, "F1": 42, "AFp4h": 43,
    "PO9": 44, "FT9": 45, "CP2": 46, "Iz": 47, "FCC1h": 48, "FC5": 49,
    "T5": 50, "CP5": 51, "CP6": 52, "FFC5h": 53, "F2": 54, "M2": 55,
    "POO9h": 56, "AFF5h": 57, "PO4": 58, "POO3h": 59, "Fp2": 60, "T3": 61,
    "CP4": 62, "POz": 63, "TTP7h": 64, "T7": 65, "A2": 66, "CCP4h": 67,
    "T8": 68, "PPO10h": 69, "FC3": 70, "F3": 71, "F5": 72, "A1": 73,
    "P3": 74, "FC4": 75, "FCC2h": 76, "FFC6h": 77, "FFT8h": 78, "CCP2h": 79,
    "CPP4h": 80, "T6": 81, "FTT9h": 82, "PPO6h": 83, "CP3": 84, "CP1": 85,
    "AF3": 86, "FT10": 87, "OI1h": 88, "TPP9h": 89, "P5": 90, "I2": 91,
    "CCP1h": 92, "T4": 93, "CCP3h": 94, "O1": 95, "PO5": 96, "PPO9h": 97,
    "PPO5h": 98, "P1": 99, "AFz": 100, "PO6": 101, "PO3": 102, "O2": 103,
    "CPP5h": 104, "FFC1h": 105, "FCC4h": 106, "FFT7h": 107, "FFC2h": 108,
    "FFC4h": 109, "Cz": 110, "TP7": 111, "Fpz": 112, "FTT10h": 113,
    "PO7": 114, "CPP3h": 115, "P4": 116, "P2": 117, "F8": 118, "CPz": 119,
    "FCC3h": 120, "FFC3h": 121, "FT7": 122, "I1": 123, "TTP8h": 124,
    "AFF6h": 125, "CCP5h": 126, "C6": 127, "PPO1": 128, "PO8": 129,
    "C3": 130, "POO4h": 131, "TPP7h": 132, "F7": 133, "T9": 134, "TP9": 135,
    "T10": 136, "TP10": 137, "POO1": 138, "POO2": 139, "PPO1h": 140,
    "PPO2h": 141,
}

_NUM_CHANNEL_POSITIONS = 145  # embedding table size (original pretrained model)

# Reverse lookup for diagnostics.
_INDEX_TO_CHANNEL = {v: k for k, v in STEEGFORMER_CHANNEL_MAP.items()}


STEEGFORMER_CONFIGS: Dict[str, Dict] = {
    "small": {"embed_dim": 512, "depth": 8, "num_heads": 8},
    "base":  {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}


# ---------------------------------------------------------------------------
# EEG-specific modules
# ---------------------------------------------------------------------------

class PatchEmbedEEG(nn.Module):
    """Non-overlapping 1-D patch embedding via nn.Unfold.

    Input:  (B, C, L) raw EEG at 128 Hz
    Output: (B, L//patch_size, C, embed_dim)  — (B, Seq, Ch, D)
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.p = patch_size
        self.embed_dim = embed_dim
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=patch_size)
        self.proj = nn.Linear(self.p, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, L = x.shape
        x = x.unsqueeze(2)  # (B, C, 1, L)
        unfolded = self.unfold(x)  # (B, C*patch_size, n_patches)
        seq = unfolded.shape[2]
        unfolded = unfolded.reshape(bs, c, self.p, seq)
        output = unfolded.permute(0, 3, 1, 2)  # (B, Seq, C, patch_size)
        return self.proj(output)  # (B, Seq, C, embed_dim)


class ChannelPositionalEmbed(nn.Module):
    """Learnable channel position embedding (up to 145 positions)."""

    def __init__(self, embedding_dim: int, num_positions: int = _NUM_CHANNEL_POSITIONS):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_positions, embedding_dim)
        init.zeros_(self.channel_transformation.weight)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        return self.channel_transformation(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for temporal patch indices."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float()
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def get_cls_token(self) -> torch.Tensor:
        return self.pe[0, 0, :]

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq_indices.shape
        return self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)


# ---------------------------------------------------------------------------
# Encoder-only VisionTransformer for EEG
# ---------------------------------------------------------------------------

class VisionTransformerEEG(timm.models.vision_transformer.VisionTransformer):
    """ST-EEGFormer encoder (no classification head, no decoder).

    Accepts ``(B, C, T)`` raw EEG + ``(C,)`` or ``(B, C)`` integer channel
    indices and returns ``(B, embed_dim)`` embeddings.

    Pooling modes:
        - ``global_pool=False``  (default): LayerNorm + CLS token (index 0).
        - ``global_pool=True``:  mean over non-CLS tokens.
    """

    def __init__(self, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        embed_dim = kwargs["embed_dim"]

        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            self.fc_norm = norm_layer(embed_dim)
            if hasattr(self, "norm"):
                del self.norm

        self.patch_embed = PatchEmbedEEG(
            patch_size=kwargs["patch_size"], embed_dim=embed_dim
        )
        self.enc_channel_emd = ChannelPositionalEmbed(embed_dim)
        self.enc_temporal_emd = TemporalPositionalEncoding(embed_dim, 512)

    def forward_features(
        self,
        eeg: torch.Tensor,
        chan_idx: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Encoder forward pass.

        Parameters
        ----------
        eeg : (B, C, T) float32 — z-scored EEG at 128 Hz.
        chan_idx : (C,) or (B, C) int — channel position indices.
        return_all_tokens : if True, return all tokens (B, 1+N, D)
            including CLS instead of the pooled (B, D) output.

        Returns
        -------
        (B, embed_dim) or (B, 1+N, embed_dim) tensor.
        """
        B = eeg.shape[0]
        x = self.patch_embed(eeg)  # (B, Seq, Ch, D)
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq * Ch_all
        x = x.view(B, Seq_total, Dmodel)

        if chan_idx.dim() == 1:
            chan_idx = chan_idx.unsqueeze(0).expand(B, -1)

        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1, Seq, 1).view(B, Seq_total)

        seq_tensor = torch.arange(1, Seq + 1, device=eeg.device)
        eeg_seq_indices = (
            seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(B, 1, Ch_all).view(B, Seq_total)
        )

        tp_embd = self.enc_temporal_emd(eeg_seq_indices)
        ch_embd = self.enc_channel_emd(eeg_chan_indices)
        x = x + tp_embd + ch_embd

        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            if return_all_tokens:
                return x[:, 1:, :]
            return x[:, 1:, :].mean(dim=1)
        else:
            x = self.norm(x)
            if return_all_tokens:
                return x
            return x[:, 0]

    def forward(self, eeg: torch.Tensor, chan_idx: torch.Tensor) -> torch.Tensor:
        return self.forward_features(eeg, chan_idx)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_steegformer(variant: str, **overrides) -> VisionTransformerEEG:
    """Construct a VisionTransformerEEG from a named variant."""
    cfg = STEEGFORMER_CONFIGS.get(variant)
    if cfg is None:
        raise ValueError(
            f"Unknown STEEGFormer variant {variant!r}; "
            f"expected one of {sorted(STEEGFORMER_CONFIGS)}."
        )
    kw = dict(
        patch_size=16,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        global_pool=False,
    )
    kw.update(overrides)
    return VisionTransformerEEG(**kw)
