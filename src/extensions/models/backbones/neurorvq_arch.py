"""Vendored NeuroRVQ foundation model architecture (encoder-only).

Source: https://huggingface.co/ntinosbarmpas/NeuroRVQ
Paper:  https://arxiv.org/abs/2510.13068

Only the components needed for frozen embedding extraction are included:
MultiDimentionalTemporalConv, transformer blocks, and the NeuroRVQFM forward
path with use_as_encoder=True.  Decoder, quantizer, and tokenizer classes are
omitted.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -- Channel list used during NeuroRVQ-EEG v1 pretraining (99 channels) ------
CH_NAMES_GLOBAL = np.array([
    b'a1', b'a2', b'af3', b'af4', b'af7', b'af8', b'afz', b'c1', b'c2',
    b'c3', b'c4', b'c5', b'c6', b'ccp1', b'ccp2', b'ccp3', b'ccp4',
    b'ccp5', b'ccp6', b'ccp7', b'ccp8', b'cfc1', b'cfc2', b'cfc3',
    b'cfc4', b'cfc5', b'cfc6', b'cfc7', b'cfc8', b'cp1', b'cp2',
    b'cp3', b'cp4', b'cp5', b'cp6', b'cpz', b'cz', b'eog', b'f1',
    b'f10', b'f2', b'f3', b'f4', b'f5', b'f6', b'f7', b'f8', b'f9',
    b'fc1', b'fc2', b'fc3', b'fc4', b'fc5', b'fc6', b'fcz', b'fp1',
    b'fp2', b'fpz', b'ft7', b'ft8', b'fz', b'iz', b'loc', b'o1', b'o2',
    b'oz', b'p08', b'p1', b'p10', b'p2', b'p3', b'p4', b'p5', b'p6',
    b'p7', b'p8', b'p9', b'po1', b'po10', b'po2', b'po3', b'po4',
    b'po7', b'po8', b'po9', b'poz', b'pz', b'roc', b'sp1', b'sp2',
    b't1', b't10', b't2', b't3', b't4', b't5', b't6', b't7', b't8',
    b't9', b'tp10', b'tp7', b'tp8', b'tp9',
])

N_GLOBAL_ELECTRODES = len(CH_NAMES_GLOBAL)  # 99


# -- Weight init helpers ------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# -- Transformer building blocks -----------------------------------------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None,
                 attn_drop=0.0, proj_drop=0.0, window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ))
        else:
            qkv_bias = None

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_norm=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# -- Multi-scale temporal convolution (Inception-style patch embedding) --------

class MultiDimentionalTemporalConv(nn.Module):
    """Inception-style multi-scale temporal filtering at 4 frequency bands.

    Assumes 200 Hz sampling rate.  Each branch targets a different cutoff:
    branch 1: >10 Hz (kernel 21), branch 2: >13 Hz (kernel 15),
    branch 3: >20 Hz (kernel 9),  branch 4: >40 Hz (kernel 5).

    Two groups of conv+norm+pool reduce the temporal dimension by 8x total
    (pool 2x then pool 4x), so a 200-sample patch becomes 25 time steps.
    With out_chans=8, the output per token is 25*8 = 200 = embed_dim.
    """

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # Group 1 — pool by 2
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 21), padding=(0, 10))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), padding=(0, 7))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm1_3 = nn.GroupNorm(4, out_chans)
        self.pool1_3 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_4 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_4 = nn.GroupNorm(4, out_chans)
        self.pool1_4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.gelu1 = nn.GELU()

        # Group 2 — pool by 4
        self.conv2_1 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 7), padding=(0, 3))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm2_3 = nn.GroupNorm(4, out_chans)
        self.pool2_3 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_4 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_4 = nn.GroupNorm(4, out_chans)
        self.pool2_4 = nn.AvgPool2d(kernel_size=(1, 4))
        self.gelu2 = nn.GELU()

    def forward(self, x):
        # x: (B, N_ch, n_time_patches, patch_size)
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)  # (B, 1, N*A, T)

        x1 = self.pool1_1(self.gelu1(self.norm1_1(self.conv1_1(x))))
        x2 = self.pool1_2(self.gelu1(self.norm1_2(self.conv1_2(x))))
        x3 = self.pool1_3(self.gelu1(self.norm1_3(self.conv1_3(x))))
        x4 = self.pool1_4(self.gelu1(self.norm1_4(self.conv1_4(x))))

        x1 = self.pool2_1(self.gelu2(self.norm2_1(self.conv2_1(x1))))
        x2 = self.pool2_2(self.gelu2(self.norm2_2(self.conv2_2(x2))))
        x3 = self.pool2_3(self.gelu2(self.norm2_3(self.conv2_3(x3))))
        x4 = self.pool2_4(self.gelu2(self.norm2_4(self.conv2_4(x4))))

        x1 = rearrange(x1, 'B C NA T -> B NA (T C)')
        x2 = rearrange(x2, 'B C NA T -> B NA (T C)')
        x3 = rearrange(x3, 'B C NA T -> B NA (T C)')
        x4 = rearrange(x4, 'B C NA T -> B NA (T C)')
        return x1, x2, x3, x4


# -- NeuroRVQ Foundation Model (encoder path only) ----------------------------

class NeuroRVQFM(nn.Module):
    """NeuroRVQ Foundation Model — encoder-only forward for embedding extraction.

    Architecture:
    1. MultiDimentionalTemporalConv produces 4 multi-scale feature branches.
    2. Each branch gets CLS token + learned spatial/temporal position embeddings.
    3. Shared transformer blocks process each branch independently.
    4. Per-branch LayerNorm produces the final token features.
    """

    def __init__(
        self,
        n_patches: int = 256,
        patch_size: int = 200,
        in_chans: int = 1,
        out_chans: int = 8,
        num_classes: int = 0,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values=None,
        init_scale: float = 0.001,
        n_global_electrodes: int = N_GLOBAL_ELECTRODES,
        vocab_size: int = 8192,
        use_as_encoder: bool = True,
        use_for_pretraining: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_as_encoder = use_as_encoder

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = MultiDimentionalTemporalConv(out_chans=out_chans)

        self.pos_embed = nn.Parameter(
            torch.zeros(n_global_electrodes + 1, embed_dim), requires_grad=True,
        )
        self.time_embed = nn.Parameter(
            torch.zeros(n_patches, embed_dim), requires_grad=True,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=nn.LayerNorm, init_values=init_values,
            )
            for i in range(depth)
        ])

        self.norm = nn.Identity()
        self.fc_norm_1 = nn.LayerNorm(embed_dim)
        self.fc_norm_2 = nn.LayerNorm(embed_dim)
        self.fc_norm_3 = nn.LayerNorm(embed_dim)
        self.fc_norm_4 = nn.LayerNorm(embed_dim)
        self.head_1 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_2 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_3 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_4 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        for head in (self.head_1, self.head_2, self.head_3, self.head_4):
            if isinstance(head, nn.Linear):
                trunc_normal_(head.weight, std=0.02)
                head.weight.data.mul_(init_scale)
                head.bias.data.mul_(init_scale)

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def forward_encoder(self, x, temporal_embedding_ix, spatial_embedding_ix):
        """Encoder forward that returns per-branch LayerNorm'd features.

        Parameters
        ----------
        x : Tensor (B, N_ch, n_time_patches, patch_size)
        temporal_embedding_ix : LongTensor (1, n_ch * n_time)
        spatial_embedding_ix  : LongTensor (1, n_ch * n_time)

        Returns
        -------
        tuple of 4 Tensors, each (B, n_ch * n_time, embed_dim)
        """
        x1, x2, x3, x4 = self.patch_embed(x)
        batch_size = x1.size(0)
        seq_len = x1.size(1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x3 = torch.cat((cls_tokens, x3), dim=1)
        x4 = torch.cat((cls_tokens, x4), dim=1)

        # Spatial position embeddings (pad +1 for CLS at index 0)
        spat_ix = F.pad(spatial_embedding_ix, (1, 0), value=0)  # (B_or_1, seq+1)
        spat_emb = self.pos_embed[spat_ix.reshape(-1), :]
        spat_emb = spat_emb.reshape(spat_ix.shape[0], spat_ix.shape[1], -1)

        x1 = x1 + spat_emb
        x2 = x2 + spat_emb
        x3 = x3 + spat_emb
        x4 = x4 + spat_emb

        # Temporal position embeddings (non-CLS tokens only)
        temp_emb = self.time_embed[temporal_embedding_ix.reshape(-1), :]
        temp_emb = temp_emb.reshape(
            temporal_embedding_ix.shape[0], temporal_embedding_ix.shape[1], -1,
        )

        x1[:, 1:, :] += temp_emb
        x2[:, 1:, :] += temp_emb
        x3[:, 1:, :] += temp_emb
        x4[:, 1:, :] += temp_emb

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x3 = self.pos_drop(x3)
        x4 = self.pos_drop(x4)

        branches = [x1, x2, x3, x4]
        fc_norms = [self.fc_norm_1, self.fc_norm_2, self.fc_norm_3, self.fc_norm_4]
        heads = [self.head_1, self.head_2, self.head_3, self.head_4]
        results = []
        for xi, fc, head in zip(branches, fc_norms, heads):
            for blk in self.blocks:
                xi = blk(xi)
            xi = self.norm(xi)
            xi = xi[:, 1:, :]  # drop CLS token
            xi = head(fc(xi))
            results.append(xi)

        return results[0], results[1], results[2], results[3]
