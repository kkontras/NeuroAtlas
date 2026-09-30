from __future__ import annotations

import math

import torch
import torch.nn as nn


CHANNEL_DICT = {k.upper(): v for v, k in enumerate(
    [
        "FP1", "FPZ", "FP2",
        "AF7", "AF3", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
        "O1", "OZ", "O2",
    ]
)}


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


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        stride = patch_size if patch_stride is None else patch_stride
        if patch_stride is None:
            self.num_patches = (img_size[0], img_size[1] // patch_size)
        else:
            self.num_patches = (img_size[0], ((img_size[1] - patch_size) // patch_stride) + 1)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size), stride=(1, stride))

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.proj(x).transpose(1, 3)


class EEGTransformer(nn.Module):
    def __init__(
        self,
        img_size=(64, 1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=512,
        embed_num=4,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
    ):
        super().__init__()
        self.embed_num = embed_num
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        self.init_std = init_std
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def prepare_chan_ids(self, channels):
        channel_ids = []
        for channel in channels:
            normalized = channel.upper().strip(".")
            if normalized not in CHANNEL_DICT:
                raise KeyError(f"Unsupported EEGPT channel name {channel!r}.")
            channel_ids.append(CHANNEL_DICT[normalized])
        return torch.tensor(channel_ids).unsqueeze(0).long()

    def _fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None):
        x = self.patch_embed(x)
        batch_size, num_patches, num_channels, _ = x.shape
        if chan_ids is None:
            chan_ids = torch.arange(0, num_channels, device=x.device).unsqueeze(0)
        chan_ids = chan_ids.to(x.device)
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)
        x = x.flatten(0, 1)
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)
        for block in self.blocks:
            x = block(x)
        x = x[:, -summary_token.shape[1]:, :]
        x = self.norm(x)
        x = x.flatten(-2)
        x = x.reshape((batch_size, num_patches, -1))
        return x.reshape((batch_size, num_patches, self.embed_num, -1))
