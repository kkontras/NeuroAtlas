"""Vendored SleePyCo architecture.

Copied from https://github.com/gist-ailab/SleePyCo (Lee et al., ESWA 2024) —
``models/sleepyco.py`` (backbone), ``models/classifiers.py`` (Transformer
head), and the relevant ``models/main_model.py`` wiring, merged into one
self-contained file. Only structural code is vendored — training utilities,
DeepSleepNet/UTime/etc. siblings are dropped.

Inputs to ``MainModel.forward`` are 1-channel raw EEG sequences shaped
``(B, 1, T_total)`` where ``T_total = seq_len * 30 * fs``. The backbone is a
1D CNN with 5 stages and a 3-scale feature pyramid; the classifier is a
6-layer Transformer encoder with attention pooling and a 5-class FC head.

The official SHHS checkpoint distributed by the authors (``ckpt_fold-01.pth``)
contains 223 parameters keyed under ``module.feature.*`` and
``module.classifier.*`` — the wrapper strips the ``module.`` DataParallel
prefix before strict load.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone (models/sleepyco.py)
# ---------------------------------------------------------------------------


class MaxPool1d(nn.Module):
    """Same-pad MaxPool1d that pads when the time dim isn't divisible."""

    def __init__(self, maxpool_size: int) -> None:
        super().__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.size(-1)
        if n_samples % self.maxpool_size != 0:
            pad = self.maxpool_size - (n_samples % self.maxpool_size)
            left = pad // 2
            right = pad - left
            x = F.pad(x, (left, right), mode="constant")
        return self.maxpool(x)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
        att = self.mlp(avg_pool)
        scale = torch.sigmoid(att).unsqueeze(2).expand_as(x)
        return x * scale


class SleePyCoBackbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.training_mode = config["training_params"]["mode"]

        self.init_layer = self._make_layers(1, 64, n_layers=2, maxpool_size=None, first=True)
        self.layer1 = self._make_layers(64, 128, n_layers=2, maxpool_size=5)
        self.layer2 = self._make_layers(128, 192, n_layers=3, maxpool_size=5)
        self.layer3 = self._make_layers(192, 256, n_layers=3, maxpool_size=5)
        self.layer4 = self._make_layers(256, 256, n_layers=3, maxpool_size=5)

        if self.training_mode in ("freezefinetune", "scratch", "fullfinetune"):
            self.fp_dim = config["feature_pyramid"]["dim"]
            self.num_scales = config["feature_pyramid"]["num_scales"]
            self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
            if self.num_scales > 1:
                self.conv_c4 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
            if self.num_scales > 2:
                self.conv_c3 = nn.Conv1d(192, self.fp_dim, 1, 1, 0)

    @staticmethod
    def _make_layers(
        in_channels: int, out_channels: int, n_layers: int, maxpool_size, first: bool = False
    ) -> nn.Sequential:
        layers = [] if first else [MaxPool1d(maxpool_size)]
        for i in range(n_layers):
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                layers += [ChannelGate(in_channels)]
            layers += [nn.PReLU()]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = []
        if self.training_mode == "pretrain":
            out.append(c5)
        else:
            out.append(self.conv_c5(c5))
            if self.num_scales > 1:
                out.append(self.conv_c4(c4))
            if self.num_scales > 2:
                out.append(self.conv_c3(c3))
        return out


# ---------------------------------------------------------------------------
# Classifier (models/classifiers.py)
# ---------------------------------------------------------------------------


# (seq_len-1) -> (num_scales-1)  pre-computed feature lengths for SleePyCo,
# copied verbatim from the upstream `feature_len_dict["SleePyCo"]`.
FEATURE_LEN_TABLE = [
    [5, 24, 120],
    [10, 48, 240],
    [15, 72, 360],
    [20, 96, 480],
    [24, 120, 600],
    [29, 144, 720],
    [34, 168, 840],
    [39, 192, 960],
    [44, 216, 1080],
    [48, 240, 1200],
]


class PositionalEncoding(nn.Module):
    def __init__(self, config: dict, in_features: int, out_features: int) -> None:
        super().__init__()
        self.cfg = config["classifier"]["pos_enc"]
        self.num_scales = config["feature_pyramid"]["num_scales"]
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = nn.PReLU()

        if self.num_scales > 1:
            seq_len = config["dataset"]["seq_len"]
            self.max_len = FEATURE_LEN_TABLE[seq_len - 1][self.num_scales - 1]
        else:
            self.max_len = 5000

        if self.cfg.get("dropout"):
            self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(self.max_len, out_features)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, out_features, 2).float() * (-math.log(10000.0) / out_features))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.fc(x))
        if self.num_scales > 1:
            hop = self.max_len // x.size(0)
            pe = self.pe[hop // 2 :: hop, :]
        else:
            pe = self.pe
        if pe.shape[0] != x.size(0):
            pe = pe[: x.size(0), :]
        x = x + pe
        if self.cfg.get("dropout"):
            x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: dict, nheads: int = 8, num_encoder_layers: int = 6, pool: str = "mean") -> None:
        super().__init__()
        self.cfg = config["classifier"]
        self.model_dim = self.cfg["model_dim"]
        self.feedforward_dim = self.cfg["feedforward_dim"]
        self.in_features = config["feature_pyramid"]["dim"]
        self.out_features = self.cfg["model_dim"]

        self.pos_encoding = PositionalEncoding(config, self.in_features, self.out_features)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nheads,
            dim_feedforward=self.feedforward_dim,
            dropout=0.1 if self.cfg.get("dropout") else 0.0,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_encoder_layers)
        self.pool = pool

        if self.cfg.get("dropout"):
            self.dropout = nn.Dropout(p=0.5)

        if pool == "attn":
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)

        self.fc = nn.Linear(self.model_dim, self.cfg["num_classes"])

    def forward(self, x: torch.Tensor):
        # x: (B, L, in_features) -> transformer expects (L, B, in_features)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        feats = self.transformer(x)
        feats_bl = feats.transpose(0, 1)  # (B, L, model_dim)

        if self.pool == "mean":
            pooled = feats_bl.mean(dim=1)
        elif self.pool == "last":
            pooled = feats_bl[:, -1]
        elif self.pool == "attn":
            a_states = torch.tanh(self.w_ha(feats_bl))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(feats_bl.size(0), 1, feats_bl.size(1))
            pooled = torch.bmm(alpha, a_states).view(feats_bl.size(0), -1)
        else:
            raise NotImplementedError(f"pool={self.pool!r}")

        if self.cfg.get("dropout"):
            pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits, pooled


# ---------------------------------------------------------------------------
# MainModel wiring (models/main_model.py)
# ---------------------------------------------------------------------------


class MainModel(nn.Module):
    """SleePyCo backbone + Transformer head, restricted to the SHHS finetune recipe.

    Returns a dict keyed by feature-pyramid scale: ``{"p5": (logits, pooled), ...}``.
    The native head pools across all scales (mean of logits) — matching the
    upstream ``test.py`` aggregation.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = config
        self.feature = SleePyCoBackbone(config)
        self.classifier = Transformer(
            config, nheads=8, num_encoder_layers=6, pool=config["classifier"]["pool"]
        )

    def forward(self, x: torch.Tensor):
        features = self.feature(x)
        all_logits = []
        all_pooled = []
        for feat in features:
            # feat: (B, fp_dim, L) -> (B, L, fp_dim)
            logits, pooled = self.classifier(feat.transpose(1, 2))
            all_logits.append(logits)
            all_pooled.append(pooled)
        # Average logits across scales (the upstream test.py aggregator).
        logits = torch.stack(all_logits, dim=0).mean(dim=0)
        pooled = torch.stack(all_pooled, dim=0).mean(dim=0)
        return logits, pooled
