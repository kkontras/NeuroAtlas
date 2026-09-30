from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BenchmarkBackbone


class _FeatureExtractor(nn.Module):
    """Inlined BrainAgeCNN_v2 feature extractor (no brainage_eeg dependency).

    Architecture (from SleepAgeBench/models/brain_age_cnn_v2.py):
        [B, C, 3000]
        -> Conv1d(C->64, k=5, pad=2) -> ReLU -> MaxPool(2)    -> [B, 64, 1500]
        -> Conv1d(64->128, k=3, pad=1) -> ReLU -> MaxPool(2)   -> [B, 128, 750]
        -> Conv1d(128->128, k=3, pad=1) -> ReLU -> MaxPool(2)  -> [B, 128, 375]
        -> Conv1d(128->256, k=3, pad=1) -> ReLU -> AdaptiveAvgPool(1) -> [B, 256]
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class BrainAgeCNNBackbone(BenchmarkBackbone):
    def __init__(self, spec):
        super().__init__(spec)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _FeatureExtractor(in_channels=1)
        if spec.checkpoint_path is not None:
            state = torch.load(spec.checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "features" in state:
                self.model.load_state_dict(state["features"])
            elif isinstance(state, dict) and "state_dict" in state:
                prefix = "features."
                sub = {k[len(prefix):]: v for k, v in state["state_dict"].items() if k.startswith(prefix)}
                if sub:
                    self.model.load_state_dict(sub)
                else:
                    self.model.load_state_dict(state["state_dict"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract_embeddings(self, batch) -> np.ndarray:
        x = batch["signals"]["eeg"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        embeddings = self.model(x)  # [B, 256]
        return embeddings.cpu().numpy()

    def metadata(self):
        return {
            **super().metadata(),
            "device": str(self.device),
        }
