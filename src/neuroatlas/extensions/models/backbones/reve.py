from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .base import BenchmarkBackbone
from ._preproc import (
    _PAPER_PREPROC_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    is_bci_batch,
    resample_poly_with_fallback,
    run_names,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from neuroatlas import quiet
from neuroatlas.benchmarking_helpers import CheckpointSpec
from neuroatlas._paths import models_dir

logger = logging.getLogger(__name__)

_SMOKE = bool(os.environ.get("REVE_SMOKE_LOG"))

_TARGET_SFREQ = 200.0
_CLIP_SIGMA = 15.0


def _stats_shape_matches(mean, std, expected_c: int) -> bool:
    """Verify per-channel recording stats line up with the tensor's channel axis.

    ``recording_mean`` / ``recording_std`` land in ``meta[i]`` as tensors,
    numpy arrays, or plain lists of length ``C``. Reject anything that
    doesn't flatten to exactly ``expected_c`` elements so the reshape
    below cannot silently broadcast.
    """
    def _len(v) -> int | None:
        if hasattr(v, "numel"):
            return int(v.numel())
        if hasattr(v, "size") and not callable(v.size):
            try:
                return int(np.asarray(v).size)
            except Exception:
                return None
        try:
            return len(v)
        except TypeError:
            return None

    lm, ls = _len(mean), _len(std)
    return lm == expected_c and ls == expected_c

# Bipolar channels are positioned at the midpoint of their two electrodes,
# matching REVE's own downstream eval code:
# elouayas/reve_eeg src/downstream_tasks/position_utils.py — names containing
# a '-' are split, both halves are looked up in the position bank, and the
# resulting 3D coordinates are averaged. This is the convention REVE used to
# evaluate on bipolar corpora (e.g. TUEV double-banana).

# Where `models download reve_pretrained` / `reve_random_init` put the files
_LOCAL_MODEL_DIR = models_dir("foundation", "reve")   # the position bank sits next to it


class REVEBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from transformers import AutoConfig, AutoModel

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = int(round(self.epoch_seconds * self.target_sfreq))

        # The files come from the models root, where `models download` puts
        # them, or from a complete snapshot in the Hugging Face cache (a job
        # that once ran online leaves one; transformers reads it offline too).
        # Online, a missing piece is fetched from the hub, as for every
        # hub-loaded model; offline, say which command fetches it rather than
        # let transformers fail with "couldn't connect to huggingface.co".
        from ._checkpoint_download import REVE_REPO, downloads_off, reve_sources

        random_init = getattr(spec, "source_type", "") == "random_init"
        local_dir = Path(spec.checkpoint_path or str(_LOCAL_MODEL_DIR))
        sources = reve_sources(local_dir, random_init)
        if sources.lacking and (downloads_off() or os.environ.get("HF_HUB_OFFLINE") == "1"):
            raise FileNotFoundError(
                f"REVE needs {', '.join(sources.lacking)} (downloads are off)\n"
                f"fix: neuroatlas models download {spec.identifier}, or add --online"
            )

        # CPU init for the same reason as the trunk below.
        pos_source = sources.positions or "brain-bzh/reve-positions"
        with torch.device("cpu"):
            self.pos_bank = AutoModel.from_pretrained(pos_source, trust_remote_code=True)

        # random_init reads the config from the same folder but loads no weights.
        model_source = sources.model or REVE_REPO
        # Force model construction on CPU regardless of from_config vs
        # from_pretrained. Some CUDA kernels (RTX 5060 Ti / Blackwell,
        # observed 2026-05-04) segfault during the REVE custom modeling
        # __init__ if any factory function lands on GPU; staging on CPU and
        # moving via .to(self.device) below avoids the bug for both paths.
        if getattr(spec, "source_type", "") == "random_init":
            config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
            with torch.device("cpu"):
                self.model = AutoModel.from_config(config, trust_remote_code=True)
            self._weight_source = "random_init"
        else:
            with torch.device("cpu"):
                self.model = AutoModel.from_pretrained(model_source, trust_remote_code=True)
            self._weight_source = "pretrained"

        self.model.eval()
        self.model.to(self.device)
        self.pos_bank.eval()
        self.pos_bank.to(self.device)
        self._first_batch_logged = False
        self._warned_missing_unit = False
        self._warned_missing_stats = False
        self._warned_norm_disabled = False
        self._last_norm_source: str = "uninitialized"
        self._last_unit_declared: str | None = None

        if _SMOKE:
            self._log_load_report(model_source, pos_source)

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_recording_normalization": bool(
                ov.get("apply_recording_normalization", True)
            ),
        }

    def _log_load_report(self, model_source, pos_source):
        try:
            from safetensors.torch import load_file
            ckpt = load_file(str(Path(model_source) / "model.safetensors"))
            ck_keys = set(ckpt.keys())
            sd_keys = set(self.model.state_dict().keys())
            n_params = sum(p.numel() for p in self.model.parameters())
            missing = sorted(sd_keys - ck_keys)
            unexpected = sorted(ck_keys - sd_keys)
            shape_mismatch = []
            for k in (sd_keys & ck_keys):
                a, b = self.model.state_dict()[k].shape, ckpt[k].shape
                if a != b:
                    shape_mismatch.append(f"{k}: model{tuple(a)} vs ckpt{tuple(b)}")
            print(f"[REVE_SMOKE] model_source={model_source} pos_source={pos_source}")
            print(f"[REVE_SMOKE] params={n_params:,} ckpt_keys={len(ck_keys)} model_keys={len(sd_keys)}")
            print(f"[REVE_SMOKE] missing_in_ckpt={len(missing)} unexpected_in_ckpt={len(unexpected)} shape_mismatch={len(shape_mismatch)}")
            if missing:
                print(f"[REVE_SMOKE] missing_keys_sample={missing[:5]}")
            if unexpected:
                print(f"[REVE_SMOKE] unexpected_keys_sample={unexpected[:5]}")
            if shape_mismatch:
                print(f"[REVE_SMOKE] shape_mismatch_sample={shape_mismatch[:5]}")
            # Position-bank coverage
            pos_names = list(self.pos_bank.position_names)
            pos_emb = self.pos_bank.embedding
            print(f"[REVE_SMOKE] pos_bank_names={len(pos_names)} pos_emb_shape={tuple(pos_emb.shape)} "
                  f"pos_emb_min={float(pos_emb.min()):.4f} pos_emb_max={float(pos_emb.max()):.4f}")
        except Exception as e:
            print(f"[REVE_SMOKE] load_report_error: {type(e).__name__}: {e}")

    def _channel_names(self, batch) -> list:
        """Pass-through channel names from batch metadata; bipolar pairs
        ('FP1-F3') stay as-is and are resolved to midpoint XYZ in
        ``_resolve_positions``."""
        meta = batch.get("meta", [{}])
        raw_channels = meta[0].get("channels", None) if meta else None
        if raw_channels is None:
            return ["Fpz"]
        return list(raw_channels)

    def _resolve_positions(self, ch_names) -> torch.Tensor:
        """Return (C, 3) XYZ for the given channel names. Bipolar pairs
        (name contains '-') are split and the two electrodes' positions are
        averaged — matches REVE's downstream eval (position_utils.py)."""
        flat, is_bipolar = [], []
        for n in ch_names:
            if "-" in n:
                flat.extend(n.split("-"))
                is_bipolar.append(True)
            else:
                flat.append(n)
                is_bipolar.append(False)
        # pos_bank silently drops names not in its mapping. Catch that here so
        # the model never gets a positions tensor whose channel count differs
        # from the EEG tensor's.
        known = set(self.pos_bank.position_names)
        missing = [n for n in flat if n not in known]
        if missing:
            raise ValueError(
                f"REVE: {len(missing)} channel name(s) not in position bank "
                f"(first 5: {missing[:5]}); upstream channels = {ch_names}"
            )
        all_pos = self.pos_bank(flat)  # (M, 3)
        out, ptr = [], 0
        for bp in is_bipolar:
            if bp:
                out.append((all_pos[ptr] + all_pos[ptr + 1]) / 2.0)
                ptr += 2
            else:
                out.append(all_pos[ptr])
                ptr += 1
        return torch.stack(out)

    def _log_first_batch(self, batch, x_raw, x_resampled, ch_names):
        if self._first_batch_logged or not _SMOKE:
            return
        self._first_batch_logged = True
        meta = (batch.get("meta") or [{}])[0]
        raw_channels = meta.get("channels")
        sfreq = meta.get("sfreq")
        # Position bank coverage — bipolar pairs need both halves in the bank.
        known = set(self.pos_bank.position_names)
        flat_required = []
        for n in ch_names:
            flat_required.extend(n.split("-")) if "-" in n else flat_required.append(n)
        flat_known = [c for c in flat_required if c in known]
        flat_missing = [c for c in flat_required if c not in known]
        n_bipolar = sum(1 for n in ch_names if "-" in n)
        # Scale stats from raw input (μV check)
        with torch.no_grad():
            xr = x_raw.float()
            mn, mx = float(xr.min()), float(xr.max())
            mean, std = float(xr.mean()), float(xr.std())
            absmean = float(xr.abs().mean())
        print(f"[REVE_SMOKE] raw_input_shape={tuple(x_raw.shape)} dtype={x_raw.dtype} sfreq={sfreq}")
        print(f"[REVE_SMOKE] raw_stats min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g} abs_mean={absmean:.4g}")
        # Heuristic μV vs V check
        if absmean < 1e-3:
            scale_guess = "VOLTS_LIKELY"
        elif absmean < 5.0:
            scale_guess = "MILLIVOLTS_LIKELY"
        elif absmean < 5e3:
            scale_guess = "MICROVOLTS_LIKELY"
        else:
            scale_guess = "UNKNOWN_LARGE"
        print(f"[REVE_SMOKE] scale_guess={scale_guess} (REVE pretrain expects μV)")
        print(f"[REVE_SMOKE] resampled_shape={tuple(x_resampled.shape)} target_len={self.target_len}")
        print(f"[REVE_SMOKE] raw_channels(n={len(raw_channels) if raw_channels else 'None'})={raw_channels}")
        print(f"[REVE_SMOKE] passthrough_channels(n={len(ch_names)}) bipolar_pairs={n_bipolar}={ch_names}")
        print(f"[REVE_SMOKE] required_electrodes(n={len(flat_required)}) in_bank={len(flat_known)} missing={flat_missing}")
        if flat_missing:
            print(f"[REVE_SMOKE] *** UNRESOLVABLE CHANNEL NAMES — _resolve_positions will raise ***")

    def _prepare_input(self, batch):
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            full = signals.get("full_signal")
            if full is None:
                raise KeyError("REVE backbone expects batch['signals']['eeg'] or ['full_signal'].")
            x = full[:, :1]
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(
            meta,
            keys=("sampling_rate", "unit", "channels"),
            where="reve:input",
        )

        # §0 — declared-unit → µV. Fallback to "uV" with a one-shot
        # DeprecationWarning for adapters that predate the contract.
        unit = meta[0].get("unit") if meta else None
        if unit is None:
            if not self._warned_missing_unit:
                self._warned_missing_unit = True
                warnings.warn(
                    "REVE: batch meta missing 'unit' — defaulting to 'uV'. "
                    "Adapters must publish meta[i]['unit'] per "
                    "MODEL_CONTRACTS.md §0.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            unit = "uV"
        self._last_unit_declared = unit
        x = unit_to_uv(x, unit)

        overrides = self._resolve_overrides()

        dataset = meta[0].get("dataset") if meta else None
        bci = is_bci_batch(meta)

        if dataset in _PAPER_PREPROC_DATASETS:
            x = x - x.mean(dim=-1, keepdim=True)

        # Adaptive epoch_seconds: _PAPER_PREPROC_DATASETS take 30 s from the batch meta
        if dataset in _PAPER_PREPROC_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds
        _target_len = int(round(epoch_sec * self.target_sfreq))

        T = x.shape[-1]
        meta_sfreq = meta[0].get("sampling_rate") if meta else None
        if meta_sfreq is None:
            meta_sfreq = meta[0].get("sfreq") if meta else None
        if meta_sfreq is not None:
            if not bci:
                expected_T = int(round(float(meta_sfreq) * epoch_sec))
                if abs(T - expected_T) > 1:
                    raise ValueError(
                        f"REVE: duration mismatch — meta['sampling_rate']={meta_sfreq} Hz and "
                        f"epoch_seconds={epoch_sec:g} s imply {expected_T} samples, "
                        f"but got T={T}. Upstream preprocessor contract is broken."
                    )
            src_sfreq_f = float(meta_sfreq)
        else:
            src_sfreq_f = T / epoch_sec

        _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
        x, _ = resample_poly_with_fallback(x, src_sfreq_f, self.target_sfreq, backend=_backend)
        if dataset in _PAPER_PREPROC_DATASETS:
            x = snap_to_epoch_length(x, float(self.target_sfreq), meta)
        if bci:
            # BCI parity: keep whole 200-sample (1 s) patches.
            actual_len = x.shape[-1]
            _REVE_PATCH = 200
            if actual_len % _REVE_PATCH != 0:
                trim = actual_len - (actual_len % _REVE_PATCH)
                if trim > 0:
                    x = x[..., :trim]
                else:
                    raise ValueError(
                        f"REVE: BCI resampled length {actual_len} too short for "
                        f"even one {_REVE_PATCH}-sample patch."
                    )
            self.target_len = x.shape[-1]
        elif x.shape[-1] != _target_len:
            raise ValueError(
                f"REVE: window length mismatch — expected {_target_len} samples "
                f"({epoch_sec:g} s × {self.target_sfreq:g} Hz), got {x.shape[-1]}. "
                f"Input was {T} samples at fs {src_sfreq_f:.3f} Hz. "
                f"REVE accepts any epoch_seconds but does not silently stretch."
            )

        # §2 — recording-level z-score + ±15 σ clip. REVE is defined by
        # this normalization; without it the input is off-distribution.
        if overrides["apply_recording_normalization"]:
            # PATCH (parity-2026-05-14): per-window recording stats. Each row
            # uses its own meta[i] instead of broadcasting meta[0] over the
            # whole batch (cross-recording batches were getting recording-0's
            # mean/std applied to all 64 rows).
            B_in = x.shape[0]
            per_row_means = []
            per_row_stds = []
            all_ok = bool(meta) and len(meta) >= B_in
            stats_missing = not all_ok
            if all_ok:
                for i in range(B_in):
                    m_i = meta[i] if i < len(meta) else {}
                    rm = m_i.get("recording_mean")
                    rs = m_i.get("recording_std")
                    if rm is None or rs is None or not _stats_shape_matches(rm, rs, x.shape[1]):
                        all_ok = False
                        stats_missing = rm is None or rs is None
                        break
                    per_row_means.append(rm)
                    per_row_stds.append(rs)
            if all_ok:
                mu = torch.stack([
                    torch.as_tensor(rm, dtype=x.dtype, device=x.device).reshape(x.shape[1])
                    for rm in per_row_means
                ]).reshape(B_in, x.shape[1], 1)
                sigma = torch.stack([
                    torch.as_tensor(rs, dtype=x.dtype, device=x.device).reshape(x.shape[1])
                    for rs in per_row_stds
                ]).reshape(B_in, x.shape[1], 1)
                mu = unit_to_uv(mu, unit)
                sigma = unit_to_uv(sigma, unit)
                x = (x - mu) / sigma.clamp(min=1e-6)
                norm_source = "recording_stats_per_window"
            else:
                # Each window z-scored per channel on its own. BCI trials never
                # carry recording statistics (the BCI readers do not compute
                # them): the expected path there, said at DEBUG. Elsewhere the
                # dataset gives none (or none that fit its channels): a
                # warning once per command, dataset and checkpoint.
                self._warn_window_zscore(meta, bci, stats_missing)
                mu = x.mean(dim=-1, keepdim=True)
                sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
                x = (x - mu) / sigma
                norm_source = "per_window_fallback"
            # ±15 σ clip in σ-space (§2 line 423).
            x = x.clamp(-_CLIP_SIGMA, _CLIP_SIGMA)
        else:
            if not self._warned_norm_disabled:
                self._warned_norm_disabled = True
                logger.warning(
                    "REVE: apply_recording_normalization=False is an "
                    "off-spec ablation — embeddings will not match the "
                    "paper's scale-invariant domain (MODEL_CONTRACTS §8)."
                )
            # Keep a lightweight DC removal so a non-zero bias does not
            # dominate the coordinate-embedded attention path.
            x = x - x.mean(dim=-1, keepdim=True)
            norm_source = "dc_only"

        self._last_norm_source = norm_source
        assert_finite(x, where="reve:after_norm")
        return x

    def _warn_window_zscore(self, meta, bci: bool, stats_missing: bool) -> None:
        checkpoint = getattr(getattr(self, "spec", None), "identifier", "") or ""
        dataset, checkpoint = run_names(meta, checkpoint)
        if bci:
            logger.debug("REVE: %s trials carry no per-recording statistics; each trial is "
                         "z-scored per channel on its own", dataset or "BCI")
            return
        what = ("no per-recording statistics" if stats_missing
                else "per-recording statistics that do not match its channels")
        quiet.warn_once(
            logger, f"reve window z-score:{dataset}:{checkpoint}",
            "%s gives %s %s, so each window is z-scored per channel on its own",
            dataset or "this dataset", checkpoint or "REVE", what)

    def _filter_bci_channels(self, x, ch_names, batch):
        """For BCI: drop channels not in the position bank, select from tensor."""
        if not is_bci_batch(batch.get("meta")):
            return x, ch_names
        known = set(self.pos_bank.position_names)
        def _ch_known(n):
            if "-" in n:
                return all(p in known for p in n.split("-"))
            return n in known
        keep_idx = [i for i, n in enumerate(ch_names) if _ch_known(n)]
        if len(keep_idx) == len(ch_names):
            return x, ch_names
        dropped = [n for i, n in enumerate(ch_names) if i not in set(keep_idx)]
        logger.info(
            "REVE BCI: dropped %d channel(s) not in position bank: %s",
            len(dropped), dropped[:10],
        )
        ch_names = [ch_names[i] for i in keep_idx]
        x = x[:, keep_idx, :]
        return x, ch_names

    def extract_embeddings(self, batch) -> np.ndarray:
        # Capture raw input for smoke logging before mean-removal/resampling
        x_raw = None
        if _SMOKE and not self._first_batch_logged:
            sig = batch.get("signals", {}).get("eeg")
            if sig is None:
                sig = batch.get("signals", {}).get("full_signal")
                if sig is not None:
                    sig = sig[:, :1]
            x_raw = sig.detach().clone() if sig is not None else None
        x = self._prepare_input(batch)
        ch_names = self._channel_names(batch)

        # _PAPER_PREPROC_DATASETS: strip all-zero channels before position resolution
        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _PAPER_PREPROC_DATASETS and ch_names:
            x, ch_names, _ = strip_zero_channels(x, ch_names)

        x, ch_names = self._filter_bci_channels(x, ch_names, batch)
        if _SMOKE and x_raw is not None:
            self._log_first_batch(batch, x_raw, x, ch_names)
            with torch.no_grad():
                print(
                    f"[REVE_SMOKE] post_norm mean={float(x.mean()):.4g} "
                    f"std={float(x.std()):.4g} min={float(x.min()):.4g} "
                    f"max={float(x.max()):.4g} source={self._last_norm_source} "
                    f"unit_declared={self._last_unit_declared!r}"
                )
        positions = self._resolve_positions(ch_names).unsqueeze(0).expand(x.shape[0], -1, -1)
        with torch.inference_mode():
            out = self.model(x, positions)
        if _SMOKE and not getattr(self, "_first_out_logged", False):
            self._first_out_logged = True
            print(f"[REVE_SMOKE] forward_out_shape={tuple(out.shape)} positions_shape={tuple(positions.shape)}")
        # out: (B, C, num_patches, 512) → pool to (B, 512)
        return out.mean(dim=2).mean(dim=1).detach().cpu().numpy()

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        x = self._prepare_input(batch)
        ch_names = self._channel_names(batch)
        x, ch_names = self._filter_bci_channels(x, ch_names, batch)
        positions = self._resolve_positions(ch_names).unsqueeze(0).expand(x.shape[0], -1, -1)
        with torch.inference_mode():
            out = self.model(x, positions)
        # out: (B, C, num_patches, 512) -> (B, C*num_patches, 512)
        B, C, num_patches, D = out.shape
        tokens = out.reshape(B, C * num_patches, D)
        return tokens.detach().cpu().numpy()

    def forward_features(self, batch) -> torch.Tensor:
        """Same trunk as ``extract_embeddings`` but grad-enabled, returns a tensor.

        Used by finetune entrypoints. Shape: ``(B, embedding_dim)`` on
        ``self.device``. No ``inference_mode`` so backward pass works.
        """
        x = self._prepare_input(batch)
        ch_names = self._channel_names(batch)
        x, ch_names = self._filter_bci_channels(x, ch_names, batch)
        positions = self._resolve_positions(ch_names).unsqueeze(0).expand(x.shape[0], -1, -1)
        out = self.model(x, positions)
        return out.mean(dim=2).mean(dim=1)

    def metadata(self):
        overrides = self._resolve_overrides()
        md = {
            **super().metadata(),
            "device": self.device,
            "weight_source": getattr(self, "_weight_source", "pretrained"),
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
            "embedding_reduction": "mean_over_patches_mean_over_channels",
            "apply_recording_normalization": overrides["apply_recording_normalization"],
            "norm_source": self._last_norm_source,
            "unit_declared": self._last_unit_declared,
            "reference_applied": False,
        }
        if self._last_norm_source in ("recording_stats", "per_window_fallback"):
            md["clip_sigma"] = _CLIP_SIGMA
        return md
