"""DeepSOZ-HEM, fetched at run time rather than shipped.

Upstream is https://github.com/amruth-sn/deepsoz-hem (GPL-3.0), pinned at commit
a7c13bdbb6d86e016f929ba108370cc614e9882c. Its model code
(``deepsoz-hem/src/deepsoz/baselines.py``) and fold-4 checkpoint
(``deepsoz-hem/src/deepsoz/deepsoz_fold4.pth_4.tar``) are GPL-3.0, so they are
not part of this MIT package: ``neuroatlas models download deepsoz_hem_pretrained``
fetches them, with upstream's LICENSE, into ``<models root>/foundation/deepsoz_hem/``,
each checked against the SHA-256 recorded in
``backbones/_checkpoint_download.GITHUB_COMMIT_FILES``. Until 2026-10-05 the
two files were vendored under ``backbones/third_party/deepsoz_hem/``; the
fetched files are byte-identical to those.
"""
from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec

#: The browsable tree at the pinned commit; the key of its entry in
#: _checkpoint_download.GITHUB_COMMIT_FILES (not imported here: that module's
#: package imports torch, and the registry is read without it).
UPSTREAM = "https://github.com/amruth-sn/deepsoz-hem/tree/a7c13bdbb6d86e016f929ba108370cc614e9882c"


def _load_deepsoz_hem(spec: CheckpointSpec):
    from .backbones.deepsoz_hem import DeepSOZHEMBackbone

    return DeepSOZHEMBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="deepsoz_hem",
        description=(
            "DeepSOZ-HEM (Shama et al. MICCAI 2023; SzCORE 2025 #4) — "
            "channel transformer + LSTM + MIL-pooled SOZ head, hard-example-mining "
            "variant. Pretrained on TUSZ fold-4."
        ),
        loader=_load_deepsoz_hem,
        checkpoints=[
            CheckpointSpec(
                identifier="deepsoz_hem_pretrained",
                model_family="deepsoz_hem",
                variant="amruth_sn_fold4",
                # A folder: upstream's baselines.py (imported by path),
                # deepsoz_fold4.pth_4.tar and LICENSE, fetched from the
                # pinned commit (GPL-3.0, so not shipped).
                source_type="github_commit_files",
                source_reference=UPSTREAM,
                checkpoint_path="artifacts/models/foundation/deepsoz_hem",
                input_kind="time_series",
                expected_channels=("eeg",),
                expected_sampling_rate=256.0,
                # Spec default mirrors the pretrain context (600 s); the wrapper
                # accepts any positive integer T at fs=256 Hz (see
                # backbones/deepsoz_hem.py — fs is architecturally hard, T is
                # not). The 10 s sweep overrides this via
                # `--expected-epoch-seconds 10` at the entrypoint.
                expected_epoch_seconds=600.0,
                pretraining_datasets=("tusz_v2.0.3", "siena"),
                # Authors trained on TUSZ + Siena; CHB-MIT explicitly excluded
                # because the channel transformer was trained on average-
                # referenced data, not bipolar pairs. native_head_eval is the
                # right comparison on TUSZ + Siena.
                finetuned_datasets=("tusz", "siena"),
                has_classifier_head=True,
                embedding_key="tx_global_mean",
                embedding_dim=256,
                wrapper_name="deepsoz_hem",
                expected_montage="unipolar_average_ref",
                status="ready",
                notes=(
                    "Code (baselines.py) and weights (deepsoz_fold4.pth_4.tar) from "
                    "amruth-sn/deepsoz-hem @ a7c13bd (GPL-3.0), fetched into the models "
                    "root by `neuroatlas models download deepsoz_hem_pretrained`; not "
                    "shipped with neuroatlas. Native head "
                    "emits per-second softmax logits (B,600,2); wrapper reduces "
                    "to (B,2) by max-over-time of seizure prob. Embedding = "
                    "mean-over-time of tx_encoder global-token output (256-d)."
                ),
            )
        ],
    )
]
