"""Vendored upstream model definitions for supervised baselines.

Contents are copied verbatim from their upstream repositories under
permissive licences, to avoid a cross-repo runtime dependency. Each
subpackage keeps its upstream licence file: ``seizure_transformer/`` is
Kerui Wu's MIT-licensed SeizureTransformer architecture.

DeepSOZ-HEM is deliberately not here. Its code and checkpoint
(amruth-sn/deepsoz-hem @ a7c13bdbb6d86e016f929ba108370cc614e9882c) are
GPL-3.0, which this MIT package cannot carry; `neuroatlas models download
deepsoz_hem_pretrained` fetches them from upstream at that commit (see
``_checkpoint_download.GITHUB_COMMIT_FILES`` and ``backbones/deepsoz_hem.py``).
"""
