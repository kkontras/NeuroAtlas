"""SeizureTransformer architecture — vendored from Kerui Wu et al. (MIT License).

Source: https://github.com/keruiwu/SeizureTransformer (time_step_level/model.py)
Paper:  Wu, Zhao, Yener. "Large EEG-U-Transformer for Time-Step Level Detection
        Without Pre-Training." arXiv:2504.00336, 2025. EpilepsyBench 2025 winner.

The file ``architecture.py`` mirrors the Clarity-Digital-Twin/SeizureTransformer
variant paired with the released weights — its ctor signature matches the
pretrained state_dict layout exactly.
"""
from .architecture import SeizureTransformer

__all__ = ["SeizureTransformer"]
