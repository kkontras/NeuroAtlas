"""Fitting a probe on cached embeddings and scoring it.

sklearn only -- logistic regression, ridge, and the MLP variants. The probes
are CPU-only by design, so nothing here imports torch.
"""
