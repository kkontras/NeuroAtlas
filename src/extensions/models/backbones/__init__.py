from .base import BenchmarkBackbone

# Legacy alias — some backbones import this name.
BenchmarkModelWrapper = BenchmarkBackbone

__all__ = ["BenchmarkBackbone", "BenchmarkModelWrapper"]
