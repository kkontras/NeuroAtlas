"""NeuroAtlas: a benchmark for EEG foundation models.

The package has three layers -- ``neuroatlas.benchmarking_helpers`` (the
engine), ``neuroatlas.extensions`` (datasets, models, tasks) and
``neuroatlas.entrypoints`` (the verbs) -- and the ``neuroatlas`` command in
front of them. Importing this module is cheap: it pulls in neither torch nor
any dataset reader.
"""

__version__ = "0.1.0"
