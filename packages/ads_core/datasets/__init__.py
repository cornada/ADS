"""Dataset registry and loaders for ADS.

Provides unified access to all institutional datasets:
- toy: Built-in test dataset
- mit: MIT curriculum (catalog + OCW)
- ucb: UC Berkeley outcomes (FDS)
- asu: ASU FDS survey + outcomes
"""
from ads_core.datasets.registry import (
    DatasetBundle,
    load_dataset,
    list_datasets,
    get_dataset_info,
    validate_artifact,
    DATASET_REGISTRY,
)

__all__ = [
    "DatasetBundle",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "validate_artifact",
    "DATASET_REGISTRY",
]
