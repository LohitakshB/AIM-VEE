"""AIM-VEE package."""

from __future__ import annotations

import importlib
from typing import List

_API_EXPORTS = (
    "SplitConfig",
    "build_metadata_from_exc_ss",
    "prepare_splits",
    "train_schnet",
    "train_chemprop",
    "train_rf_morgan",
    "generate_qemfi_cm",
    "prep_qemfi",
    "train_qemfi",
    "train_mff_mlp",
    "infer_model",
)

__all__ = [
    "api",
    *_API_EXPORTS,
    "datasets",
    "experiments",
    "features",
    "models",
    "qemfi_utils",
    "trainers",
    "utils",
]


def __getattr__(name: str):
    if name == "api" or name in _API_EXPORTS:
        module = importlib.import_module("aimvee.api")
        if name == "api":
            globals()[name] = module
            return module
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'aimvee' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
