"""Task container + sampler interface for priors."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np


@dataclass
class Task:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    is_cat: np.ndarray
    families: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"X_train": self.X_train, "y_train": self.y_train,
                "X_test": self.X_test, "y_test": self.y_test,
                "is_cat": self.is_cat, "families": self.families, **self.meta}


class TaskSampler(Protocol):
    def __call__(self, rng: np.random.Generator, **kw) -> Task: ...
