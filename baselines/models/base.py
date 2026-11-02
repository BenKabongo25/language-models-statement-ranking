from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from baselines.data import IndexedInteraction, RankingData


class BaseRanker(ABC):
    deterministic: bool = False
    uses_gpu: bool = False

    def __init__(self, data: RankingData, cfg: dict[str, Any], seed: int, device: str = "cpu"):
        self.data = data
        self.cfg = cfg
        self.seed = int(seed)
        self.device = device

    @abstractmethod
    def fit(self, interactions: list[IndexedInteraction], *, validation: list[IndexedInteraction] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def score(self, user: int, item: int, candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rank(self, user: int, item: int, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(self.score(user, item, candidates), dtype=np.float64)
        # Deterministic tie-break by canonical local statement index.
        order = np.lexsort((candidates, -scores))
        return candidates[order], scores[order]

    def save(self, path: str | Path) -> None:
        pass


class TorchRanker(BaseRanker):
    uses_gpu = True

    def _torch(self):
        import torch
        return torch
