from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from stcalir.utils import get_logger, save_pickle, load_pickle

logger = get_logger(__name__)


class BM25Index:
    """BM25 index backed by rank_bm25 (pure Python, no Java dependency)."""

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._pids: list[str] = []

    def build(self, passages: list[dict], tokenize_fn=None) -> "BM25Index":
        """
        Build BM25 index from passage list.
        passages: [{"pid": str, "text": str}, ...]
        """
        if tokenize_fn is None:
            tokenize_fn = lambda t: t.lower().split()

        logger.info(f"Building BM25 index over {len(passages):,} passages ...")
        self._pids = [p["pid"] for p in passages]
        tokenized = [tokenize_fn(p["text"]) for p in tqdm(passages, desc="Tokenizing for BM25")]
        self._index = BM25Okapi(tokenized)
        self._tokenize_fn = tokenize_fn
        logger.info("BM25 index built")
        return self

    def save(self, path: Path | str) -> None:
        save_pickle({"index": self._index, "pids": self._pids}, path)
        logger.info(f"BM25 index saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        data = load_pickle(path)
        obj = cls()
        obj._index = data["index"]
        obj._pids  = data["pids"]
        obj._tokenize_fn = lambda t: t.lower().split()
        logger.info(f"BM25 index loaded ← {path}")
        return obj

    def get_scores(self, query: str) -> np.ndarray:
        tokens = self._tokenize_fn(query)
        return self._index.get_scores(tokens)

    @property
    def pids(self) -> list[str]:
        return self._pids
