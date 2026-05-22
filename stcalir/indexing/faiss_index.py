from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from stcalir.utils import get_logger

logger = get_logger(__name__)


class FaissIndex:
    """
    Inner-product FAISS index (works with cosine-normalized embeddings).
    Uses Flat for ≤ 500k passages, IVFFlat for larger corpora.
    """

    IVF_THRESHOLD = 500_000

    def __init__(self):
        self._index: faiss.Index | None = None
        self._pids: list[str] = []

    def build(self, embeddings: np.ndarray, pids: list[str]) -> "FaissIndex":
        n, d = embeddings.shape
        self._pids = pids

        if n <= self.IVF_THRESHOLD:
            self._index = faiss.IndexFlatIP(d)
        else:
            n_lists = min(4096, int(n ** 0.5))
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, n_lists, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Training IVF index with {n_lists} lists on {n:,} vectors ...")
            self._index.train(embeddings)
            self._index.nprobe = 64

        self._index.add(embeddings)
        logger.info(f"FAISS index built: {n:,} vectors, dim={d}")
        return self

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) both shape (n_queries, top_k)."""
        scores, indices = self._index.search(query_embeddings, top_k)
        return scores, indices

    def pid_from_index(self, idx: int) -> str:
        return self._pids[idx]

    def save(self, index_path: Path | str, pids_path: Path | str) -> None:
        faiss.write_index(self._index, str(index_path))
        import pickle
        with open(pids_path, "wb") as f:
            pickle.dump(self._pids, f)
        logger.info(f"FAISS index saved → {index_path}")

    @classmethod
    def load(cls, index_path: Path | str, pids_path: Path | str) -> "FaissIndex":
        import pickle
        obj = cls()
        obj._index = faiss.read_index(str(index_path))
        with open(pids_path, "rb") as f:
            obj._pids = pickle.load(f)
        logger.info(f"FAISS index loaded ← {index_path} ({obj._index.ntotal:,} vectors)")
        return obj
