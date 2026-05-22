from __future__ import annotations

import numpy as np

from stcir.indexing.encoder import BiEncoder
from stcir.indexing.faiss_index import FaissIndex
from stcir.utils import get_logger

logger = get_logger(__name__)

Run = dict[str, list[tuple[str, float]]]


class DenseRetriever:
    """Retrieves top-k passages using a bi-encoder + FAISS index."""

    def __init__(self, encoder: BiEncoder, index: FaissIndex):
        self.encoder = encoder
        self.index   = index

    def retrieve(
        self,
        topics: list[dict],   # [{"qid": str, "text": str}, ...]
        top_k: int = 1000,
        batch_size: int = 64,
    ) -> Run:
        queries  = [t["text"] for t in topics]
        qids     = [str(t["qid"]) for t in topics]

        logger.info(f"Encoding {len(queries)} queries with {self.encoder.short_name} ...")
        q_embs = self.encoder.encode_queries(queries, batch_size=batch_size)

        logger.info(f"Searching FAISS (top_k={top_k}) ...")
        scores, indices = self.index.search(q_embs, top_k)

        run: Run = {}
        for i, qid in enumerate(qids):
            run[qid] = [
                (self.index.pid_from_index(int(indices[i, j])), float(scores[i, j]))
                for j in range(top_k)
                if indices[i, j] >= 0
            ]

        logger.info(f"Dense retrieval done: {len(run)} queries, model={self.encoder.short_name}")
        return run
