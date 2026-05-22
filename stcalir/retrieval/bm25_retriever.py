from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from stcalir.indexing.bm25 import BM25Index
from stcalir.utils import get_logger

logger = get_logger(__name__)

Run = dict[str, list[tuple[str, float]]]   # qid → [(pid, score), ...]


class BM25Retriever:
    """Retrieves top-k passages for each topic using BM25."""

    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(
        self,
        topics: list[dict],   # [{"qid": str, "text": str}, ...]
        top_k: int = 1000,
    ) -> Run:
        run: Run = {}
        for topic in tqdm(topics, desc="BM25 retrieval"):
            qid   = str(topic["qid"])
            query = str(topic["text"])
            scores = self.index.get_scores(query)
            top_idx = np.argsort(scores)[::-1][:top_k]
            run[qid] = [(self.index.pids[i], float(scores[i])) for i in top_idx]

        logger.info(f"BM25 retrieval done: {len(run)} queries, top_k={top_k}")
        return run
