from __future__ import annotations

from sentence_transformers.cross_encoder import CrossEncoder
from tqdm.auto import tqdm

from stcir.utils import get_logger

logger = get_logger(__name__)

Run = dict[str, list[tuple[str, float]]]


class CrossEncoderReranker:
    """
    Scores (query, passage) pairs with a cross-encoder model.
    Input: pool of candidates from Stage 1.
    Output: re-ranked run (scores from the cross-encoder).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.device = device

    def rerank(
        self,
        topics: list[dict],            # [{"qid": str, "text": str}, ...]
        pool: Run,                     # Stage-1 pool
        passage_lookup: dict[str, str],  # pid → text
        top_k: int = 1000,
        batch_size: int = 32,
    ) -> Run:
        """
        For each topic, score the pooled candidates and return sorted run.
        """
        topic_map = {str(t["qid"]): t["text"] for t in topics}
        run: Run  = {}

        for qid, candidates in tqdm(pool.items(), desc=f"Reranking [{self.short_name}]"):
            query = topic_map.get(qid, "")
            pairs: list[tuple[str, str]] = []
            pids: list[str] = []

            for pid, _ in candidates[:top_k]:
                text = passage_lookup.get(pid, "")
                if text:
                    pairs.append((query, text))
                    pids.append(pid)

            if not pairs:
                run[qid] = candidates[:top_k]
                continue

            scores = self._batch_score(pairs, batch_size)
            ranked = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
            run[qid] = ranked

        logger.info(f"Cross-encoder reranking done: {self.short_name}, {len(run)} queries")
        return run

    def _batch_score(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        all_scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores.tolist())
        return all_scores

    @property
    def short_name(self) -> str:
        return self.model_name.split("/")[-1]
