from __future__ import annotations

from stcir.retrieval.rrf import rrf_fuse
from stcir.utils import get_logger

logger = get_logger(__name__)

Run = dict[str, list[tuple[str, float]]]


def rerank_with_rrf(
    cross_encoder_runs: list[Run],
    run_labels: list[str] | None = None,
    rrf_k: int = 60,
    top_k: int = 10,
) -> Run:
    """
    Fuse multiple cross-encoder runs via RRF and return top_k per topic.
    This is Stage-2 RRF (Phase 4 of the pipeline).
    """
    labels = run_labels or [f"CE_{i}" for i in range(len(cross_encoder_runs))]
    logger.info(f"RRF Stage-2: fusing {len(cross_encoder_runs)} cross-encoder runs ({labels}) → top {top_k}")
    fused = rrf_fuse(cross_encoder_runs, k=rrf_k, top_k=top_k)
    logger.info(f"Stage-2 RRF done: {len(fused)} topics, top {top_k} per topic")
    return fused
