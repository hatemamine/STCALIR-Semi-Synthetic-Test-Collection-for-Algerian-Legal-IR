from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from stcir.utils import get_logger, save_run, save_qrels

logger = get_logger(__name__)

Run   = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]

METRIC_MAP = {
    "MRR@10":    "RR@10",
    "MRR@100":   "RR@100",
    "nDCG@10":   "nDCG@10",
    "nDCG@100":  "nDCG@100",
    "MAP":       "AP",
    "Recall@10": "R@10",
    "Recall@100":"R@100",
    "Recall@1000":"R@1000",
    "Hit@10":    "Rprec",
    "P@10":      "P@10",
}


def evaluate_run(
    qrels: Qrels,
    run: Run,
    metrics: list[str] | None = None,
    run_name: str = "system",
) -> pd.Series:
    """
    Evaluate a retrieval run against qrels.
    Returns a pd.Series of metric → score.
    """
    import ir_measures
    from ir_measures import parse_measure, nDCG, RR, AP, R, P, Judged

    if metrics is None:
        metrics = ["MRR@10", "nDCG@10", "MAP", "Recall@10", "P@10"]

    # Convert to ir_measures format
    qrels_records = [
        ir_measures.Qrel(query_id=qid, doc_id=pid, relevance=rel)
        for qid, docs in qrels.items()
        for pid, rel in docs.items()
    ]
    run_records = [
        ir_measures.ScoredDoc(query_id=qid, doc_id=pid, score=float(score))
        for qid, ranked in run.items()
        for pid, score in ranked
    ]

    measures = _parse_metrics(metrics)
    results = ir_measures.calc_aggregate(measures, qrels_records, run_records)
    scores = {str(m): v for m, v in results.items()}
    return pd.Series(scores, name=run_name)


def evaluate_multiple_runs(
    qrels: Qrels,
    runs: dict[str, Run],   # name → run
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple runs, returning a DataFrame with systems as rows and metrics as columns.
    """
    rows = []
    for name, run in runs.items():
        s = evaluate_run(qrels, run, metrics, run_name=name)
        rows.append(s)
    df = pd.DataFrame(rows)
    logger.info(f"Evaluated {len(runs)} systems on {len(df.columns)} metrics")
    return df


def hit_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    """Proportion of queries with ≥1 relevant doc in top-k."""
    hits = 0
    total = 0
    for qid, ranked in run.items():
        total += 1
        top_k_pids = {pid for pid, _ in ranked[:k]}
        rel_pids = {pid for pid, rel in qrels.get(qid, {}).items() if rel > 0}
        if top_k_pids & rel_pids:
            hits += 1
    return hits / total if total else 0.0


def _parse_metrics(metrics: list[str]):
    import ir_measures
    parsed = []
    for m in metrics:
        try:
            parsed.append(ir_measures.parse_measure(m))
        except Exception:
            logger.warning(f"Could not parse metric '{m}', skipping")
    return parsed
