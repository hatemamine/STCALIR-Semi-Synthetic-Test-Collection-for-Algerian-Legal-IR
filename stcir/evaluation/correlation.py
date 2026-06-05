from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr

from stcir.utils import get_logger, load_qrels, load_run

logger = get_logger(__name__)

Run   = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]


# ── Self-contained rank-based metrics (no ir_measures dependency) ─────────────
# Matches the approach from the original working script:
#   iterate over qids in the qrel, look up each system's ranked list,
#   compute MRR / Hit / nDCG / Recall at cutoff k.

def _base_id(pid: str) -> str:
    """Strip '#segment' suffix from a passage ID to get the document-level ID."""
    return pid.split("#")[0]


def _eval_run_custom(qrels: Qrels, run: Run, k: int = 10) -> dict[str, float]:
    """
    Compute MRR@k, Hit@k, nDCG@k, Recall@k for one system run.
    Iterates over qrels qids; queries absent from the run score 0.

    Handles the common mr-tydi mismatch where human qrels store document-level
    IDs (e.g. "1234567") while retrieval runs use passage-level IDs with a
    segment suffix (e.g. "1234567#0"). Relevance is matched by either exact
    pid or base document ID, so both formats work transparently.
    """
    run_norm = {str(qid): [(str(pid), sc) for pid, sc in docs]
                for qid, docs in run.items()}

    mrr = hit = ndcg = recall = 0.0
    n = 0

    for qid_raw, rel_docs in qrels.items():
        qid = str(qid_raw)
        rel_pids = {str(p) for p in rel_docs}
        if not rel_pids:
            continue
        n += 1
        # Build base-document set for doc-level vs passage-level matching
        rel_base = {_base_id(p) for p in rel_pids}

        ranked = [pid for pid, _ in run_norm.get(qid, [])]

        def is_rel(pid: str) -> bool:
            return pid in rel_pids or _base_id(pid) in rel_base

        # MRR@k — reciprocal rank of the first relevant passage/doc
        rr = 0.0
        for rank, pid in enumerate(ranked[:k], start=1):
            if is_rel(pid):
                rr = 1.0 / rank
                break
        mrr += rr

        # Hit@k
        top_k = ranked[:k]
        if any(is_rel(p) for p in top_k):
            hit += 1

        # nDCG@k
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, pid in enumerate(top_k, start=1)
            if is_rel(pid)
        )
        n_rel = min(len(rel_base), k)   # count distinct relevant docs
        idcg  = sum(1.0 / np.log2(r + 1) for r in range(1, n_rel + 1))
        ndcg += (dcg / idcg) if idcg > 0 else 0.0

        # Recall@k — distinct relevant base-docs retrieved / total relevant
        retrieved_bases = {_base_id(p) for p in top_k if is_rel(p)}
        recall += len(retrieved_bases) / max(len(rel_base), 1)

    denom = n if n > 0 else 1
    return {
        f"MRR@{k}":    mrr    / denom,
        f"Hit@{k}":    hit    / denom,
        f"nDCG@{k}":   ndcg   / denom,
        f"Recall@{k}": recall / denom,
    }


def _qid_overlap_info(qrels: Qrels, runs: dict[str, Run]) -> tuple[int, int, int]:
    """Return (n_qrel_qids, n_run_qids, n_overlap) for diagnostics."""
    qrel_qids = {str(q) for q in qrels}
    run_qids  = {str(q) for run in runs.values() for q in run}
    return len(qrel_qids), len(run_qids), len(qrel_qids & run_qids)


# ── System-level correlation ──────────────────────────────────────────────────

def compute_system_correlation(
    human_qrels: Qrels,
    synthetic_qrels: Qrels,
    runs: dict[str, Run],
    metrics: list[str] | None = None,
    k: int = 10,
) -> pd.DataFrame:
    """
    For each metric, compute per-system scores under human and synthetic qrels,
    then compute Kendall's τ and Spearman's ρ between the two system rankings.

    Uses a self-contained rank-based implementation (no ir_measures) to avoid
    canonical-name mismatches and to gracefully handle partial qid overlap.

    Returns a DataFrame indexed by metric with columns:
        kendall_tau, kendall_p, spearman_rho, spearman_p, n_systems
    """
    if metrics is None:
        metrics = [f"MRR@{k}", f"nDCG@{k}", f"Recall@{k}"]

    # ── Diagnostics ───────────────────────────────────────────────────────
    nq_h, nr_h, ov_h = _qid_overlap_info(human_qrels, runs)
    nq_s, nr_s, ov_s = _qid_overlap_info(synthetic_qrels, runs)
    logger.info(
        f"Human qrels: {nq_h} qids, runs: {nr_h} qids, overlap: {ov_h}. "
        f"Synthetic qrels: {nq_s} qids, overlap with runs: {ov_s}."
    )
    if ov_h == 0:
        logger.warning(
            "Zero overlap between human qrels query IDs and system run query IDs. "
            "All systems will score 0 under human qrels → correlation will be NaN. "
            f"Sample human qrel qids : {list(human_qrels)[:5]}. "
            f"Sample run qids        : {list(next(iter(runs.values())))[:5]}."
        )

    # ── Compute per-system scores ─────────────────────────────────────────
    system_names = list(runs.keys())
    h_scores: dict[str, list[float]] = {m: [] for m in metrics}
    s_scores: dict[str, list[float]] = {m: [] for m in metrics}

    for name in system_names:
        run = runs[name]
        h = _eval_run_custom(human_qrels,     run, k=k)
        s = _eval_run_custom(synthetic_qrels, run, k=k)
        for m in metrics:
            h_scores[m].append(h.get(m, 0.0))
            s_scores[m].append(s.get(m, 0.0))

    # ── Correlate ─────────────────────────────────────────────────────────
    records = []
    for metric in metrics:
        h_arr = np.array(h_scores[metric])
        s_arr = np.array(s_scores[metric])
        n     = len(system_names)

        if n < 2 or np.unique(h_arr).size < 2 or np.unique(s_arr).size < 2:
            tau, tau_p, rho, rho_p = float("nan"), float("nan"), float("nan"), float("nan")
            logger.warning(
                f"Metric '{metric}': all {n} systems have identical scores under "
                f"human or synthetic qrels — correlation undefined. "
                f"Human scores: {h_arr.tolist()}. Synthetic scores: {s_arr.tolist()}."
            )
        else:
            tau, tau_p = kendalltau(h_arr, s_arr)
            rho, rho_p = spearmanr(h_arr, s_arr)

        records.append({
            "metric":       metric,
            "kendall_tau":  round(float(tau),   4),
            "kendall_p":    round(float(tau_p), 4),
            "spearman_rho": round(float(rho),   4),
            "spearman_p":   round(float(rho_p), 4),
            "n_systems":    n,
        })

    df = pd.DataFrame(records).set_index("metric")
    logger.info(
        f"System-level correlation: {len(runs)} systems, {len(metrics)} metrics"
    )
    return df


# ── Global pair-level correlation ─────────────────────────────────────────────

def global_rank_correlation(
    human_qrels_path: str,
    synthetic_qrels_path: str,
    top_k: int = 10,
) -> dict[str, float]:
    """
    Global Kendall's τ and Spearman's ρ at the query-document pair level.
    """
    human_qrels     = load_qrels(human_qrels_path)
    synthetic_qrels = load_qrels(synthetic_qrels_path)

    all_pairs: set[tuple[str, str]] = set()
    for qid, docs in human_qrels.items():
        for pid in docs:
            all_pairs.add((str(qid), str(pid)))
    for qid, docs in synthetic_qrels.items():
        if len(docs) <= top_k:
            for pid in docs:
                all_pairs.add((str(qid), str(pid)))

    h_ranks, s_ranks = [], []
    for qid, pid in all_pairs:
        h_ranks.append(human_qrels.get(qid, {}).get(pid, 0))
        s_ranks.append(synthetic_qrels.get(qid, {}).get(pid, 0))

    tau, _ = kendalltau(h_ranks, s_ranks)
    rho, _ = spearmanr(h_ranks, s_ranks)
    return {"global_kendall_tau": round(tau, 4), "global_spearman_rho": round(rho, 4)}


# ── Scatter plot ──────────────────────────────────────────────────────────────

def plot_system_scatter(
    human_qrels: Qrels,
    synthetic_qrels: Qrels,
    runs: dict[str, Run],
    metric: str = "MRR@10",
    title: str | None = None,
    save_path: str | None = None,
    k: int = 10,
):
    """Scatter plot: human metric score (x) vs synthetic metric score (y) per system."""
    import matplotlib.pyplot as plt

    system_names = list(runs.keys())
    h_vals = [_eval_run_custom(human_qrels,     runs[n], k=k).get(metric, 0.0) for n in system_names]
    s_vals = [_eval_run_custom(synthetic_qrels, runs[n], k=k).get(metric, 0.0) for n in system_names]

    h = np.array(h_vals)
    s = np.array(s_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(h, s, zorder=3)

    for name, hv, sv in zip(system_names, h, s):
        ax.annotate(name, (hv, sv), fontsize=7, ha="left", va="bottom")

    lo = min(h.min(), s.min()) * 0.95
    hi = max(h.max(), s.max()) * 1.05
    if lo == hi:
        lo, hi = lo - 0.01, hi + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect agreement")

    if np.unique(h).size >= 2 and np.unique(s).size >= 2:
        tau, _ = kendalltau(h, s)
        rho, _ = spearmanr(h, s)
        corr_text = f"Kendall's τ = {tau:.3f}\nSpearman ρ = {rho:.3f}"
    else:
        corr_text = "correlation undefined\n(constant scores)"

    ax.set_xlabel(f"Human {metric}")
    ax.set_ylabel(f"Synthetic {metric}")
    ax.set_title(title or f"System-level correlation ({metric})")
    ax.text(lo, hi, corr_text, verticalalignment="top", fontsize=9)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Scatter plot saved → {save_path}")
    return fig
