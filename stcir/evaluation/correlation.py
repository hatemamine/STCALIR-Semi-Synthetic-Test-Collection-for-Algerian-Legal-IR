from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr

from stcir.utils import get_logger, load_qrels, load_run

logger = get_logger(__name__)

Run   = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]


def compute_system_correlation(
    human_qrels: Qrels,
    synthetic_qrels: Qrels,
    runs: dict[str, Run],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each metric, rank the systems by score under human vs. synthetic qrels.
    Compute Global Kendall's τ and Spearman's ρ between the two rankings.

    Returns a DataFrame: index=metric, columns=[kendall_tau, spearman_rho].
    """
    from stcir.evaluation.metrics import evaluate_multiple_runs

    if metrics is None:
        metrics = ["MRR@10", "nDCG@10", "Recall@10"]

    human_scores     = evaluate_multiple_runs(human_qrels,     runs, metrics)
    synthetic_scores = evaluate_multiple_runs(synthetic_qrels, runs, metrics)

    records = []
    for metric in human_scores.columns:
        h = human_scores[metric].dropna()
        s = synthetic_scores[metric].reindex(h.index).dropna()
        common = h.index.intersection(s.index)
        h, s = h[common], s[common]

        tau, tau_p   = kendalltau(h.values, s.values)
        rho, rho_p   = spearmanr(h.values, s.values)
        records.append({
            "metric":       metric,
            "kendall_tau":  round(tau, 4),
            "kendall_p":    round(tau_p, 4),
            "spearman_rho": round(rho, 4),
            "spearman_p":   round(rho_p, 4),
            "n_systems":    len(common),
        })

    df = pd.DataFrame(records).set_index("metric")
    logger.info(f"System-level correlation computed over {len(runs)} systems, {len(metrics)} metrics")
    return df


def global_rank_correlation(
    human_qrels_path: str,
    synthetic_qrels_path: str,
    top_k: int = 10,
) -> dict[str, float]:
    """
    Global Kendall's τ and Spearman's ρ at the query-document pair level.
    Mirrors the approach in KENDALL_TAU+R@k_MRR@k.py.
    """
    human_qrels     = load_qrels(human_qrels_path)
    synthetic_qrels = load_qrels(synthetic_qrels_path)

    # Build pair-level rank arrays
    all_pairs: set[tuple[str, str]] = set()
    for qid, docs in human_qrels.items():
        for pid in docs:
            all_pairs.add((qid, pid))
    for qid, docs in synthetic_qrels.items():
        if len(list(docs.items())) <= top_k:
            for pid in docs:
                all_pairs.add((qid, pid))

    h_ranks, s_ranks = [], []
    for qid, pid in all_pairs:
        h_rel = human_qrels.get(qid, {}).get(pid, 0)
        s_rel = synthetic_qrels.get(qid, {}).get(pid, 0)
        h_ranks.append(h_rel)
        s_ranks.append(s_rel)

    tau, _  = kendalltau(h_ranks, s_ranks)
    rho, _  = spearmanr(h_ranks, s_ranks)
    return {"global_kendall_tau": round(tau, 4), "global_spearman_rho": round(rho, 4)}


def plot_system_scatter(
    human_qrels: Qrels,
    synthetic_qrels: Qrels,
    runs: dict[str, Run],
    metric: str = "MRR@10",
    title: str | None = None,
    save_path: str | None = None,
):
    """Scatter plot: human metric score (x) vs synthetic metric score (y) per system."""
    import matplotlib.pyplot as plt
    from stcir.evaluation.metrics import evaluate_multiple_runs

    h = evaluate_multiple_runs(human_qrels,     runs, [metric])[metric]
    s = evaluate_multiple_runs(synthetic_qrels, runs, [metric])[metric]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(h.values, s.values, zorder=3)

    for name, hv, sv in zip(h.index, h.values, s.values):
        ax.annotate(name, (hv, sv), fontsize=7, ha="left", va="bottom")

    lo = min(h.min(), s.min()) * 0.95
    hi = max(h.max(), s.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect agreement")

    tau, _ = kendalltau(h.values, s.values)
    rho, _ = spearmanr(h.values, s.values)
    ax.set_xlabel(f"Human {metric}")
    ax.set_ylabel(f"Synthetic {metric}")
    ax.set_title(title or f"System-level correlation ({metric})\nτ={tau:.3f}, ρ={rho:.3f}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Scatter plot saved → {save_path}")
    return fig
