import os
import pandas as pd
import numpy as np

# ---------------------- RRF ----------------------
import pandas as pd
from collections import defaultdict
import glob

def load_ranking_file(path):
    """Load a ranking file of format: qid, docid, rank."""
    return pd.read_csv(path, sep="\t", names=["qid", "docid", "rank", "score"])

def rrf(ranking_lists, k=60):
    """Apply RRF fusion to a list of ranking lists (docid lists)."""
    scores = defaultdict(float)
    for ranking in ranking_lists:
        for rank, docid in enumerate(ranking):
            scores[docid] += 1.0 / (k + rank + 1)
    # Sort documents by RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def rrf_fusion(run_files, output_path=None, k=60, top_k=1000):
    """
    Apply RRF to multiple ranking files with identical structure.
    
    run_files: list of file paths
    output_path: where to save fused run (optional)
    Returns: fused DataFrame
    """
    
    # 1) Load all runs
    runs = [load_ranking_file(f) for f in run_files]

    # All topics from first file
    topics = runs[0]["qid"].unique()

    fused_rows = []

    for qid in topics:
        ranking_lists = []

        # 2) Extract docids per topic from each run
        for df in runs:
            docs = df[df["qid"] == qid].sort_values("rank")["docid"].tolist()
            ranking_lists.append(docs)

        # 3) Apply RRF
        fused_docs = rrf(ranking_lists, k=k)

        # 4) Keep top_k
        for rank, (docid, score) in enumerate(fused_docs[:top_k], start=1):
            fused_rows.append([qid, docid, rank])

    # Create DataFrame
    fused_df = pd.DataFrame(fused_rows, columns=["qid", "docid", "rank"])

    # Optional save
    if output_path:
        fused_df.to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Saved RRF run to: {output_path}")

    return fused_df


# ---------------------- LOAD CSVs ----------------------
human_qrels_path = os.path.expanduser("STCALIRjudgments.csv")
fs = ["./test model/"+f for f in os.listdir("./test model/")]
print(fs)
print(len(fs))




# ---------------------- METRICS ----------------------
def compute_metrics(human_qrels, synthetic, k=10):
    """
    Compute metrics for single-human-doc queries:
    - MRR
    - Hit@k
    - nDCG@k
    - Mean Rank
    - Recall@k
    - Top-k coverage
    - Failure rate
    """
    qids = human_qrels["qid"].unique()
    mrr_total = 0
    hit_total = 0
    ndcg_total = 0
    mean_rank_total = 0
    recall_total = 0
    coverage_count = 0
    failure_count = 0
    n_queries = len(qids)

    for qid in qids:
        human_doc = human_qrels[human_qrels["qid"] == qid]["docid"].iloc[0]

        synth_docs = synthetic[synthetic["qid"] == qid].sort_values("rank")["docid"].tolist()

        if human_doc in synth_docs:
            rank_position = synth_docs.index(human_doc) + 1  # ranks start at 1
            mrr_total += 1.0 / rank_position
            mean_rank_total += rank_position
            hit_total += 1 if rank_position <= k else 0
            recall_total += 1 if rank_position <= k else 0
            ndcg_total += 1 / np.log2(rank_position + 1)

            coverage_count += 1  # human doc appears in ranking
        else:
            # human doc not in synthetic ranking
            rank_position = len(synth_docs) + 1
            mean_rank_total += rank_position
            ndcg_total += 0
            # no hit, no recall
            failure_count += 1

    mrr = mrr_total / n_queries
    hit_k = hit_total / n_queries
    mean_rank = mean_rank_total / n_queries
    ndcg_k = ndcg_total / n_queries
    recall_k = recall_total / n_queries
    topk_coverage = coverage_count / n_queries
    failure_rate = failure_count / n_queries

    return {
        "MRR": mrr,
        f"Hit@{k}": hit_k,
        f"nDCG@{k}": ndcg_k,
        "Mean Rank": mean_rank,
        f"Recall@{k}": recall_k,
        f"Top-{k} Coverage": topk_coverage,
        "Failure Rate": failure_rate
    }


# Load human qrels (only keep qid and docid)
# Load CSV with no header
human_qrels = pd.read_csv(
	"STCALIRjudgments.csv",
	sep=",",
	header=0,                # use the first row ("query_id,doc_id,relevant")
	dtype={"query_id": int, "doc_id": str, "relevant": int}
	)	
#print(human_qrels.head())
human_qrels = human_qrels.rename(columns={
	"query_id": "qid",
	"doc_id": "docid"
	})
# Sort: relevant=1 first, then 0
human_qrels = human_qrels.sort_values(by=["qid", "relevant"], ascending=[True, False])
# Total number of judgments
num_judgments = len(human_qrels)
print("Number of Judgments:", num_judgments)

# Optionally, number of relevant judgments
num_relevant = human_qrels['relevant'].sum()
print("Number of Relevant Judgments:", num_relevant)
# Assign ranks per query
human_qrels["rank"] = human_qrels.groupby("qid").cumcount() + 1

# Final columns
human_qrels = human_qrels[["qid", "docid", "rank"]]
#print(human_qrels)
# Assign ranks per query
human_qrels["rank"] = human_qrels.groupby("qid").cumcount() + 1

# Final columns
human_qrels = human_qrels[["qid", "docid", "rank"]]

for i in fs : 
	synthetic_path = i
	#print(human_qrels.head())
	#Load synthetic rankings (qid, docid, rank)
	synthetic = pd.read_csv(
    synthetic_path,
    sep="\t",
    header=None,
    names=["qid", "docid", "rank", "score", "querytext", "doctext"]
	)[["qid", "docid", "rank"]]


	# ---------------------- RUN ----------------------
	print("---------------------- RUN ---------------------")
	print(i)
	metrics = compute_metrics(human_qrels, synthetic, k=10)
	print(metrics)
	print("---------------------- RUN ---------------------")
