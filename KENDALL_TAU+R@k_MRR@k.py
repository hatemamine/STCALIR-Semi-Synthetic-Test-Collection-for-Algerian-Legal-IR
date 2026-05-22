import os
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import spearmanr

# ---------------------- LOAD CSVs ----------------------
human_qrels_path = os.path.expanduser("qrels.test.mrtydi.txt")
import os
fs = ["./FirstStage_mrTydi/"+f for f in os.listdir("./FirstStage_mrTydi/")]
#print(fs)
#print(len(fs))

ss = ["./SecondStage/"+f for f in os.listdir("./SecondStage/")]
#print(ss)
#print(len(ss))

synthetic_path = fs[0]

# Load human qrels
human_qrels = pd.read_csv(
    human_qrels_path,
    sep=r"\s+",
    header=None,
    names=["qid", "Q0", "docid", "rank"]
)[["qid", "docid", "rank"]]
#print(human_qrels)
# Load synthetic pseudo-qrels
pairs = pd.read_csv(
    synthetic_path,
    sep="\t",
    header=None,
    names=["qid", "docid", "rank", "score", "querytext", "doctext"]
)[["qid", "docid", "rank"]]
# Keep only rank < 11
pairs = pairs[pairs["rank"] < 11]
print(len(pairs))
# ---------------------- MERGE ----------------------
# Only keep overlapping query-document pairs
#merged = pd.merge(human_qrels, pairs, on=["qid", "docid"], how="inner", suffixes=('_human', '_synthetic'))
merged = pd.merge(human_qrels, pairs, on=["qid","docid"], how="outer", suffixes=('_human', '_synthetic'))
merged["rank_human"].fillna(9999, inplace=True)
merged["rank_synthetic"].fillna(9999, inplace=True)

print("Overlap size:", len(merged))
print(merged.head())
# ---------------------- GLOBAL KENDALL'S TAU ----------------------
global_tau = kendalltau(merged['rank_human'], merged['rank_synthetic'])
print("Global Kendall’s tau:", global_tau.statistic)

# Spearman's rank correlation
global_spearman = spearmanr(merged['human_rank_pos'], merged['synthetic_rank_pos'])
print("Global Spearman’s rho:", global_spearman.correlation)

"""
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
plt.hexbin(
    merged['human_rank_pos'],
    merged['synthetic_rank_pos'],
    gridsize=50,
    cmap='Blues',
    mincnt=1
)
plt.colorbar(label='Count')
plt.plot([merged['human_rank_pos'].min(), merged['human_rank_pos'].max()],
         [merged['human_rank_pos'].min(), merged['human_rank_pos'].max()],
         color='red', linestyle='--')
plt.title('Human vs Synthetic Ranks (Density)')
plt.xlabel('Human Rank (Position)')
plt.ylabel('Synthetic Rank (Position)')
plt.show()
"""

