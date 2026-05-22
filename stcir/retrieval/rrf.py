from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stcir.utils import get_logger, load_run

logger = get_logger(__name__)

Run = dict[str, list[tuple[str, float]]]


# ── Core RRF algorithm ────────────────────────────────────────────────────────

def rrf_fuse(runs: list[Run], k: int = 60, top_k: int = 1000) -> Run:
    """
    Reciprocal Rank Fusion across multiple retrieval runs.
    runs: list of {qid: [(pid, score), ...]}  (rank order assumed by list position)
    """
    all_qids: set[str] = set()
    for run in runs:
        all_qids.update(run.keys())

    fused: Run = {}
    for qid in all_qids:
        rrf_scores: dict[str, float] = defaultdict(float)
        for run in runs:
            ranked = run.get(qid, [])
            for rank, (pid, _score) in enumerate(ranked):
                rrf_scores[pid] += 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = sorted_docs[:top_k]

    return fused


# ── Stage-1 Pool builder ──────────────────────────────────────────────────────

@dataclass
class RunFile:
    model_name: str
    path: Path
    run: Run = field(default_factory=dict)

    def load(self) -> "RunFile":
        self.run = load_run(self.path)
        return self


class Stage1Pool:
    """
    Builds the unified 1000-candidate pool for each topic.

    Two input modes:
      - from_computed : uses runs produced in this session (Cells 3 + 4)
      - from_prebuilt : downloads pre-built run files from HuggingFace
    """

    def __init__(self, rrf_k: int = 60, top_k: int = 1000):
        self.rrf_k  = rrf_k
        self.top_k  = top_k

    # ── mode A: computed this session ────────────────────────────────────────

    def from_computed(
        self,
        bm25_run: Run,
        dense_runs: list[Run],
        run_labels: Optional[list[str]] = None,
    ) -> Run:
        """Fuse BM25 + N dense retrieval runs."""
        all_runs = [bm25_run] + dense_runs
        labels   = ["BM25"] + (run_labels or [f"dense_{i}" for i in range(len(dense_runs))])
        logger.info(f"RRF Stage-1: fusing {len(all_runs)} runs ({labels}) → top {self.top_k}")
        pool = rrf_fuse(all_runs, k=self.rrf_k, top_k=self.top_k)
        logger.info(f"Pool built: {len(pool)} topics, avg {self._avg_pool(pool):.0f} candidates/topic")
        return pool

    # ── mode B: pre-built HuggingFace files ──────────────────────────────────

    def from_prebuilt(
        self,
        hf_repo: str,
        hf_folder: str,   # "FirstStage_mrTydi" | "FirstStage_mmarco"
        cache_dir: str = ".cache/prebuilt",
    ) -> Run:
        """Download pre-built TSV run files from HuggingFace and fuse with RRF."""
        from huggingface_hub import list_repo_files, hf_hub_download

        logger.info(f"Downloading pre-built runs from {hf_repo}/{hf_folder} ...")
        cache_path = Path(cache_dir) / hf_folder
        cache_path.mkdir(parents=True, exist_ok=True)

        # list files in the subfolder
        all_files = list(list_repo_files(hf_repo, repo_type="dataset"))
        folder_files = [f for f in all_files if f.startswith(hf_folder + "/") and f.endswith(".txt")]

        if not folder_files:
            raise FileNotFoundError(
                f"No .txt files found in {hf_folder}/ of {hf_repo}. "
                f"Available files: {all_files[:10]}"
            )

        logger.info(f"Found {len(folder_files)} run files: {[Path(f).name for f in folder_files]}")

        runs: list[Run] = []
        for hf_path in folder_files:
            local = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_path,
                repo_type="dataset",
                local_dir=str(cache_path),
            )
            run = load_run(local)
            runs.append(run)
            logger.info(f"  Loaded {Path(hf_path).name}: {len(run)} queries")

        pool = rrf_fuse(runs, k=self.rrf_k, top_k=self.top_k)
        logger.info(f"Pre-built pool built: {len(pool)} topics, avg {self._avg_pool(pool):.0f} candidates/topic")
        return pool

    @staticmethod
    def _avg_pool(pool: Run) -> float:
        if not pool:
            return 0.0
        return sum(len(v) for v in pool.values()) / len(pool)
