from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from stcir.config import STCIRConfig
from stcir.registry import MODEL_REGISTRY, PREBUILT_FOLDER_MAP
from stcir.utils import (
    Checkpoint, Timer, get_logger,
    load_jsonl, save_jsonl,
    load_run, save_run,
    load_qrels, save_qrels,
)

logger = get_logger(__name__)

Run   = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]
Topic = dict


class PipelineRunner:
    """
    Programmatic, fully-checkpointed runner for the STCIR pipeline.

    Designed for unattended batch execution across multiple datasets.
    Each phase writes a marker file on completion; re-instantiating with
    the same config automatically skips finished phases.

    Parameters
    ----------
    config            : STCIRConfig for the dataset to run
    auto_use_prebuilt : When True, stage1_source / stage2_source are
                        overridden to "prebuilt_*" whenever the registry
                        contains pre-built runs for this dataset — no GPU
                        time is wasted encoding what already exists.
    """

    def __init__(
        self,
        config: STCIRConfig,
        auto_use_prebuilt: bool = True,
        hf_token: Optional[str] = None,
    ):
        import os
        self.config    = config
        self.ckpt      = Checkpoint(config.cache_path("checkpoints"))
        self.hf_token  = hf_token or os.getenv("HF_TOKEN")
        if auto_use_prebuilt:
            self._apply_prebuilt_overrides()

    # ── Prebuilt auto-detection ───────────────────────────────────────────

    def _apply_prebuilt_overrides(self) -> None:
        pb = (
            MODEL_REGISTRY
            .get(self.config.language, {})
            .get("prebuilt", {})
            .get(self.config.dataset, {})
        )
        if not pb:
            return
        if "stage1" in pb and self.config.stage1_source == "computed":
            self.config.stage1_source = pb["stage1"]
            logger.info(f"[{self.config.dataset}] prebuilt stage1 → {self.config.stage1_source}")
        if "stage2" in pb and self.config.stage2_source == "computed":
            self.config.stage2_source = pb["stage2"]
            logger.info(f"[{self.config.dataset}] prebuilt stage2 → {self.config.stage2_source}")

    # ── Phase 1: Corpus ───────────────────────────────────────────────────

    def run_phase1_corpus(self) -> list[dict]:
        corpus_cache = self.config.cache_path("corpus", "passages.jsonl")

        if self.ckpt.is_done("phase1"):
            logger.info(f"[{self.config.dataset}] Phase 1 ✓ (cached)")
            return load_jsonl(corpus_cache)

        if self.config.mode == "domain":
            from stcir.corpus.loader import load_corpus
            from stcir.corpus.chunker import TokenAwareChunker
            raw = load_corpus(self.config.corpus_path)
            chunker = TokenAwareChunker(
                tokenizer_name = self.config.bi_encoders[0],
                max_tokens     = self.config.chunk_max_tokens,
                stride         = self.config.chunk_stride,
                min_tokens     = self.config.chunk_min_tokens,
            )
            passages = chunker.chunk_passages(raw)
        else:
            from stcir.topics.loader import load_corpus_ir
            passages = load_corpus_ir(self.config.dataset)

        save_jsonl(passages, corpus_cache)
        self.ckpt.mark_done("phase1")
        logger.info(f"[{self.config.dataset}] Phase 1 ✓ ({len(passages):,} passages)")
        return passages

    # ── Phase 2: Topics ───────────────────────────────────────────────────

    def run_phase2_topics(self, passages: list[dict]) -> list[Topic]:
        topics_cache = self.config.cache_path("topics", "topics.jsonl")

        if self.ckpt.is_done("phase2"):
            logger.info(f"[{self.config.dataset}] Phase 2 ✓ (cached)")
            from stcir.topics.loader import load_topics_from_file
            return load_topics_from_file(topics_cache)

        if self.config.mode == "standard":
            from stcir.topics.loader import load_topics
            topics = load_topics(self.config.dataset)
        else:
            if self.config.topic_mode != "llm":
                raise RuntimeError(
                    "Batch mode requires topic_mode='llm' for domain datasets. "
                    "The human Flask UI cannot run unattended."
                )
            from stcir.topics.selector import sample_passages_for_topics, save_topic_seeds
            from stcir.annotation.topic_gen import GemmaTopicGenerator

            seeds_path = self.config.cache_path("topics", "seeds.jsonl")
            seeds = sample_passages_for_topics(
                passages, n=self.config.n_topics, seed=self.config.topic_seed
            )
            save_topic_seeds(seeds, seeds_path)
            topics = GemmaTopicGenerator(
                model_name = self.config.llm_model,
                device     = self.config.device,
                language   = self.config.language,
            ).generate(seeds)

        save_jsonl(topics, topics_cache)
        self.ckpt.mark_done("phase2")
        logger.info(f"[{self.config.dataset}] Phase 2 ✓ ({len(topics):,} topics)")
        return topics

    # ── Phase 3: Stage-1 pool ─────────────────────────────────────────────

    def run_phase3_stage1(
        self,
        passages: list[dict],
        topics: list[Topic],
    ) -> Run:
        pool_path = self.config.cache_path("runs", "stage1_pool.tsv")

        if self.ckpt.is_done("phase3c"):
            logger.info(f"[{self.config.dataset}] Phase 3 ✓ (cached)")
            return load_run(pool_path)

        # Pre-built Stage-2 contains the final top-10 directly →
        # Stage-1 pool is not needed; phase 4 will download it.
        if self.config.stage2_source != "computed":
            logger.info(
                f"[{self.config.dataset}] Phase 3 SKIPPED — "
                f"stage2_source='{self.config.stage2_source}' (pre-built Stage-2 available)"
            )
            return {}

        from stcir.retrieval.rrf import Stage1Pool
        stage1 = Stage1Pool(rrf_k=self.config.rrf_k, top_k=self.config.pool_top_k)

        if self.config.stage1_source != "computed":
            folder = PREBUILT_FOLDER_MAP["stage1"][self.config.stage1_source]
            logger.info(f"[{self.config.dataset}] Downloading stage1 prebuilt: {folder}")
            pool = stage1.from_prebuilt(
                hf_repo   = self.config.prebuilt_hf_repo,
                hf_folder = folder,
                cache_dir = str(self.config.cache_path("prebuilt")),
                token     = self.hf_token,
            )
        else:
            bm25_run              = self._run_bm25(passages, topics)
            dense_runs, d_labels  = self._run_dense(passages, topics)
            pool = stage1.from_computed(
                bm25_run   = bm25_run,
                dense_runs = dense_runs,
                run_labels = d_labels,
            )

        save_run(pool, pool_path)
        for phase in ("phase3a", "phase3b", "phase3c"):
            self.ckpt.mark_done(phase)
        logger.info(f"[{self.config.dataset}] Phase 3 ✓ ({len(pool):,} topics in pool)")
        return pool

    def _run_bm25(self, passages: list[dict], topics: list[Topic]) -> Run:
        path = self.config.cache_path("runs", "bm25.tsv")
        if path.exists():
            return load_run(path)

        from stcir.indexing.bm25 import BM25Index
        from stcir.retrieval.bm25_retriever import BM25Retriever

        idx_path = self.config.cache_path("indexes", "bm25.pkl")
        if idx_path.exists():
            idx = BM25Index.load(idx_path)
        else:
            with Timer(f"BM25 index [{self.config.dataset}]"):
                idx = BM25Index().build(passages)
            idx.save(idx_path)

        run = BM25Retriever(idx).retrieve(topics, top_k=self.config.bm25_top_k)
        save_run(run, path)
        return run

    def _run_dense(
        self,
        passages: list[dict],
        topics: list[Topic],
    ) -> tuple[list[Run], list[str]]:
        from stcir.indexing.encoder import BiEncoder
        from stcir.indexing.faiss_index import FaissIndex
        from stcir.retrieval.dense_retriever import DenseRetriever

        runs: list[Run] = []
        labels: list[str] = []

        for model_name in self.config.bi_encoders:
            short     = model_name.split("/")[-1]
            run_path  = self.config.cache_path("runs", f"dense_{short}.tsv")
            idx_path  = self.config.cache_path("indexes", f"{short}.faiss")
            pids_path = self.config.cache_path("indexes", f"{short}.pids.pkl")
            emb_path  = self.config.cache_path("embeddings", f"{short}.npy")

            if run_path.exists():
                runs.append(load_run(run_path))
                labels.append(short)
                continue

            encoder = BiEncoder(model_name, device=self.config.device)

            if emb_path.exists():
                corpus_embs = np.load(emb_path)
            else:
                with Timer(f"Encode corpus [{short}]"):
                    corpus_embs = encoder.encode_corpus(
                        passages, batch_size=self.config.encode_batch_size
                    )
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, corpus_embs)

            if idx_path.exists():
                faiss_idx = FaissIndex.load(idx_path, pids_path)
            else:
                pids = [p["pid"] for p in passages]
                with Timer(f"FAISS [{short}]"):
                    faiss_idx = FaissIndex().build(corpus_embs, pids)
                faiss_idx.save(idx_path, pids_path)

            run = DenseRetriever(encoder, faiss_idx).retrieve(
                topics, top_k=self.config.dense_top_k, batch_size=self.config.encode_batch_size
            )
            save_run(run, run_path)
            runs.append(run)
            labels.append(short)

        return runs, labels

    # ── Phase 4: Stage-2 reranking ────────────────────────────────────────

    def run_phase4_stage2(
        self,
        pool: Run,
        topics: list[Topic],
        passage_lookup: dict[str, str],
    ) -> Run:
        top10_path = self.config.cache_path("runs", "top10_reranked.tsv")

        if self.ckpt.is_done("phase4"):
            logger.info(f"[{self.config.dataset}] Phase 4 ✓ (cached)")
            return load_run(top10_path)

        from stcir.retrieval.rrf import Stage1Pool

        if self.config.stage2_source != "computed":
            folder = PREBUILT_FOLDER_MAP["stage2"][self.config.stage2_source]
            logger.info(f"[{self.config.dataset}] Downloading stage2 prebuilt: {folder}")
            top10_run = Stage1Pool(
                rrf_k=self.config.rrf_k, top_k=self.config.final_top_k
            ).from_prebuilt(
                hf_repo   = self.config.prebuilt_hf_repo,
                hf_folder = folder,
                cache_dir = str(self.config.cache_path("prebuilt")),
                token     = self.hf_token,
            )
        else:
            from stcir.reranking.cross_encoder import CrossEncoderReranker
            from stcir.reranking.rrf_rerank import rerank_with_rrf

            ce_runs: list[Run] = []
            ce_labels: list[str] = []

            for model_name in self.config.cross_encoders:
                short    = model_name.split("/")[-1]
                run_path = self.config.cache_path("runs", f"ce_{short}.tsv")
                if run_path.exists():
                    ce_runs.append(load_run(run_path))
                    ce_labels.append(short)
                    continue
                reranker = CrossEncoderReranker(model_name, device=self.config.device)
                with Timer(f"Rerank [{short}]"):
                    run = reranker.rerank(
                        topics         = topics,
                        pool           = pool,
                        passage_lookup = passage_lookup,
                        top_k          = self.config.rerank_top_k,
                        batch_size     = self.config.rerank_batch_size,
                    )
                save_run(run, run_path)
                ce_runs.append(run)
                ce_labels.append(short)

            top10_run = rerank_with_rrf(
                cross_encoder_runs = ce_runs,
                run_labels         = ce_labels,
                rrf_k              = self.config.rrf_k,
                top_k              = self.config.final_top_k,
            )

        save_run(top10_run, top10_path)
        self.ckpt.mark_done("phase4")
        logger.info(
            f"[{self.config.dataset}] Phase 4 ✓ "
            f"({len(top10_run):,} topics → top-{self.config.final_top_k})"
        )
        return top10_run

    # ── Phase 5: Annotation (LLM only in batch mode) ──────────────────────

    def run_phase5_annotation(
        self,
        top10_run: Run,
        topics: list[Topic],
        passage_lookup: dict[str, str],
    ) -> Qrels:
        qrels_path = self.config.output_path("qrels.tsv")

        if self.ckpt.is_done("phase5"):
            logger.info(f"[{self.config.dataset}] Phase 5 ✓ (cached)")
            return load_qrels(qrels_path)

        if self.config.annotation_mode != "llm":
            raise RuntimeError(
                "Batch runner only supports annotation_mode='llm'. "
                "Set annotation_mode='llm' in your config."
            )

        from stcir.annotation.llm import GemmaAnnotator
        qrels = GemmaAnnotator(
            model_name = self.config.llm_model,
            device     = self.config.device,
            language   = self.config.language,
        ).annotate(
            topics         = topics,
            top10_run      = top10_run,
            passage_lookup = passage_lookup,
            batch_size     = self.config.llm_batch_size,
        )
        save_qrels(qrels, qrels_path)
        self.ckpt.mark_done("phase5")
        n_rel = sum(len(v) for v in qrels.values())
        logger.info(
            f"[{self.config.dataset}] Phase 5 ✓ "
            f"({len(qrels):,} topics, {n_rel:,} relevant judgments)"
        )
        return qrels

    # ── Evaluation ────────────────────────────────────────────────────────

    def run_evaluation(self, qrels: Qrels, system_runs: dict[str, Run]):
        import pandas as pd
        from stcir.evaluation.metrics import evaluate_multiple_runs, hit_at_k

        results_df = evaluate_multiple_runs(qrels, system_runs, self.config.metrics)
        results_df["Hit@10"] = pd.Series(
            {name: hit_at_k(qrels, run, k=10) for name, run in system_runs.items()}
        )
        results_path = self.config.output_path("evaluation_results.csv")
        results_df.to_csv(results_path)
        logger.info(f"[{self.config.dataset}] Evaluation → {results_path}")
        return results_df

    # ── Full pipeline ─────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """
        Execute all phases end-to-end with per-phase checkpointing.
        Completed phases are skipped on re-run — safe to interrupt and resume.
        Returns a dict with keys: config, passages, topics, pool, top10_run, qrels, results.
        """
        cfg = self.config
        logger.info(
            f"\n{'='*60}\n"
            f"  Dataset  : {cfg.dataset}  ({cfg.language})\n"
            f"  Stage-1  : {cfg.stage1_source}\n"
            f"  Stage-2  : {cfg.stage2_source}\n"
            f"  Annotate : {cfg.annotation_mode}\n"
            f"{'='*60}"
        )

        passages       = self.run_phase1_corpus()
        passage_lookup = {p["pid"]: p["text"] for p in passages}
        topics         = self.run_phase2_topics(passages)
        pool           = self.run_phase3_stage1(passages, topics)
        top10_run      = self.run_phase4_stage2(pool, topics, passage_lookup)
        qrels          = self.run_phase5_annotation(top10_run, topics, passage_lookup)
        system_runs    = self._collect_system_runs()
        results        = self.run_evaluation(qrels, system_runs)

        logger.info(f"[{cfg.dataset}] ✅ Pipeline complete")
        return {
            "config":    cfg,
            "passages":  passages,
            "topics":    topics,
            "pool":      pool,
            "top10_run": top10_run,
            "qrels":     qrels,
            "results":   results,
        }

    def _collect_system_runs(self) -> dict[str, Run]:
        runs_dir: Path = self.config.cache_path("runs")
        out: dict[str, Run] = {}

        for label, fname in [("BM25", "bm25.tsv"), ("Pool-RRF", "stage1_pool.tsv"),
                              ("Full-Pipeline", "top10_reranked.tsv")]:
            p = runs_dir / fname
            if p.exists():
                out[label] = load_run(p)

        for m in self.config.bi_encoders:
            s = m.split("/")[-1]
            p = runs_dir / f"dense_{s}.tsv"
            if p.exists():
                out[f"Dense-{s}"] = load_run(p)

        for m in self.config.cross_encoders:
            s = m.split("/")[-1]
            p = runs_dir / f"ce_{s}.tsv"
            if p.exists():
                out[f"CE-{s}"] = load_run(p)

        return out
