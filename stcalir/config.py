from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from stcalir.registry import MODEL_REGISTRY, PREBUILT_HF_REPO


class STCALIRConfig(BaseModel):
    # ── Identity ────────────────────────────────────────────────────────────
    language: Literal["arabic", "english"] = "arabic"
    dataset: str = "mrtydi_arabic"
    mode: Literal["standard", "domain"] = "standard"

    # ── Paths ───────────────────────────────────────────────────────────────
    corpus_path: Optional[str] = None          # raw corpus for domain mode
    output_dir: str = "outputs/"
    cache_dir: str = ".cache/"

    # ── Chunking (domain mode) ───────────────────────────────────────────────
    chunk_max_tokens: int = 512
    chunk_stride: int = 50
    chunk_min_tokens: int = 20

    # ── Topics (domain mode) ─────────────────────────────────────────────────
    n_topics: int = 100
    topic_mode: Literal["human", "llm"] = "human"
    topic_seed: int = 42

    # ── Stage 1 retrieval ────────────────────────────────────────────────────
    bm25_top_k: int = 1000
    dense_top_k: int = 1000
    pool_top_k: int = 1000
    rrf_k: int = 60

    # ── Stage 1 source ───────────────────────────────────────────────────────
    stage1_source: Literal["computed", "prebuilt_mrtydi", "prebuilt_mmarco"] = "computed"
    prebuilt_hf_repo: str = PREBUILT_HF_REPO

    # ── Stage 2 reranking ────────────────────────────────────────────────────
    rerank_top_k: int = 1000
    final_top_k: int = 10

    # ── Models ───────────────────────────────────────────────────────────────
    bi_encoders: list[str] = Field(default_factory=list)
    cross_encoders: list[str] = Field(default_factory=list)

    # ── Annotation ───────────────────────────────────────────────────────────
    annotation_mode: Literal["human", "llm"] = "human"
    llm_model: str = "google/gemma-3-4b-it"
    llm_batch_size: int = 8
    relevance_scale: Literal["binary", "graded"] = "binary"

    # ── Evaluation ───────────────────────────────────────────────────────────
    metrics: list[str] = Field(
        default_factory=lambda: ["MRR@10", "nDCG@10", "MAP", "Recall@10", "Hit@10", "P@10"]
    )
    qrels_reference: Optional[str] = None  # human qrels for correlation

    # ── Hardware ─────────────────────────────────────────────────────────────
    device: str = "cuda"
    encode_batch_size: int = 64
    rerank_batch_size: int = 32

    # ── UI ───────────────────────────────────────────────────────────────────
    flask_port: int = 5000
    flask_host: str = "0.0.0.0"

    @model_validator(mode="after")
    def _fill_models_from_registry(self) -> "STCALIRConfig":
        registry = MODEL_REGISTRY[self.language]
        if not self.bi_encoders:
            self.bi_encoders = registry["bi_encoders"]
        if not self.cross_encoders:
            self.cross_encoders = registry["cross_encoders"]
        if self.llm_model == "google/gemma-3-4b-it":
            self.llm_model = registry.get("default_llm", "google/gemma-3-4b-it")
        return self

    @model_validator(mode="after")
    def _validate_domain_mode(self) -> "STCALIRConfig":
        if self.mode == "domain" and self.corpus_path is None:
            raise ValueError("corpus_path is required when mode='domain'")
        if self.stage1_source != "computed" and self.language == "english":
            raise ValueError("Pre-built stage1 is only available for Arabic datasets")
        return self

    def output_path(self, *parts: str) -> Path:
        p = Path(self.output_dir, self.dataset, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cache_path(self, *parts: str) -> Path:
        p = Path(self.cache_dir, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def from_yaml(cls, path: str) -> "STCALIRConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
