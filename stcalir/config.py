from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from stcalir.registry import MODEL_REGISTRY, PREBUILT_HF_REPO

_VALID_LANGUAGES   = ("arabic", "english")
_VALID_MODES       = ("standard", "domain")
_VALID_STAGE1      = ("computed", "prebuilt_mrtydi", "prebuilt_mmarco")
_VALID_ANNOTATION  = ("human", "llm")
_VALID_TOPIC_MODES = ("human", "llm")


@dataclass
class STCALIRConfig:
    # ── Identity ─────────────────────────────────────────────────────────────
    language: str = "arabic"
    dataset:  str = "mrtydi_arabic"
    mode:     str = "standard"

    # ── Paths ─────────────────────────────────────────────────────────────────
    corpus_path: Optional[str] = None
    output_dir:  str = "outputs/"
    cache_dir:   str = ".cache/"

    # ── Chunking (domain mode) ────────────────────────────────────────────────
    chunk_max_tokens: int = 512
    chunk_stride:     int = 50
    chunk_min_tokens: int = 20

    # ── Topics (domain mode) ──────────────────────────────────────────────────
    n_topics:   int = 100
    topic_mode: str = "human"
    topic_seed: int = 42

    # ── Stage 1 retrieval ─────────────────────────────────────────────────────
    bm25_top_k:  int = 1000
    dense_top_k: int = 1000
    pool_top_k:  int = 1000
    rrf_k:       int = 60

    # ── Stage 1 source ────────────────────────────────────────────────────────
    stage1_source:    str = "computed"
    prebuilt_hf_repo: str = PREBUILT_HF_REPO

    # ── Stage 2 reranking ─────────────────────────────────────────────────────
    rerank_top_k: int = 1000
    final_top_k:  int = 10

    # ── Models ────────────────────────────────────────────────────────────────
    bi_encoders:    list = field(default_factory=list)
    cross_encoders: list = field(default_factory=list)

    # ── Annotation ────────────────────────────────────────────────────────────
    annotation_mode:  str = "human"
    llm_model:        str = "google/gemma-3-4b-it"
    llm_batch_size:   int = 8
    relevance_scale:  str = "binary"

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics: list = field(
        default_factory=lambda: ["MRR@10", "nDCG@10", "MAP", "Recall@10", "Hit@10", "P@10"]
    )
    qrels_reference: Optional[str] = None

    # ── Hardware ──────────────────────────────────────────────────────────────
    device:            str = "cuda"
    encode_batch_size: int = 64
    rerank_batch_size: int = 32

    # ── UI ────────────────────────────────────────────────────────────────────
    flask_port: int = 5000
    flask_host: str = "0.0.0.0"

    def __post_init__(self) -> None:
        # ── Validation ───────────────────────────────────────────────────────
        if self.language not in _VALID_LANGUAGES:
            raise ValueError(f"language must be one of {_VALID_LANGUAGES}, got '{self.language}'")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{self.mode}'")
        if self.stage1_source not in _VALID_STAGE1:
            raise ValueError(f"stage1_source must be one of {_VALID_STAGE1}")
        if self.mode == "domain" and self.corpus_path is None:
            raise ValueError("corpus_path is required when mode='domain'")
        if self.stage1_source != "computed" and self.language == "english":
            raise ValueError("Pre-built stage1 is only available for Arabic datasets")

        # ── Auto-fill models from registry ───────────────────────────────────
        registry = MODEL_REGISTRY[self.language]
        if not self.bi_encoders:
            self.bi_encoders = list(registry["bi_encoders"])
        if not self.cross_encoders:
            self.cross_encoders = list(registry["cross_encoders"])
        self.llm_model = registry.get("default_llm", self.llm_model)

    # ── Path helpers ─────────────────────────────────────────────────────────

    def output_path(self, *parts: str) -> Path:
        p = Path(self.output_dir, self.dataset, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cache_path(self, *parts: str) -> Path:
        p = Path(self.cache_dir, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── YAML serialisation ───────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "STCALIRConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, allow_unicode=True)
