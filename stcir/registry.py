from __future__ import annotations

MODEL_REGISTRY: dict[str, dict] = {
    "arabic": {
        "bi_encoders": [
            "hatemestinbejaia/mmarco-Arabic-AraElectra-bi-encoder-KD-v1",
            "hatemestinbejaia/mmarco-Arabic-AraElectra-bi-encoder-NoKD-v1",
            "hatemestinbejaia/mmarco-Arabic-AraDPR-bi-encoder-KD-v1",
            "hatemestinbejaia/mmarco-Arabic-AraDPR-bi-encoder-NoKD-v1",
            "hatemestinbejaia/mmarco-Arabic-mMiniLML-bi-encoder-KD-v1",
            "hatemestinbejaia/mmarco-Arabic-mMiniLML-bi-encoder-NoKD-v1",
        ],
        "cross_encoders": [
            "hatemestinbejaia/mmarco-Arabic-AraElectra-cross-encoder-v1",
            "hatemestinbejaia/mmarco-Arabic-AraBERT-cross-encoder-v1",
            "hatemestinbejaia/mmarco-Arabic-mMiniLML-cross-encoder-v1",
            "hatemestinbejaia/mmarco-Arabic-CAMeL-cross-encoder-v1",
            "hatemestinbejaia/mmarco-Arabic-mBERT-cross-encoder-v1",
        ],
        "benchmarks": ["mrtydi_arabic", "mmarco_arabic", "algerian_legal"],
        "prebuilt": {
            "mrtydi_arabic": "FirstStage_mrTydi",
            "mmarco_arabic": "FirstStage_mmarco",
        },
        "default_llm": "google/gemma-3-4b-it",
    },
    "english": {
        "bi_encoders": [
            "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
            "sentence-transformers/msmarco-MiniLM-L12-cos-v5",
            "sentence-transformers/msmarco-distilbert-cos-v5",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/multi-qa-distilbert-cos-v1",
            "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        ],
        "cross_encoders": [
            "cross-encoder/ms-marco-TinyBERT-L2-v2",
            "cross-encoder/ms-marco-MiniLM-L2-v2",
            "cross-encoder/ms-marco-MiniLM-L4-v2",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "cross-encoder/ms-marco-MiniLM-L12-v2",
            "cross-encoder/ms-marco-electra-base",
        ],
        "benchmarks": ["mrtydi_english", "msmarco", "custom_domain"],
        "prebuilt": {},
        "default_llm": "google/gemma-3-4b-it",
    },
}

# ── ir_datasets IDs ────────────────────────────────────────────────────────────
# corpus_id   : used by docs_iter() to build the full passage lookup
# queries_*   : used by queries_iter() per split
# qrels_*     : same dataset id also has qrels_iter()
IR_DATASETS_MAP: dict[str, dict] = {
    "mrtydi_arabic": {
        "corpus_id":      "mr-tydi/ar",
        "queries_test":   "mr-tydi/ar/test",
        "queries_dev":    "mr-tydi/ar/dev",
        "default_split":  "test",
    },
    "mrtydi_english": {
        "corpus_id":      "mr-tydi/en",
        "queries_test":   "mr-tydi/en/test",
        "queries_dev":    "mr-tydi/en/dev",
        "default_split":  "test",
    },
    "mmarco_arabic": {
        "corpus_id":      "mmarco/v2/ar",
        "queries_dev":    "mmarco/v2/ar/dev/small",
        "queries_train":  "mmarco/v2/ar/train",
        "default_split":  "dev",
    },
    "msmarco": {
        "corpus_id":      "msmarco-passage",
        "queries_dev":    "msmarco-passage/dev/small",
        "queries_train":  "msmarco-passage/train",
        "default_split":  "dev",
    },
}

PREBUILT_HF_REPO = "hatemestinbejaia/ExperimentDATA_knowledge_distillation_vs_fine_tuning"

# ── Primary HuggingFace dataset repos (default loader; ir_datasets is fallback) ──
# Keys per entry:
#   hf_repo        : HuggingFace dataset repo ID
#   queries_split  : split name that contains queries
#   corpus_split   : split name that contains passages
#   qrels_split    : split name that contains qrels (omit → same as queries_split)
#
# mmarco-subset DatasetDict structure:
#   arabic_collection  → pid (int), passage (str)
#   english_collection → pid (int), passage (str)
#   arabic_queries     → qid (int), query (str)
#   english_queries    → qid (int), query (str)
#   qrels              → qid (int), pid (int), relevance (int)
HF_PRIMARY_MAP: dict[str, dict] = {
    "mrtydi_arabic": {
        "hf_repo":       "hatemestinbejaia/mr-tydi-ar-dev",
        "queries_split": "train",   # auto-detected fallback if split missing
        "corpus_split":  "train",
        # qrels_split omitted → same as queries_split
    },
    "mmarco_arabic": {
        "hf_repo":       "hatemestinbejaia/mmarco-subset",
        "queries_split": "arabic_queries",
        "corpus_split":  "arabic_collection",
        "qrels_split":   "qrels",
    },
}


def get_models(language: str) -> dict:
    if language not in MODEL_REGISTRY:
        raise ValueError(f"Unknown language '{language}'. Choose: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[language]


def get_ir_datasets_ids(dataset: str) -> dict:
    if dataset not in IR_DATASETS_MAP:
        return {}
    return IR_DATASETS_MAP[dataset]


def get_hf_primary(dataset: str) -> dict | None:
    """Return HF_PRIMARY_MAP entry if one exists for the given dataset key."""
    return HF_PRIMARY_MAP.get(dataset)
