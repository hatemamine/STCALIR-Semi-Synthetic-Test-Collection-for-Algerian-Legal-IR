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
            "hatemestinbejaia/mmarco-Arabic-AraElectra-cross-encoder-KD-v1",
            "hatemestinbejaia/mmarco-Arabic-AraDPR-cross-encoder-KD-v1",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ],
        "benchmarks": ["mrtydi_arabic", "mmarco_arabic", "algerian_legal"],
        # Values are stage1_source / stage2_source config strings.
        # PREBUILT_FOLDER_MAP (below) maps those strings → HF folder names.
        "prebuilt": {
            "mrtydi_arabic": {"stage1": "prebuilt_mrtydi", "stage2": "prebuilt_mrtydi"},
            "mmarco_arabic": {"stage1": "prebuilt_mmarco"},
        },
        "default_llm": "google/gemma-4-E4B-it",
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
        "default_llm": "google/gemma-4-E4B-it",
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

PREBUILT_HF_REPO = "hatemestinbejaia/STCIR_Synthetic-Test-Collection-IR"

# Maps stage1_source / stage2_source config values → HF folder names inside PREBUILT_HF_REPO
PREBUILT_FOLDER_MAP: dict[str, dict[str, str]] = {
    "stage1": {
        "prebuilt_mrtydi": "MrTydi_first-stage",
        "prebuilt_mmarco": "FirstStage_mmarco",
    },
    "stage2": {
        "prebuilt_mrtydi": "MrTydi_second_stage",
    },
}

# ── Primary HuggingFace dataset repos (default loader; ir_datasets is fallback) ──
# Keys per entry:
#   hf_repo         : HuggingFace dataset repo ID
#   queries_config  : named dataset config for the queries split
#   queries_split   : split name inside that config (usually "train")
#   corpus_config   : named dataset config for the corpus
#   corpus_split    : split name inside that config
#   qrels_config    : named dataset config for qrels
#   qrels_split     : split name inside that config
#
# Column names are auto-detected; non-standard schemas (e.g. mmarco no-header
# CSV where first row became column names) fall back to positional access:
# col[0]=pid, col[1]=text.
HF_PRIMARY_MAP: dict[str, dict] = {
    "mrtydi_arabic": {
        "hf_repo":        "hatemestinbejaia/mr-tydi-ar-dev",
        "queries_config": "arabic_queries",
        "queries_split":  "train",
        "corpus_config":  "collection",
        "corpus_split":   "train",
        "qrels_config":   "qrels",
        "qrels_split":    "train",
    },
    "mmarco_arabic": {
        "hf_repo":        "hatemestinbejaia/mmarco-arabic-dev",
        "queries_config": "arabic_queries",
        "queries_split":  "train",
        "corpus_config":  "collection",
        "corpus_split":   "train",
        "qrels_config":   "qrels",
        "qrels_split":    "train",
    },
    "msmarco": {
        "hf_repo":        "hatemestinbejaia/mmarco-english-dev",
        "queries_config": "english_queries",
        "queries_split":  "train",
        "corpus_config":  "collection",
        "corpus_split":   "train",
        "qrels_config":   "qrels",
        "qrels_split":    "train",
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
