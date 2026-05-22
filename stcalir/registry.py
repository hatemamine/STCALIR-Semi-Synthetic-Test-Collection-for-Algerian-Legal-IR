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

BENCHMARK_LOADERS: dict[str, dict] = {
    "mrtydi_arabic":  {"hf_dataset": "castorini/mr-tydi", "lang": "arabic",  "split": "test"},
    "mrtydi_english": {"hf_dataset": "castorini/mr-tydi", "lang": "english", "split": "test"},
    "mmarco_arabic":  {"hf_dataset": "unicamp-dl/mmarco", "lang": "arabic",  "split": "train"},
    "msmarco":        {"hf_dataset": "ms_marco",          "config": "v1.1",   "split": "validation"},
}

PREBUILT_HF_REPO = "hatemestinbejaia/ExperimentDATA_knowledge_distillation_vs_fine_tuning"


def get_models(language: str) -> dict:
    if language not in MODEL_REGISTRY:
        raise ValueError(f"Unknown language '{language}'. Choose: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[language]


def get_benchmark_loader(dataset: str) -> dict:
    if dataset not in BENCHMARK_LOADERS:
        return {}
    return BENCHMARK_LOADERS[dataset]
