from __future__ import annotations

import json
from pathlib import Path

from stcalir.registry import IR_DATASETS_MAP
from stcalir.utils import get_logger

logger = get_logger(__name__)

Topic = dict   # {"qid": str, "text": str}
Qrels = dict[str, dict[str, int]]


def load_topics(dataset: str, split: str | None = None) -> list[Topic]:
    """
    Load queries for a benchmark dataset using ir_datasets.

    dataset : one of mrtydi_arabic / mrtydi_english / mmarco_arabic / msmarco
    split   : "test" | "dev" | "train" (defaults to dataset's default_split)
    """
    import ir_datasets

    ids = IR_DATASETS_MAP.get(dataset)
    if ids is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Known: {list(IR_DATASETS_MAP)}\n"
            "For a custom corpus use mode='domain'."
        )

    _split = split or ids["default_split"]
    ir_id  = ids.get(f"queries_{_split}")
    if ir_id is None:
        available = [k.replace("queries_", "") for k in ids if k.startswith("queries_")]
        raise ValueError(
            f"Split '{_split}' not available for '{dataset}'. Available: {available}"
        )

    logger.info(f"Loading topics via ir_datasets: {ir_id}")
    ds = ir_datasets.load(ir_id)
    topics = [{"qid": str(q.query_id), "text": str(q.text)} for q in ds.queries_iter()]
    logger.info(f"Loaded {len(topics):,} topics ({dataset} / {_split})")
    return topics


def load_reference_qrels(dataset: str, split: str | None = None) -> Qrels:
    """
    Load human-annotated qrels for a benchmark dataset using ir_datasets.
    Returns {qid: {pid: relevance}} — only binary (1) labels kept.
    """
    import ir_datasets

    ids = IR_DATASETS_MAP.get(dataset)
    if ids is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(IR_DATASETS_MAP)}")

    _split = split or ids["default_split"]
    ir_id  = ids.get(f"queries_{_split}")   # qrels live on the same split id
    if ir_id is None:
        raise ValueError(f"Split '{_split}' not available for '{dataset}'")

    logger.info(f"Loading reference qrels via ir_datasets: {ir_id}")
    ds = ir_datasets.load(ir_id)
    qrels: Qrels = {}
    for qrel in ds.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = int(qrel.relevance)
    logger.info(f"Loaded qrels for {len(qrels):,} queries ({dataset} / {_split})")
    return qrels


def load_corpus_ir(dataset: str) -> list[dict]:
    """
    Load the full passage corpus for a benchmark dataset using ir_datasets.
    Returns [{"pid": str, "text": str, "doc_id": str}, ...]
    """
    import ir_datasets

    ids = IR_DATASETS_MAP.get(dataset)
    if ids is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(IR_DATASETS_MAP)}")

    ir_id = ids["corpus_id"]
    logger.info(f"Loading corpus via ir_datasets: {ir_id}")
    ds = ir_datasets.load(ir_id)

    passages = []
    for doc in ds.docs_iter():
        text = getattr(doc, "text", None) or getattr(doc, "body", "")
        pid  = str(doc.doc_id)
        passages.append({"pid": pid, "text": str(text), "doc_id": pid})

    logger.info(f"Loaded {len(passages):,} passages for '{dataset}'")
    return passages


def load_topics_from_file(path: str | Path) -> list[Topic]:
    """Load topics from a local JSONL file: {"qid": ..., "text": ...}"""
    path = Path(path)
    topics: list[Topic] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec  = json.loads(line)
            qid  = str(rec.get("qid", rec.get("query_id", rec.get("id", ""))))
            text = str(rec.get("text", rec.get("query", "")))
            topics.append({"qid": qid, "text": text})
    logger.info(f"Loaded {len(topics):,} topics from {path}")
    return topics
