from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stcir.registry import IR_DATASETS_MAP, HF_PRIMARY_MAP
from stcir.utils import get_logger

logger = get_logger(__name__)

Topic = dict   # {"qid": str, "text": str}
Qrels = dict[str, dict[str, int]]

# ── column-name candidates (tried in order) ───────────────────────────────────
_QID_COLS   = ("qid", "query_id", "id", "question_id")
_QTEXT_COLS = ("query", "text", "question", "query_text")
_PID_COLS   = ("pid", "doc_id", "passage_id", "docid", "id")
_PTEXT_COLS = ("passage", "text", "body", "content", "answer")
_REL_COLS   = ("relevance", "label", "score", "rel")


def _pick(row: dict, candidates: tuple[str, ...], default: str = "") -> str:
    for col in candidates:
        if col in row and row[col] is not None:
            return str(row[col])
    return default


# ── HuggingFace primary loader ─────────────────────────────────────────────────

def _hf_load_split(
    hf_repo: str,
    split: str,
    config: str | None = None,
) -> Any:
    """Load one split from a HuggingFace dataset, with optional named config."""
    from datasets import load_dataset
    kwargs: dict = dict(trust_remote_code=False)
    try:
        if config:
            return load_dataset(hf_repo, config, split=split, **kwargs)
        return load_dataset(hf_repo, split=split, **kwargs)
    except Exception:
        if split != "train":
            if config:
                return load_dataset(hf_repo, config, split="train", **kwargs)
            return load_dataset(hf_repo, split="train", **kwargs)
        raise


def _hf_topics(
    hf_repo: str,
    split: str,
    config: str | None = None,
) -> list[Topic]:
    logger.info(
        f"Loading topics from HuggingFace: {hf_repo}"
        + (f" [{config}]" if config else "")
        + f" (split={split})"
    )
    ds = _hf_load_split(hf_repo, split, config)
    sample = ds[0] if len(ds) > 0 else {}
    topics: list[Topic] = []
    for row in ds:
        qid  = _pick(row, _QID_COLS)
        text = _pick(row, _QTEXT_COLS)
        if qid and text:
            topics.append({"qid": qid, "text": text})
    if not topics:
        raise ValueError(
            f"No topics extracted from {hf_repo}. "
            f"Available columns: {list(sample.keys())}"
        )
    logger.info(f"Loaded {len(topics):,} topics from HuggingFace")
    return topics


def _hf_qrels(
    hf_repo: str,
    qrels_split: str,
    config: str | None = None,
) -> Qrels:
    logger.info(
        f"Loading qrels from HuggingFace: {hf_repo}"
        + (f" [{config}]" if config else "")
        + f" (split={qrels_split})"
    )
    ds = _hf_load_split(hf_repo, qrels_split, config)
    sample = ds[0] if len(ds) > 0 else {}
    qrels: Qrels = {}
    for row in ds:
        qid     = _pick(row, _QID_COLS)
        pid     = _pick(row, _PID_COLS)
        rel_raw = _pick(row, _REL_COLS, default="0")
        try:
            rel = int(float(rel_raw))
        except (ValueError, TypeError):
            rel = 0
        if qid and pid and rel > 0:
            qrels.setdefault(qid, {})[pid] = rel
    if not qrels:
        logger.warning(
            f"No positive relevance judgments found in {hf_repo}/{qrels_split}; "
            f"defaulting all (qid, pid) pairs to rel=1. "
            f"Available columns: {list(sample.keys())}"
        )
        for row in ds:
            qid = _pick(row, _QID_COLS)
            pid = _pick(row, _PID_COLS)
            if qid and pid:
                qrels.setdefault(qid, {})[pid] = 1
    logger.info(f"Loaded qrels for {len(qrels):,} queries from HuggingFace")
    return qrels


def _hf_corpus(
    hf_repo: str,
    split: str,
    config: str | None = None,
) -> list[dict]:
    logger.info(
        f"Loading corpus from HuggingFace: {hf_repo}"
        + (f" [{config}]" if config else "")
        + f" (split={split})"
    )
    ds = _hf_load_split(hf_repo, split, config)
    cols   = list(ds.features.keys())
    sample = ds[0] if len(ds) > 0 else {}

    # Check whether the dataset has recognisable column names.
    # mmarco collections were uploaded without headers so the first data row
    # became the column names (e.g. '0' and a long passage string).  In that
    # case fall back to positional access: col[0]=pid, col[1]=text.
    has_pid  = any(c in sample for c in _PID_COLS)
    has_text = any(c in sample for c in _PTEXT_COLS)

    passages: list[dict] = []

    if has_pid and has_text:
        for row in ds:
            pid  = _pick(row, _PID_COLS)
            text = _pick(row, _PTEXT_COLS)
            if not pid:
                pid = str(len(passages))
            if text:
                passages.append({"pid": pid, "text": text, "doc_id": pid})
    else:
        # Positional fallback
        if len(cols) < 2:
            raise ValueError(
                f"Cannot parse corpus from {hf_repo}: only {len(cols)} column(s) "
                f"and none match known names {_PID_COLS + _PTEXT_COLS}. "
                f"Columns: {cols}"
            )
        pid_col, text_col = cols[0], cols[1]
        logger.warning(
            f"Non-standard corpus columns {cols[:2]} — using positional fallback "
            f"(col[0]={pid_col!r} → pid, col[1]={text_col!r} → text)"
        )
        for i, row in enumerate(ds):
            pid  = str(row[pid_col]) if row[pid_col] is not None else str(i)
            text = str(row[text_col]) if row[text_col] is not None else ""
            if text:
                passages.append({"pid": pid, "text": text, "doc_id": pid})

    if not passages:
        raise ValueError(
            f"No passages extracted from {hf_repo}. "
            f"Available columns: {cols}"
        )
    logger.info(f"Loaded {len(passages):,} passages from HuggingFace")
    return passages


# ── Public API ─────────────────────────────────────────────────────────────────

def load_topics(dataset: str, split: str | None = None) -> list[Topic]:
    """
    Load queries for a benchmark dataset.

    Tries HF_PRIMARY_MAP repo first; falls back to ir_datasets.
    dataset : mrtydi_arabic | mrtydi_english | mmarco_arabic | msmarco
    split   : "test" | "dev" | "train"  (defaults to dataset's default_split)
    """
    hf_cfg = HF_PRIMARY_MAP.get(dataset)
    ids    = IR_DATASETS_MAP.get(dataset)

    if ids is None and hf_cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Known: "
            f"{sorted(set(list(IR_DATASETS_MAP) + list(HF_PRIMARY_MAP)))}\n"
            "For a custom corpus use mode='domain'."
        )

    _split = split or (ids["default_split"] if ids else "train")

    if hf_cfg:
        try:
            return _hf_topics(
                hf_cfg["hf_repo"],
                hf_cfg.get("queries_split", _split),
                hf_cfg.get("queries_config"),
            )
        except Exception as e:
            logger.warning(f"HF primary topics load failed ({e}); falling back to ir_datasets")

    if ids is None:
        raise RuntimeError(f"No ir_datasets entry for '{dataset}' and HF load failed.")

    import ir_datasets
    ir_id = ids.get(f"queries_{_split}")
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
    Load human-annotated qrels for a benchmark dataset.

    Tries HuggingFace primary first, falls back to ir_datasets.
    Returns {qid: {pid: relevance}}.
    """
    hf_cfg = HF_PRIMARY_MAP.get(dataset)
    ids    = IR_DATASETS_MAP.get(dataset)

    if ids is None and hf_cfg is None:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    _split = split or (ids["default_split"] if ids else "train")

    if hf_cfg:
        try:
            return _hf_qrels(
                hf_cfg["hf_repo"],
                hf_cfg.get("qrels_split") or hf_cfg.get("queries_split", _split),
                hf_cfg.get("qrels_config") or hf_cfg.get("queries_config"),
            )
        except Exception as e:
            logger.warning(f"HF primary qrels load failed ({e}); falling back to ir_datasets")

    if ids is None:
        raise RuntimeError(f"No ir_datasets entry for '{dataset}' and HF load failed.")

    import ir_datasets
    ir_id = ids.get(f"queries_{_split}")
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
    Load the full passage corpus for a benchmark dataset.

    Tries HuggingFace primary first, falls back to ir_datasets.
    Returns [{"pid": str, "text": str, "doc_id": str}, ...]
    """
    hf_cfg = HF_PRIMARY_MAP.get(dataset)
    ids    = IR_DATASETS_MAP.get(dataset)

    if ids is None and hf_cfg is None:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    if hf_cfg:
        try:
            return _hf_corpus(
                hf_cfg["hf_repo"],
                hf_cfg.get("corpus_split", "train"),
                hf_cfg.get("corpus_config"),
            )
        except Exception as e:
            logger.warning(f"HF primary corpus load failed ({e}); falling back to ir_datasets")

    if ids is None:
        raise RuntimeError(f"No ir_datasets entry for '{dataset}' and HF load failed.")

    import ir_datasets
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
