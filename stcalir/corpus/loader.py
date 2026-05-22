from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from stcalir.utils import get_logger

logger = get_logger(__name__)


Passage = dict  # {"pid": str, "text": str, "doc_id": Optional[str]}


def load_corpus(path: str | Path, format: Optional[str] = None) -> list[Passage]:
    """Load a corpus from file. Auto-detects format from extension if not given."""
    path = Path(path)
    fmt = format or path.suffix.lstrip(".")

    loaders = {
        "jsonl": _load_jsonl,
        "json":  _load_jsonl,
        "csv":   _load_csv,
        "tsv":   _load_tsv,
        "txt":   _load_txt,
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format '{fmt}'. Supported: {list(loaders)}")

    passages = loader(path)
    logger.info(f"Loaded {len(passages):,} passages from {path}")
    return passages


def load_corpus_from_hf(dataset_name: str, split: str = "train", lang: Optional[str] = None) -> list[Passage]:
    """Load corpus from a HuggingFace dataset."""
    from datasets import load_dataset

    kwargs: dict = {"split": split}
    if lang:
        kwargs["name"] = lang

    logger.info(f"Downloading HF dataset '{dataset_name}' (split={split}, lang={lang})")
    ds = load_dataset(dataset_name, **kwargs)

    passages: list[Passage] = []
    for row in ds:
        pid  = str(row.get("id", row.get("docid", row.get("pid", len(passages)))))
        text = str(row.get("text", row.get("passage", row.get("context", ""))))
        passages.append({"pid": pid, "text": text, "doc_id": pid})

    logger.info(f"Loaded {len(passages):,} passages from HuggingFace")
    return passages


def _load_jsonl(path: Path) -> list[Passage]:
    passages = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid  = str(rec.get("pid", rec.get("id", rec.get("docid", i))))
            text = str(rec.get("text", rec.get("passage", rec.get("contents", ""))))
            passages.append({"pid": pid, "text": text, "doc_id": str(rec.get("doc_id", pid))})
    return passages


def _load_csv(path: Path) -> list[Passage]:
    return _load_delimited(path, delimiter=",")


def _load_tsv(path: Path) -> list[Passage]:
    return _load_delimited(path, delimiter="\t")


def _load_delimited(path: Path, delimiter: str) -> list[Passage]:
    passages = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            pid  = str(row.get("pid", row.get("id", row.get("docid", i))))
            text = str(row.get("text", row.get("passage", row.get("contents", ""))))
            passages.append({"pid": pid, "text": text, "doc_id": str(row.get("doc_id", pid))})
    return passages


def _load_txt(path: Path) -> list[Passage]:
    """One passage per line, auto-generated pids."""
    passages = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                passages.append({"pid": str(i), "text": text, "doc_id": str(i)})
    return passages
