from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from stcalir.registry import BENCHMARK_LOADERS
from stcalir.utils import get_logger, load_qrels

logger = get_logger(__name__)

Topic = dict   # {"qid": str, "text": str}


def load_topics(dataset: str, split: str = "test") -> list[Topic]:
    """Load topics (queries) for a benchmark dataset."""
    loader_cfg = BENCHMARK_LOADERS.get(dataset)

    if loader_cfg is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(BENCHMARK_LOADERS)}")

    hf_name = loader_cfg["hf_dataset"]
    lang     = loader_cfg.get("lang")
    config   = loader_cfg.get("config")
    _split   = loader_cfg.get("split", split)

    from datasets import load_dataset

    logger.info(f"Loading topics for '{dataset}' from HuggingFace ({hf_name}) ...")

    if hf_name == "castorini/mr-tydi":
        ds = load_dataset(hf_name, lang, split=_split)
        topics = [{"qid": str(row["query_id"]), "text": str(row["query"])} for row in ds]

    elif hf_name == "unicamp-dl/mmarco":
        # mmarco queries
        ds = load_dataset("unicamp-dl/mmarco", f"queries-{lang}", split="train")
        topics = [{"qid": str(row["id"]), "text": str(row["query"])} for row in ds]

    elif hf_name == "ms_marco":
        ds = load_dataset("ms_marco", config, split=_split)
        topics = [{"qid": str(row["query_id"]), "text": str(row["query"])} for row in ds]

    else:
        raise NotImplementedError(f"No loader implemented for {hf_name}")

    logger.info(f"Loaded {len(topics):,} topics for '{dataset}'")
    return topics


def load_topics_from_file(path: str | Path) -> list[Topic]:
    """Load topics from a local JSONL file: {"qid": ..., "text": ...}"""
    path = Path(path)
    topics: list[Topic] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid  = str(rec.get("qid", rec.get("query_id", rec.get("id", ""))))
            text = str(rec.get("text", rec.get("query", "")))
            topics.append({"qid": qid, "text": text})
    logger.info(f"Loaded {len(topics):,} topics from {path}")
    return topics


def load_reference_qrels(dataset: str, split: str = "test") -> dict[str, dict[str, int]]:
    """Load human-annotated qrels for a benchmark dataset."""
    from datasets import load_dataset
    loader_cfg = BENCHMARK_LOADERS.get(dataset, {})
    hf_name = loader_cfg.get("hf_dataset", "")
    lang    = loader_cfg.get("lang")
    _split  = loader_cfg.get("split", split)

    if hf_name == "castorini/mr-tydi":
        ds = load_dataset(hf_name, lang, split=_split)
        qrels: dict[str, dict[str, int]] = {}
        for row in ds:
            qid = str(row["query_id"])
            for pid in row.get("positive_passages", []):
                qrels.setdefault(qid, {})[str(pid["id"])] = 1
        logger.info(f"Loaded {len(qrels)} qrels for '{dataset}'")
        return qrels

    raise NotImplementedError(f"Reference qrels not implemented for {dataset}")
