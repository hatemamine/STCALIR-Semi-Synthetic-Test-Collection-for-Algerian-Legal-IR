from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import torch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


def save_jsonl(records: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pickle(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path | str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_run(run: dict[str, list[tuple[str, float]]], path: Path | str) -> None:
    """Save a retrieval run in TREC format: qid  docid  rank  score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            for rank, (pid, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\t{pid}\t{rank}\t{score:.6f}\n")


def load_run(path: Path | str) -> dict[str, list[tuple[str, float]]]:
    """Load TREC-format run file. Handles 4 or 6 column variants."""
    run: dict[str, list[tuple[str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid = parts[0], parts[1]
            score = float(parts[3]) if len(parts) >= 4 else float(len(run.get(qid, [])) * -1)
            run.setdefault(qid, []).append((pid, score))
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run


def save_qrels(qrels: dict[str, dict[str, int]], path: Path | str) -> None:
    """Save qrels in TREC format: qid  0  docid  relevance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for pid, rel in docs.items():
                f.write(f"{qid}\t0\t{pid}\t{rel}\n")


def load_qrels(path: Path | str) -> dict[str, dict[str, int]]:
    """Load TREC qrels file."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[pid] = rel
    return qrels


class Timer:
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logging.info(f"{self.label}: {self.elapsed:.2f}s")


class Checkpoint:
    """Phase-level checkpointing: skip already-completed phases."""

    def __init__(self, checkpoint_dir: Path | str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def is_done(self, phase: str) -> bool:
        return (self.dir / f"{phase}.done").exists()

    def mark_done(self, phase: str) -> None:
        (self.dir / f"{phase}.done").touch()

    def reset(self, phase: str) -> None:
        p = self.dir / f"{phase}.done"
        if p.exists():
            p.unlink()
