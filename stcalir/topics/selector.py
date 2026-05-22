from __future__ import annotations

import random
from pathlib import Path

from stcalir.utils import get_logger, save_jsonl

logger = get_logger(__name__)


def sample_passages_for_topics(
    passages: list[dict],
    n: int = 100,
    seed: int = 42,
    min_tokens: int = 30,
) -> list[dict]:
    """
    Randomly sample n passages from the corpus to serve as seeds for topic creation.
    Filters out very short passages to ensure topics have enough content.
    Returns list of {"pid": str, "text": str} dicts.
    """
    eligible = [
        p for p in passages
        if len(p.get("text", "").split()) >= min_tokens
    ]

    if len(eligible) < n:
        logger.warning(
            f"Only {len(eligible)} eligible passages (>= {min_tokens} tokens); "
            f"requested {n}. Using all eligible passages."
        )
        n = len(eligible)

    random.seed(seed)
    sampled = random.sample(eligible, n)
    logger.info(f"Sampled {len(sampled)} passages for topic creation (seed={seed})")
    return sampled


def save_topic_seeds(passages: list[dict], path: str | Path) -> None:
    """Save sampled passages so the Flask UI can display them."""
    records = [{"pid": p["pid"], "text": p["text"]} for p in passages]
    save_jsonl(records, path)
    logger.info(f"Topic seeds saved → {path}")
