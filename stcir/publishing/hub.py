from __future__ import annotations

from typing import Optional

from stcir.utils import get_logger

logger = get_logger(__name__)

STCALIR_DATASET_REPO = "hatemestinbejaia/STCALIR-Dataset"


def push_to_hub(
    passages:  list[dict],
    topics:    list[dict],
    qrels:     dict[str, dict[str, int]],
    repo_id:   str = STCALIR_DATASET_REPO,
    dataset:   Optional[str] = None,
    private:   bool = False,
    token:     Optional[str] = None,
) -> str:
    """
    Push collection, topics, and qrels to a HuggingFace dataset repo.

    The repo will contain three named splits:
        collection  → pid (str), text (str)
        topics      → qid (str), text (str)
        qrels       → qid (str), pid (str), relevance (int)

    If the repo already exists, the push appends/overwrites the same splits.
    Pass ``token`` or set the HF_TOKEN environment variable before calling.

    Returns the URL of the published dataset.
    """
    from datasets import Dataset, DatasetDict

    logger.info(f"Building DatasetDict for '{repo_id}' …")

    collection_rows = [
        {"pid": str(p["pid"]), "text": str(p.get("text", ""))}
        for p in passages
    ]

    topics_rows = [
        {"qid": str(t["qid"]), "text": str(t.get("text", ""))}
        for t in topics
    ]

    qrels_rows = [
        {"qid": str(qid), "pid": str(pid), "relevance": int(rel)}
        for qid, docs in qrels.items()
        for pid, rel in docs.items()
    ]

    ds = DatasetDict({
        "collection": Dataset.from_list(collection_rows),
        "topics":     Dataset.from_list(topics_rows),
        "qrels":      Dataset.from_list(qrels_rows),
    })

    logger.info(
        f"  collection : {len(collection_rows):,} passages\n"
        f"  topics     : {len(topics_rows):,} queries\n"
        f"  qrels      : {len(qrels_rows):,} judgments"
    )

    kwargs: dict = dict(repo_id=repo_id, private=private)
    if token:
        kwargs["token"] = token

    ds.push_to_hub(**kwargs)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"✅ Published → {url}")
    return url


def push_batch_to_hub(
    batch_results: dict,
    repo_id:       str = STCALIR_DATASET_REPO,
    private:       bool = False,
    token:         Optional[str] = None,
) -> str:
    """
    Push results from a batch run (dict returned by batch_run.ipynb)
    to HuggingFace. Each dataset is stored as separate splits named
    <dataset>_collection, <dataset>_topics, <dataset>_qrels.

    batch_results : {dataset_key: {passages, topics, qrels, ...}}
    """
    from datasets import Dataset, DatasetDict

    splits: dict = {}
    for ds_key, res in batch_results.items():
        passages = res.get("passages", [])
        topics   = res.get("topics",   [])
        qrels    = res.get("qrels",    {})

        splits[f"{ds_key}_collection"] = Dataset.from_list([
            {"pid": str(p["pid"]), "text": str(p.get("text", ""))} for p in passages
        ])
        splits[f"{ds_key}_topics"] = Dataset.from_list([
            {"qid": str(t["qid"]), "text": str(t.get("text", ""))} for t in topics
        ])
        splits[f"{ds_key}_qrels"] = Dataset.from_list([
            {"qid": str(qid), "pid": str(pid), "relevance": int(rel)}
            for qid, docs in qrels.items()
            for pid, rel in docs.items()
        ])
        logger.info(
            f"[{ds_key}] "
            f"{len(passages):,} passages · {len(topics):,} topics · "
            f"{sum(len(v) for v in qrels.values()):,} qrels"
        )

    ds = DatasetDict(splits)

    kwargs: dict = dict(repo_id=repo_id, private=private)
    if token:
        kwargs["token"] = token

    ds.push_to_hub(**kwargs)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"✅ Published → {url}")
    return url
