from __future__ import annotations

import re
from typing import Optional

from tqdm.auto import tqdm

from stcalir.utils import get_logger

logger = get_logger(__name__)

Qrels = dict[str, dict[str, int]]   # qid → {pid → relevance}


RELEVANCE_PROMPT_EN = (
    "You are a relevance assessor for an information retrieval system.\n"
    "Given the query and the passage below, decide if the passage is relevant to the query.\n\n"
    "Query: {query}\n\n"
    "Passage: {passage}\n\n"
    "Answer with a single word: 'yes' if the passage is relevant, 'no' if it is not relevant.\n"
    "Answer:"
)

RELEVANCE_PROMPT_AR = (
    "أنت مقيِّم صلاحية في نظام استرجاع المعلومات.\n"
    "بناءً على الاستعلام والمقطع التاليين، قرر ما إذا كان المقطع ذا صلة بالاستعلام.\n\n"
    "الاستعلام: {query}\n\n"
    "المقطع: {passage}\n\n"
    "أجب بكلمة واحدة: 'نعم' إذا كان المقطع ذا صلة، 'لا' إذا لم يكن كذلك.\n"
    "الإجابة:"
)

PROMPTS = {"english": RELEVANCE_PROMPT_EN, "arabic": RELEVANCE_PROMPT_AR}
YES_TOKENS = {"yes", "نعم", "relevant", "1", "true"}


class GemmaAnnotator:
    """
    Uses a Gemma model (or any causal LM) to annotate relevance judgments.
    Produces binary qrels: 1 = relevant, 0 = not relevant.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cpu",
        language: str = "arabic",
        max_new_tokens: int = 8,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info(f"Loading LLM annotator: {model_name}")
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.prompt_template = PROMPTS.get(language, RELEVANCE_PROMPT_EN)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        self.model.eval()

    def annotate(
        self,
        topics: list[dict],             # [{"qid": str, "text": str}, ...]
        top10_run: dict[str, list[tuple[str, float]]],  # Phase-4 output
        passage_lookup: dict[str, str],   # pid → text
        batch_size: int = 8,
    ) -> Qrels:
        """Score top-10 candidates per topic and produce binary qrels."""
        topic_map = {str(t["qid"]): t["text"] for t in topics}
        qrels: Qrels = {}

        for qid, candidates in tqdm(top10_run.items(), desc="LLM annotation"):
            query = topic_map.get(qid, "")
            qrels[qid] = {}
            for pid, _ in candidates:
                text = passage_lookup.get(pid, "")
                label = self._judge(query, text)
                qrels[qid][pid] = label

        n_rel = sum(sum(d.values()) for d in qrels.values())
        logger.info(f"LLM annotation done: {len(qrels)} topics, {n_rel} relevant judgments")
        return qrels

    def _judge(self, query: str, passage: str) -> int:
        import torch
        prompt = self.prompt_template.format(query=query, passage=passage[:1000])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        answer = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return 1 if any(tok in answer.lower() for tok in YES_TOKENS) else 0
