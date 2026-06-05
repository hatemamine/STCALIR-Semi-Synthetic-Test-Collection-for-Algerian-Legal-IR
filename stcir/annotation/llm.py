from __future__ import annotations

import re
from typing import Optional

from tqdm.auto import tqdm

from stcir.utils import get_logger

logger = get_logger(__name__)

Qrels = dict[str, dict[str, int]]   # qid → {pid → relevance}

# ── Prompt: select the single best passage from a numbered list ───────────────
# This produces exactly 1 relevant document per query, matching the mr-tydi
# single-positive paradigm. The model replies with just a number (1-N).

SELECT_BEST_PROMPT_EN = (
    "You are a relevance assessor for an information retrieval system.\n"
    "Given the query and {n} candidate passages below, identify the SINGLE passage "
    "that best answers the query.\n\n"
    "Query: {query}\n\n"
    "{passages}\n\n"
    "Reply with only the number of the most relevant passage (1–{n}). "
    "Do not explain.\n"
    "Answer:"
)

SELECT_BEST_PROMPT_AR = (
    "أنت مقيِّم صلاحية في نظام استرجاع المعلومات.\n"
    "من بين {n} مقطع مرشح أدناه، حدد المقطع الوحيد الأكثر صلةً بالاستعلام.\n\n"
    "الاستعلام: {query}\n\n"
    "{passages}\n\n"
    "أجب برقم المقطع الأكثر صلةً فقط (1–{n}). لا تشرح.\n"
    "الإجابة:"
)

PROMPTS = {
    "english": SELECT_BEST_PROMPT_EN,
    "arabic":  SELECT_BEST_PROMPT_AR,
}


class GemmaAnnotator:
    """
    Uses a Gemma model (or any causal LM) to annotate relevance judgments.

    For each query the model is shown all top-k candidate passages at once and
    asked to pick the single best one (1-indexed). This produces exactly ONE
    relevant document per query — matching the mr-tydi single-positive paradigm.

    Passage text is truncated to `passage_chars` characters each so the full
    prompt fits within the model's context window.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        device: str = "cpu",
        language: str = "arabic",
        max_new_tokens: int = 4,
        passage_chars: int = 300,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info(f"Loading LLM annotator: {model_name}")
        self.language       = language
        self.max_new_tokens = max_new_tokens
        self.passage_chars  = passage_chars
        self.prompt_template = PROMPTS.get(language, SELECT_BEST_PROMPT_EN)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        self.model.eval()

    def annotate(
        self,
        topics: list[dict],
        top10_run: dict[str, list[tuple[str, float]]],
        passage_lookup: dict[str, str],
        batch_size: int = 8,          # kept for API compatibility; not used
    ) -> Qrels:
        """
        For each query, ask the model which candidate passage is most relevant.
        Returns qrels with exactly 1 relevant document (score=1) per query.
        """
        topic_map = {str(t["qid"]): t["text"] for t in topics}
        qrels: Qrels = {}

        for qid, candidates in tqdm(top10_run.items(), desc="LLM annotation"):
            query = topic_map.get(str(qid), "")
            pids  = [pid for pid, _ in candidates]
            n     = len(pids)

            passages_text = "\n".join(
                f"{i + 1}. {passage_lookup.get(pid, '')[:self.passage_chars]}"
                for i, pid in enumerate(pids)
            )

            best_idx = self._select_best(query, passages_text, n=n)

            # Only the best document is stored; irrelevant ones are omitted.
            qrels[str(qid)] = {pids[best_idx]: 1}

        n_rel = sum(sum(d.values()) for d in qrels.values())
        logger.info(
            f"LLM annotation done: {len(qrels)} topics, "
            f"{n_rel} relevant judgments ({n_rel / max(len(qrels), 1):.1f} per query)"
        )
        return qrels

    def _select_best(self, query: str, passages_text: str, n: int) -> int:
        """
        Ask the model to pick the single best passage.
        Returns a 0-based index; falls back to 0 on parse failure.
        """
        import torch

        prompt = self.prompt_template.format(
            query=query, passages=passages_text, n=n
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        answer = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        numbers = re.findall(r"\d+", answer)
        if numbers:
            idx = int(numbers[0]) - 1   # model outputs 1-based
            if 0 <= idx < n:
                return idx

        logger.warning(
            f"Could not parse passage number from '{answer}' (n={n}) — defaulting to rank 0"
        )
        return 0
