from __future__ import annotations

from tqdm.auto import tqdm

from stcir.utils import get_logger

logger = get_logger(__name__)


TOPIC_PROMPT_EN = (
    "You are a search query generator.\n"
    "Given the passage below, write a single natural-language search query that a user might type "
    "to find this passage. The query should be concise (5-15 words) and specific.\n\n"
    "Passage: {passage}\n\n"
    "Query:"
)

TOPIC_PROMPT_AR = (
    "أنت مولّد استعلامات بحث.\n"
    "بناءً على المقطع التالي، اكتب استعلام بحث واحد بلغة طبيعية قد يكتبه مستخدم للعثور على هذا المقطع. "
    "يجب أن يكون الاستعلام مختصراً (5-15 كلمة) ومحدداً.\n\n"
    "المقطع: {passage}\n\n"
    "الاستعلام:"
)

PROMPTS = {"english": TOPIC_PROMPT_EN, "arabic": TOPIC_PROMPT_AR}


class GemmaTopicGenerator:
    """
    Generates search queries from passage seeds using a Gemma causal LM.
    Used in domain mode when topic_mode='llm'.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        device: str = "cpu",
        language: str = "arabic",
        max_new_tokens: int = 40,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info(f"Loading topic generator: {model_name}")
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.prompt_template = PROMPTS.get(language, TOPIC_PROMPT_EN)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        self.model.eval()

    def generate(self, passages: list[dict]) -> list[dict]:
        """
        For each passage seed, generate a query.
        Returns [{"qid": str, "text": str, "source_pid": str}, ...]
        """
        import torch

        topics: list[dict] = []
        for i, p in enumerate(tqdm(passages, desc="Generating topics")):
            pid  = str(p.get("pid", i))
            text = str(p.get("text", ""))
            prompt = self.prompt_template.format(passage=text[:1500])

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            query = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().split("\n")[0].strip()

            topics.append({"qid": str(i), "text": query, "source_pid": pid})

        logger.info(f"Generated {len(topics)} topics via LLM")
        return topics
