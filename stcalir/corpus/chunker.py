from __future__ import annotations

from typing import Optional

from tqdm.auto import tqdm

from stcalir.utils import get_logger

logger = get_logger(__name__)


Passage = dict  # {"pid": str, "doc_id": str, "text": str}


class TokenAwareChunker:
    """
    Splits raw documents into passages that fit within a tokenizer's max length.
    Uses a sliding window with configurable stride and minimum token filter.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_tokens: int = 512,
        stride: int = 50,
        min_tokens: int = 20,
    ):
        from transformers import AutoTokenizer

        self.max_tokens = max_tokens
        self.stride = stride
        self.min_tokens = min_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"TokenAwareChunker: model={tokenizer_name}, max={max_tokens}, stride={stride}")

    def chunk_documents(
        self,
        documents: list[dict],  # {"doc_id": str, "text": str}
        pid_prefix: str = "",
    ) -> list[Passage]:
        """Chunk a list of raw documents into token-bounded passages."""
        passages: list[Passage] = []
        for doc in tqdm(documents, desc="Chunking documents"):
            doc_id = str(doc.get("doc_id", doc.get("id", doc.get("pid", ""))))
            text   = str(doc.get("text", doc.get("contents", "")))
            chunks = self._sliding_window(text)
            for i, chunk_text in enumerate(chunks):
                pid = f"{pid_prefix}{doc_id}#{i}" if pid_prefix else f"{doc_id}#{i}"
                passages.append({"pid": pid, "doc_id": doc_id, "text": chunk_text})

        logger.info(f"Produced {len(passages):,} passages from {len(documents):,} documents")
        return passages

    def chunk_passages(
        self,
        passages: list[dict],  # already-split passages that may still be too long
        pid_prefix: str = "",
    ) -> list[Passage]:
        """Re-chunk passages that exceed max_tokens."""
        result: list[Passage] = []
        for p in tqdm(passages, desc="Re-chunking passages"):
            pid    = str(p.get("pid", p.get("id", "")))
            doc_id = str(p.get("doc_id", pid))
            text   = str(p.get("text", ""))
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= self.max_tokens:
                result.append({"pid": pid, "doc_id": doc_id, "text": text})
            else:
                chunks = self._sliding_window(text)
                for i, chunk_text in enumerate(chunks):
                    new_pid = f"{pid_prefix}{pid}#{i}" if pid_prefix else f"{pid}#{i}"
                    result.append({"pid": new_pid, "doc_id": doc_id, "text": chunk_text})

        logger.info(f"Re-chunked {len(passages):,} → {len(result):,} passages")
        return result

    def _sliding_window(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            if len(chunk_tokens) >= self.min_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += self.max_tokens - self.stride
        return chunks if chunks else [text[: 500]]  # fallback for very short texts
