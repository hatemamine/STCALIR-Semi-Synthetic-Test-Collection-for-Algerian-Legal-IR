from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from stcalir.utils import get_logger

logger = get_logger(__name__)


class BiEncoder:
    """Wraps a SentenceTransformer bi-encoder for corpus and query encoding."""

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"Loading bi-encoder: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def encode_corpus(
        self,
        passages: list[dict],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = [p["text"] for p in passages]
        return self._encode(texts, batch_size, normalize, show_progress, desc="Encoding corpus")

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self._encode(queries, batch_size, normalize, show_progress, desc="Encoding queries")

    def _encode(self, texts, batch_size, normalize, show_progress, desc) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def short_name(self) -> str:
        return self.model_name.split("/")[-1]
