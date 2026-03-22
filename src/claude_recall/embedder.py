"""Embedding generation and cross-encoder reranking for semantic search."""

from __future__ import annotations

import numpy as np

_embedder_instance = None
_reranker_instance = None


class Embedder:
    """Wrapper around FastEmbed for generating text embeddings."""

    MODEL = "BAAI/bge-small-en-v1.5"  # 33MB, 384 dimensions, ONNX

    def __init__(self):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self.MODEL)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts. Returns list of numpy arrays."""
        return list(self._model.embed(texts))

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return list(self._model.embed([text]))[0]


class Reranker:
    """Cross-encoder reranker for precise relevance scoring.

    Takes (query, document) pairs and scores them with full
    cross-attention — much more accurate than bi-encoder similarity.
    """

    MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # 80MB, 18ms for 20 docs

    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=self.MODEL)

    def rerank(
        self, query: str, documents: list[str]
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Returns list of (original_index, score) sorted by score descending.
        """
        scores = list(self._model.rerank(query, documents, top_k=len(documents)))
        # scores is a list of floats, one per document in original order
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores


def get_embedder() -> Embedder | None:
    """Get the singleton embedder instance, or None if not available."""
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    try:
        _embedder_instance = Embedder()
        return _embedder_instance
    except ImportError:
        return None
    except Exception:
        return None


def get_reranker(allow_download: bool = False) -> Reranker | None:
    """Get the singleton reranker instance, or None if not available.

    Args:
        allow_download: If True, allow downloading the model on first use.
            If False (default for search), skip if model not cached.
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    if not allow_download:
        # Check if model is already cached before loading
        try:
            from fastembed.common.utils import define_cache_dir

            cache = define_cache_dir()
            # Look for the model in cache
            model_dirs = list(cache.glob("*ms-marco*MiniLM*"))
            if not model_dirs:
                return None  # Not downloaded yet — skip reranking
        except Exception:
            pass

    try:
        _reranker_instance = Reranker()
        return _reranker_instance
    except (ImportError, Exception):
        return None


def ensure_models_downloaded() -> None:
    """Pre-download all models. Called during 'index' command."""
    try:
        get_embedder()
    except Exception:
        pass
    try:
        get_reranker(allow_download=True)
    except Exception:
        pass
