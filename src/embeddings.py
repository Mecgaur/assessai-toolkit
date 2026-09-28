"""
Sentence embedding wrapper for AssessAI.

Uses sentence-transformers to encode questions. Keeps a small in-memory
cache so we do not re-embed the same text every time a metric asks for it.

Author: Aditya Gaur
"""

import logging
from typing import Dict, List, Sequence, Union

import numpy as np

# The huggingface/httpx stack logs every HTTP request at INFO level, which
# floods the terminal. Keep only warnings and above from those libraries.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# MiniLM is the default: ~80 MB, runs on CPU, good enough for our use.
# Swap in "all-mpnet-base-v2" for higher quality at ~5x the cost.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Caching wrapper around a sentence-transformers model.

    The model is loaded the first time encode() is called. Subsequent
    calls reuse the loaded model and a per-instance cache of embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info(
            "Model loaded. Embedding dimension: %d",
            self._model.get_embedding_dimension(),
        )

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a string or a list of strings.

        normalize=True (default) makes cosine similarity equal to a dot
        product, which is what the metrics rely on. Pass a single string
        to get a single vector back; pass a list to get a 2D array.
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        if not items:
            if self._model is None:
                self._ensure_model()
            dim = self._model.get_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        # Figure out which items are already cached and only encode the rest.
        to_encode: List[str] = []
        for text in items:
            if text not in self._cache:
                to_encode.append(text)

        if to_encode:
            self._ensure_model()
            new_vectors = self._model.encode(
                to_encode,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            for text, vec in zip(to_encode, new_vectors):
                self._cache[text] = vec.astype(np.float32)

        # Reassemble in the original order.
        result = np.stack([self._cache[t] for t in items])
        return result[0] if single else result

    def clear_cache(self):
        """Drop all cached embeddings."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity for L2-normalized vectors.

    Returns a symmetric (n, n) matrix. If your vectors are not normalized,
    run them through Embedder.encode() first.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {vectors.shape}")
    return vectors @ vectors.T


def cross_similarity(
    queries: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Similarity between every query and every target.

    Used by the coverage metric: questions (queries) against sub-categories
    (targets). Returns (n_queries, n_targets).
    """
    if queries.ndim != 2 or targets.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")
    if queries.shape[1] != targets.shape[1]:
        raise ValueError(
            f"Dimension mismatch: queries have dim {queries.shape[1]}, "
            f"targets have dim {targets.shape[1]}. Both must come from "
            "the same embedding model."
        )
    return queries @ targets.T


# TODO: add optional disk caching (.npy keyed by a hash of model + text)
# so notebooks do not re-embed on every restart. Only needed once we
# start running the validation loop repeatedly.

# NOTE: We are planning an experiment that compares embeddings from
# two different models (MiniLM vs mpnet) to check whether the metrics
# are stable across encoders. The Embedder class already supports this
# by taking model_name as an argument.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "Does your organization have a formalized maintenance strategy?",
        "How is data used in your maintenance workflow?",
        "Is predictive maintenance implemented in your plant?",
    ]

    emb = Embedder()

    print("--- encoding ---")
    vectors = emb.encode(samples)
    print("Shape:", vectors.shape)
    print("Norms:", np.linalg.norm(vectors, axis=1))

    print("\n--- caching check ---")
    emb.encode(samples)
    print("Cache size after two encodes:", len(emb._cache))

    print("\n--- pairwise similarity ---")
    sim = cosine_similarity_matrix(vectors)
    print(np.round(sim, 3))