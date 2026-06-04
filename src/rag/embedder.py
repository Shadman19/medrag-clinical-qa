"""
src/rag/embedder.py
-------------------
Handles sentence embedding using HuggingFace sentence-transformers.
"""

import hashlib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

_model = None
_use_fallback = False


def _tfidf_embed(texts, dim=384):
    """
    Lightweight deterministic embedding fallback using character n-grams.
    Used when HuggingFace model cannot be downloaded (no internet).
    Produces consistent vectors — enough for demo/dev without a GPU.
    """
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        text_lower = text.lower()
        # Character trigrams as pseudo-features
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def get_model():
    """Load model once and reuse (singleton pattern)."""
    global _model, _use_fallback
    if _model is None and not _use_fallback:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception:
            print("⚠️  Could not load HuggingFace model (no internet?). Using local fallback embedder.")
            _use_fallback = True
    return _model

def embed_texts(texts: list) -> np.ndarray:
    """
    Embed a list of text strings into dense vectors.
    Uses HuggingFace sentence-transformers when available,
    falls back to a local trigram-based embedder otherwise.
    """
    get_model()
    if _use_fallback:
        return _tfidf_embed(texts)
    return _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    get_model()
    if _use_fallback:
        return _tfidf_embed([query])
    return _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )
