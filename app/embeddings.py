import os
import re
import hashlib
from functools import lru_cache
import numpy as np


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9+#.]+", text.lower()))


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    """Create semantic embeddings; use a transparent token fallback if disabled."""
    if os.getenv("USE_SENTENCE_TRANSFORMERS", "true").lower() == "true":
        try:
            return _model().encode(text, normalize_embeddings=True).tolist()
        except Exception:
            # Keep the demo usable if model download/runtime is unavailable.
            pass
    # Fixed-size hashed bag-of-words vector keeps dimensions consistent across texts.
    vector = [0.0] * 256
    for token in _tokens(text):
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 256
        vector[index] += 1.0
    return vector


def similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    av, bv = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom else 0.0
