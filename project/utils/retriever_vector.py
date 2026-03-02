from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from kb_loader import DocChunk


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return mat / denom


class VectorRetriever:
    """
    Simple in-memory cosine-similarity retriever for small KBs.
    """

    def __init__(self):
        self._chunks: List[DocChunk] = []
        self._X: Optional[np.ndarray] = None  # (n, d), normalized

    def build(self, chunks: List[DocChunk], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch.")
        self._chunks = chunks
        X = np.array(embeddings, dtype=np.float32)
        self._X = _l2_normalize(X)

    def search(self, query_embedding: List[float], top_k: int = 6) -> List[Tuple[DocChunk, float]]:
        if not self._chunks or self._X is None:
            return []
        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q = _l2_normalize(q)[0]
        sims = (self._X @ q).astype(float)  # cosine since both normalized
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[int(i)], float(sims[int(i)])) for i in idxs]