from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from kb_loader import DocChunk
from retriever_tfidf import TfidfRetriever
from retriever_vector import VectorRetriever


@dataclass
class HybridHit:
    chunk: DocChunk
    vec_score: float = 0.0
    tfidf_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """
    Merge VectorRetriever and TF-IDF retriever hits.
    """

    def __init__(self):
        self.vec = VectorRetriever()
        self.tfidf = TfidfRetriever()

    def build(self, chunks: List[DocChunk], embeddings: List[List[float]]):
        self.vec.build(chunks, embeddings)
        self.tfidf.build(chunks)

    def search(
        self,
        *,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int = 6,
        vec_top_k: int = 24,
        tfidf_top_k: int = 24,
        w_vec: float = 0.7,
        w_tfidf: float = 0.3,
    ) -> List[HybridHit]:
        by_id: Dict[str, HybridHit] = {}

        if query_embedding is not None:
            for ch, s in self.vec.search(query_embedding, top_k=vec_top_k):
                by_id.setdefault(ch.doc_id, HybridHit(chunk=ch)).vec_score = max(by_id[ch.doc_id].vec_score, s)

        for ch, s in self.tfidf.search(query, top_k=tfidf_top_k):
            if ch.doc_id not in by_id:
                by_id[ch.doc_id] = HybridHit(chunk=ch)
            hit = by_id[ch.doc_id]
            hit.tfidf_score = max(hit.tfidf_score, s)

        hits = list(by_id.values())
        for h in hits:
            h.final_score = w_vec * float(h.vec_score) + w_tfidf * float(h.tfidf_score)

        hits.sort(key=lambda x: x.final_score, reverse=True)
        return hits[:top_k]