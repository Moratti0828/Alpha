# ✅ 新增文件：retriever_tfidf.py
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kb_loader import DocChunk


class TfidfRetriever:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            max_features=80000,
            ngram_range=(1, 2),
        )
        self._chunks: List[DocChunk] = []
        self._X = None

    def build(self, chunks: List[DocChunk]):
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._X = self._vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int = 6) -> List[Tuple[DocChunk, float]]:
        if not self._chunks or self._X is None:
            return []
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._X)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self._chunks[i], float(sims[i])) for i in idxs]