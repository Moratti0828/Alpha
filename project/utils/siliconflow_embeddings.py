import os
import json
import time
from typing import Dict, List, Optional, Tuple

import requests


class SiliconFlowEmbeddings:
    """
    OpenAI-compatible embeddings client for SiliconFlow.

    Env:
      - SILICONFLOW_API_KEY: required
      - SILICONFLOW_BASE_URL: optional, default "https://api.siliconflow.cn/v1"
      - SILICONFLOW_EMBED_MODEL: optional, default "BAAI/bge-m3"
      - KB_EMBED_CACHE_PATH: optional, default ".cache/kb_embeddings.jsonl"

    Endpoint:
      POST {base_url}/embeddings
      body: {model: "...", input: ["...", "..."]}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        cache_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY for embeddings.")

        self.base_url = (base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1").rstrip("/")
        self.model = model or os.getenv("SILICONFLOW_EMBED_MODEL") or "BAAI/bge-m3"
        self.timeout = timeout

        self.cache_path = cache_path or os.getenv("KB_EMBED_CACHE_PATH") or os.path.join(".cache", "kb_embeddings.jsonl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._cache: Dict[str, List[float]] = {}
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("key")
                    vec = obj.get("embedding")
                    if isinstance(key, str) and isinstance(vec, list):
                        self._cache[key] = vec
        except Exception:
            # cache is best-effort
            pass

    def _append_cache(self, key: str, embedding: List[float]):
        try:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "embedding": embedding}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def embed_texts(
        self,
        texts: List[str],
        *,
        keys: Optional[List[str]] = None,
        batch_size: int = 64,
        retries: int = 2,
    ) -> Tuple[List[List[float]], int]:
        """
        Returns: (embeddings, n_cached)
        - If keys provided, will use local cache for embeddings.
        """
        if keys is not None and len(keys) != len(texts):
            raise ValueError("keys length must match texts length.")

        out: List[Optional[List[float]]] = [None] * len(texts)
        n_cached = 0

        # fill from cache
        if keys is not None:
            for i, k in enumerate(keys):
                if k in self._cache:
                    out[i] = self._cache[k]
                    n_cached += 1

        # prepare uncached indices
        idxs = [i for i, v in enumerate(out) if v is None]
        if not idxs:
            return [v for v in out if v is not None], n_cached

        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # batch call
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_idxs]
            payload = {"model": self.model, "input": batch_texts}

            last_err = None
            for attempt in range(retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("data") or []
                    # OpenAI format: items list has entries with "embedding" and "index"
                    # Some providers may keep the same order; we assume order matches input if index missing.
                    if items and isinstance(items, list):
                        # map by index if present
                        if all(isinstance(it, dict) and "index" in it for it in items):
                            items_sorted = sorted(items, key=lambda x: x.get("index", 0))
                        else:
                            items_sorted = items
                        if len(items_sorted) != len(batch_texts):
                            raise RuntimeError("Embeddings response size mismatch.")
                        for j, it in enumerate(items_sorted):
                            emb = it.get("embedding")
                            if not isinstance(emb, list):
                                raise RuntimeError("Invalid embedding in response.")
                            i = batch_idxs[j]
                            out[i] = emb
                            if keys is not None:
                                k = keys[i]
                                self._cache[k] = emb
                                self._append_cache(k, emb)
                    else:
                        raise RuntimeError("Embeddings response missing data.")
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.2 * (attempt + 1))
            else:
                raise RuntimeError(f"Embeddings call failed after retries: {last_err}")

        return [v for v in out if v is not None], n_cached