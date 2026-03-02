import time
import hashlib
import json
from typing import Any, Dict, Optional


class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 2048):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, Any] = {}

    def _now(self) -> float:
        return time.time()

    def make_key(self, obj: Any) -> str:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if self._now() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        if len(self._store) >= self.max_items:
            self._store.pop(next(iter(self._store)))
        self._store[key] = (self._now() + self.ttl, value)