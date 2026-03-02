from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import os
import re

from kb_loader import load_kb, DocChunk, chunk_fingerprint
from retriever_hybrid import HybridRetriever
from siliconflow_embeddings import SiliconFlowEmbeddings


def _days_from_today(iso_date: str) -> Optional[int]:
    if not iso_date:
        return None
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d").date()
        return (datetime.now().date() - d).days
    except Exception:
        return None


def _is_compliance(chunk: DocChunk) -> bool:
    if (chunk.category or "").lower() == "compliance":
        return True
    src = (chunk.source or "").replace("\\", "/").lower()
    return "/compliance/" in src


# 国内常见敏感意图（用于决定合规条数）
_SENSITIVE_RE = re.compile(
    r"(稳赚|保证|必涨|无风险|买点|卖点|代码|个股|内幕|荐股|带单|抄底|翻倍|今天买|明天买|直接买|立刻买|现在买)",
    flags=re.IGNORECASE,
)


def _is_sensitive_query(query: str) -> bool:
    return bool(_SENSITIVE_RE.search(query or ""))


# “市场/时效/今日”类问题：才注入 dynamic
_MARKET_RE = re.compile(
    r"(今天|今日|当前|现在|最新|市场|大盘|指数|波动|回撤|成交额|风险偏好|risk[\s_-]?on|risk[\s_-]?off)",
    flags=re.IGNORECASE,
)


def _is_market_query(q: str) -> bool:
    return bool(_MARKET_RE.search(q or ""))


class RagService:
    def __init__(self, kb_root: str):
        self.kb_root = kb_root
        self.retriever = HybridRetriever()

        self._embedder: Optional[SiliconFlowEmbeddings] = None
        try:
            self._embedder = SiliconFlowEmbeddings()
        except Exception:
            self._embedder = None

        self._chunks: List[DocChunk] = []
        self.refresh()

    def refresh(self):
        chunks = load_kb(self.kb_root)
        self._chunks = chunks

        if self._embedder is not None and chunks:
            keys = [chunk_fingerprint(c) for c in chunks]
            texts = [c.text for c in chunks]
            embs, _ = self._embedder.embed_texts(
                texts,
                keys=keys,
                batch_size=int(os.getenv("KB_EMBED_BATCH", "64")),
            )
            if len(embs) != len(texts):
                self._embedder = None
                self.retriever.tfidf.build(chunks)
            else:
                self.retriever.build(chunks, embs)
        else:
            self.retriever.tfidf.build(chunks)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        if self._embedder is None:
            return None
        embs, _ = self._embedder.embed_texts(
            [query],
            keys=[f"__query__:{hash(query)}"],
            batch_size=1,
        )
        return embs[0] if embs else None

    def _is_dynamic(self, chunk: DocChunk) -> bool:
        if (chunk.category or "").lower() == "dynamic":
            return True
        src = (chunk.source or "").replace("\\", "/").lower()
        return "/dynamic/" in src

    def _pick_latest_dynamic(self, n: int = 1) -> List[DocChunk]:
        dyn = [c for c in self._chunks if self._is_dynamic(c)]
        dyn.sort(key=lambda c: (c.updated_at or "", c.title or ""), reverse=True)
        return dyn[:n]

    def _pick_compliance_chunks(self, n: int) -> List[DocChunk]:
        comps = [c for c in self._chunks if _is_compliance(c)]

        def _rank(c: DocChunk):
            lvl = (c.compliance_level or "").lower()
            lvl_score = 1 if lvl == "high" else 0
            return (lvl_score, c.updated_at or "")

        comps.sort(key=_rank, reverse=True)
        return comps[:n]

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        freshness_boost: float = 0.15,
        # 关键：不要再给 compliance boost（否则它会天然靠前）
        compliance_boost: float = 0.0,
        per_title_limit: int = 2,
        insert_compliance_after: int = 2,
        # dynamic 插入位置策略：你选的是“2”（尽量不打扰主建议）
        insert_dynamic_after: int = 4,
    ) -> List[Dict[str, Any]]:
        sensitive = _is_sensitive_query(query)
        market_query = _is_market_query(query)

        # 非敏感：强制1条合规；敏感：强制2条合规
        force_n = 2 if sensitive else 1

        # 1) 召回候选并重排（多取一些）
        q_emb = self._embed_query(query)
        hits = self.retriever.search(
            query=query,
            query_embedding=q_emb,
            top_k=max(top_k * 4, 24),
            vec_top_k=max(top_k * 6, 36),
            tfidf_top_k=max(top_k * 6, 36),
        )

        scored: List[Tuple[float, float, float, float, DocChunk]] = []
        for h in hits:
            chunk = h.chunk
            days = _days_from_today(chunk.updated_at)
            if days is None:
                freshness = 0.0
            else:
                freshness = max(0.0, min(1.0, (60 - days) / 60))

            is_comp = 1.0 if _is_compliance(chunk) else 0.0
            final = float(h.final_score) + float(freshness_boost) * float(freshness) + float(compliance_boost) * is_comp
            scored.append((final, float(h.final_score), float(freshness), is_comp, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 2) 去重：按 title 限制
        title_count: Dict[str, int] = {}
        picked: List[DocChunk] = []
        for _, __, ___, ____, chunk in scored:
            t = chunk.title or ""
            if title_count.get(t, 0) >= per_title_limit:
                continue
            title_count[t] = title_count.get(t, 0) + 1
            picked.append(chunk)
            if len(picked) >= top_k * 3:
                break

        # 3) 强制插入合规证据（插入式保障，不参与“霸榜”）
        forced_comp = self._pick_compliance_chunks(n=force_n)
        forced_comp_ids = {c.doc_id for c in forced_comp}

        # 4) 主内容候选：
        # 非敏感：把所有 compliance 从主候选中剔除（只保留强制插入那1条）
        # 敏感：允许主候选中出现合规，但最终会限制数量
        if not sensitive:
            main_candidates = [c for c in picked if (not _is_compliance(c)) and (c.doc_id not in forced_comp_ids)]
        else:
            main_candidates = [c for c in picked if c.doc_id not in forced_comp_ids]

        # 5) 拼装：主内容 + 合规插入
        merged: List[DocChunk] = []

        # 5.1 前 N 条最相关主内容
        for c in main_candidates:
            merged.append(c)
            if len(merged) >= insert_compliance_after:
                break

        # 5.2 插入 forced compliance（1或2条）
        for c in forced_comp:
            if len(merged) < top_k:
                merged.append(c)

        # 5.3 补齐剩余主内容
        for c in main_candidates:
            if c in merged:
                continue
            merged.append(c)
            if len(merged) >= top_k:
                break

        # 6) 最终控制 compliance 最大数量（非敏感<=1，敏感<=2）
        max_comp = 2 if sensitive else 1
        comp_count = 0
        final_list: List[DocChunk] = []
        for c in merged:
            if _is_compliance(c):
                if comp_count >= max_comp:
                    continue
                comp_count += 1
            final_list.append(c)
            if len(final_list) >= top_k:
                break

        # 兜底：保证至少有1条 compliance
        if not any(_is_compliance(c) for c in final_list):
            fb = self._pick_compliance_chunks(1)
            if fb:
                if len(final_list) < top_k:
                    final_list.append(fb[0])
                else:
                    final_list[-1] = fb[0]

        # 7) 仅在“市场/今日/指数”等时效问题里插入 dynamic（放在 4~6 位，不打扰主建议）
        if market_query:
            dyn = self._pick_latest_dynamic(1)
            if dyn:
                dyn0 = dyn[0]
                if not any(c.doc_id == dyn0.doc_id for c in final_list):
                    insert_pos = min(max(0, insert_dynamic_after), len(final_list))
                    final_list.insert(insert_pos, dyn0)
                    final_list = final_list[:top_k]

        if os.getenv("RAG_DEBUG") == "1":
            print("[RAG_DEBUG] sensitive =", sensitive)
            print("[RAG_DEBUG] market_query =", market_query)
            print("[RAG_DEBUG] forced_comp =", [c.doc_id for c in forced_comp])
            print("[RAG_DEBUG] final top =", [c.doc_id for c in final_list])

        # 8) 输出
        out: List[Dict[str, Any]] = []
        for chunk in final_list:
            days = _days_from_today(chunk.updated_at)
            if days is None:
                freshness = 0.0
            else:
                freshness = max(0.0, min(1.0, (60 - days) / 60))

            out.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "tags": chunk.tags,
                    "source": chunk.source,
                    "updated_at": chunk.updated_at,
                    "compliance": _is_compliance(chunk),
                    "freshness": round(float(freshness), 4),
                    "text": (chunk.text or "")[:1200],
                }
            )
        return out