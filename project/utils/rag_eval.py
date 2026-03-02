import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# 用法：
#   python rag_eval.py eval_queries.jsonl
#
# 环境变量：
#   KB_ROOT 指向你的 knowledge_base 根目录（默认为当前脚本同目录下 knowledge_base）

import rag_service
print("[rag_eval] rag_service file =", rag_service.__file__)
def load_queries(path: str) -> List[Dict[str, Any]]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    return qs

import rag_service
print("[rag_eval] rag_service file =", rag_service.__file__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_eval.py eval_queries.jsonl")
        sys.exit(2)

    q_path = sys.argv[1]
    queries = load_queries(q_path)

    kb_root = os.getenv("KB_ROOT") or os.path.join(os.path.dirname(__file__), "knowledge_base")
    print(f"[rag_eval] KB_ROOT={kb_root}")
    print(f"[rag_eval] queries={len(queries)}")

    # Add code directory to sys.path to allow imports from code/
    sys.path.append(os.path.join(os.path.dirname(__file__), "code"))

    # 直接复用你的 RagService（会自动TF-IDF/混合检索，取决于你是否启用embedding）
    from rag_service import RagService  # noqa

    rag = RagService(kb_root=kb_root)

    # 统计指标
    hit_any = 0
    hit_by_category = Counter()
    missing_compliance_in_topk = 0  # 检查强制合规是否生效
    per_q_results: List[Tuple[str, List[Dict[str, Any]]]] = []

    for q in queries:
        qid = q.get("id")
        text = q.get("query", "")
        expect = set(q.get("expect_any_category") or [])

        hits = rag.retrieve(text, top_k=6)
        per_q_results.append((qid, hits))

        cats = [h.get("category") or "" for h in hits]
        cat_set = set([c for c in cats if c])

        # 命中：top_k 任意一个 category 在 expect 中
        ok = bool(expect & cat_set)
        if ok:
            hit_any += 1
            for c in (expect & cat_set):
                hit_by_category[c] += 1

        # 检查是否包含 compliance
        if "compliance" not in cat_set:
            missing_compliance_in_topk += 1

    # 输出汇总
    total = len(queries)
    print("\n=== Summary ===")
    print(f"Hit@6(any expected category): {hit_any}/{total} = {hit_any/total:.1%}")
    print("Hit breakdown (count of queries where expected category appeared):")
    for k, v in hit_by_category.most_common():
        print(f"  - {k}: {v}")

    print(f"\nCompliance present in top6: {total-missing_compliance_in_topk}/{total} "
          f"= {(total-missing_compliance_in_topk)/total:.1%}")
    if missing_compliance_in_topk > 0:
        print("WARN: Some queries top6 missing compliance. Consider forcing compliance chunks in RagService.")

    # 输出每条 query 的 top hits（便于人工抽查）
    print("\n=== Per-query top hits ===")
    for qid, hits in per_q_results:
        print(f"\n[{qid}]")
        for i, h in enumerate(hits, 1):
            print(
                f"  {i}. doc_id={h.get('doc_id')} | category={h.get('category')} | "
                f"title={h.get('title')} | updated_at={h.get('updated_at')} | compliance={h.get('compliance')}"
            )


if __name__ == "__main__":
    main()