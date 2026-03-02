import os
from collections import Counter

from kb_loader import load_kb

kb_root = os.getenv("KB_ROOT") or os.path.join(os.path.dirname(__file__), "knowledge_base")
chunks = load_kb(kb_root)

cats = [c.category for c in chunks]
cnt = Counter(cats)

print(f"KB_ROOT={kb_root}")
print(f"Total chunks={len(chunks)}")
print("Category counts:")
for k, v in cnt.most_common():
    print(f"  - {k}: {v}")

# show a few compliance chunks (if any)
comps = [c for c in chunks if (c.category or "").lower() == "compliance" or "/compliance/" in (c.source or "").replace("\\", "/").lower()]
print(f"\nCompliance chunks found: {len(comps)}")
for c in comps[:10]:
    print(f"  - {c.doc_id} | category={c.category} | source={c.source}")