import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Iterable


# Parse metadata header like:
# UPDATED_AT: 2025-12-14
# TITLE: ...
# TAGS: a, b, c
# CATEGORY: compliance
_META_RE = re.compile(r"^([A-Z0-9_]+)\s*:\s*(.*)\s*$")


@dataclass
class DocChunk:
    doc_id: str
    source: str          # file path or URL
    title: str
    updated_at: str      # YYYY-MM-DD
    text: str

    # optional metadata
    category: str = ""
    tags: List[str] = field(default_factory=list)
    lang: str = "zh"
    compliance_level: str = ""  # high|normal|...


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_text(text: str, max_chars: int = 900) -> List[str]:
    # Split by paragraphs; keep chunks short for LLM evidence.
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def iter_kb_files(kb_root: str) -> Iterable[str]:
    for root, _, files in os.walk(kb_root):
        rel_root = os.path.relpath(root, kb_root).replace("\\", "/")
        top = rel_root.split("/")[0].lower() if rel_root and rel_root != "." else ""

        # Exclude internal / accidental nested roots
        if top in {"scripts", ".cache", "__pycache__", "knowledge_base"}:
            continue

        for fn in files:
            if fn.lower().endswith((".md", ".txt")):
                yield os.path.join(root, fn)


def _parse_header_meta(raw: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in raw.splitlines()[:60]:
        m = _META_RE.match(line.strip())
        if not m:
            if meta:
                break
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        # Keep only a whitelist to avoid accidental keys
        if k in {
            "UPDATED_AT",
            "TITLE",
            "TAGS",
            "CATEGORY",
            "LANG",
            "SCOPE",
            "SOURCE",
            "COMPLIANCE_LEVEL",
            "APPLIES_TO",
            "AUDIENCE",
            "VERSION",
        }:
            meta[k] = v
    return meta


def _infer_title(path: str, meta: Dict[str, str]) -> str:
    if meta.get("TITLE"):
        return meta["TITLE"]
    return os.path.splitext(os.path.basename(path))[0]


def _infer_updated_at(raw: str, meta: Dict[str, str]) -> str:
    if meta.get("UPDATED_AT"):
        return meta["UPDATED_AT"]
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("UPDATED_AT:"):
        return lines[0].replace("UPDATED_AT:", "").strip()
    return ""


def _infer_category(path: str, meta: Dict[str, str], kb_root: str) -> str:
    if meta.get("CATEGORY"):
        return meta["CATEGORY"].strip()
    rel = os.path.relpath(path, kb_root)
    parts = rel.split(os.sep)
    return parts[0] if parts else ""


def _parse_tags(meta: Dict[str, str]) -> List[str]:
    raw = (meta.get("TAGS") or "").strip()
    if not raw:
        return []
    parts = re.split(r"[,，]\s*", raw)
    return [p.strip() for p in parts if p.strip()]


def load_kb(kb_root: str) -> List[DocChunk]:
    out: List[DocChunk] = []
    for path in iter_kb_files(kb_root):
        raw = _read_text(path)
        meta = _parse_header_meta(raw)

        title = _infer_title(path, meta)
        updated_at = _infer_updated_at(raw, meta)
        category = _infer_category(path, meta, kb_root)
        tags = _parse_tags(meta)
        lang = (meta.get("LANG") or "zh").strip() or "zh"
        compliance_level = (meta.get("COMPLIANCE_LEVEL") or "").strip()

        chunks = _split_text(raw)
        for i, c in enumerate(chunks):
            meta_prefix = (
                f"[TITLE]{title}\n"
                f"[CATEGORY]{category}\n"
                f"[TAGS]{' '.join(tags)}\n"
                f"[UPDATED_AT]{updated_at}\n\n"
            )
            text = meta_prefix + c

            out.append(
                DocChunk(
                    doc_id=f"{title}#{i}",
                    source=path,
                    title=title,
                    updated_at=updated_at,
                    text=text,
                    category=category,
                    tags=tags,
                    lang=lang,
                    compliance_level=compliance_level,
                )
            )
    return out


def chunk_fingerprint(chunk: DocChunk) -> str:
    """Stable fingerprint for caching embeddings."""
    h = hashlib.sha256()
    h.update((chunk.doc_id or "").encode("utf-8"))
    h.update(b"\n")
    h.update((chunk.updated_at or "").encode("utf-8"))
    h.update(b"\n")
    h.update((chunk.category or "").encode("utf-8"))
    h.update(b"\n")
    h.update((",".join(chunk.tags) if chunk.tags else "").encode("utf-8"))
    h.update(b"\n")
    h.update((chunk.text or "").encode("utf-8"))
    return h.hexdigest()