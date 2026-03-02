"""
日更动态知识库（国内）：东方财富 push2 公开接口
- 拉取宽基指数 + 关键ETF 快照
- 生成 knowledge_base/dynamic/YYYY-MM-DD_market_brief.md
- 同时落地 raw: knowledge_base/dynamic/YYYY-MM-DD_market_brief.raw.json（可追溯）
"""
import os
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.getenv("KB_ROOT") or os.path.join(BASE_DIR, "knowledge_base")
KB_DYNAMIC_DIR = os.path.join(KB_ROOT, "dynamic")


def _get_json(url: str, timeout: int = 20, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, Any, str]:
    try:
        resp = requests.get(url, timeout=timeout, headers=headers or {})
        if resp.status_code != 200:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:300]}"
        return True, resp.json(), ""
    except Exception as e:
        return False, None, str(e)


def _em_ulist(secids: str, fields: str) -> Dict[str, Any]:
    """
    东方财富 push2：批量行情
    secids: "1.000001,0.399001,..."
    fields: "f12,f14,f2,f3,f4,f6,..."
    """
    url = (
        "https://push2.eastmoney.com/api/qt/ulist.np/get"
        f"?fltt=2&invt=2&fields={fields}"
        f"&secids={secids}"
    )
    ok, data, err = _get_json(url)
    return {"ok": ok, "url": url, "error": err, "data": data}


def _safe_get_diff(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    inner = data.get("data")
    if not isinstance(inner, dict):
        return []
    diff = inner.get("diff")
    return diff if isinstance(diff, list) else []


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        # 东财有时会返回 "-" 或空字符串
        if isinstance(x, str) and x.strip() in {"", "-", "—"}:
            return None
        return float(x)
    except Exception:
        return None


def _parse_quote_items(diff: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in diff or []:
        out.append(
            {
                "code": it.get("f12"),
                "name": it.get("f14"),
                "last": _to_float(it.get("f2")),   # 最新价
                "pct": _to_float(it.get("f3")),    # 涨跌幅(%)
                "chg": _to_float(it.get("f4")),    # 涨跌额
                "vol": _to_float(it.get("f5")),    # 成交量（手/份，口径随标的）
                "amt": _to_float(it.get("f6")),    # 成交额
            }
        )
    return out


def source_cn_broad_indices() -> Dict[str, Any]:
    """
    中国常见宽基指数：
    - 上证指数 000001 -> 1.000001
    - 深证成指 399001 -> 0.399001
    - 创业板指 399006 -> 0.399006
    - 科创50 000688 -> 1.000688
    - 沪深300 000300 -> 1.000300
    - 中证500 000905 -> 1.000905
    - 中证1000 000852 -> 1.000852
    """
    secids = "1.000001,0.399001,0.399006,1.000688,1.000300,1.000905,1.000852"
    fields = "f12,f14,f2,f3,f4,f5,f6"
    raw = _em_ulist(secids, fields)
    diff = _safe_get_diff(raw.get("data"))
    parsed = _parse_quote_items(diff)
    return {
        "name": "cn_broad_indices",
        "ok": raw["ok"],
        "url": raw["url"],
        "error": raw["error"],
        "parsed": parsed,
        "raw": raw["data"],
    }


def source_cn_key_etfs() -> Dict[str, Any]:
    """
    常见 ETF（示例，按需增删）：
    - 沪深300ETF：510300 (上交所) -> 1.510300
    - 中证500ETF：510500 -> 1.510500
    - 科创50ETF：588000 -> 1.588000
    - 创业板ETF：159915 (深交所) -> 0.159915
    - 红利ETF：510880 -> 1.510880
    - 国债ETF(示例)：511010 -> 1.511010
    """
    secids = "1.510300,1.510500,1.588000,0.159915,1.510880,1.511010"
    fields = "f12,f14,f2,f3,f4,f5,f6"
    raw = _em_ulist(secids, fields)
    diff = _safe_get_diff(raw.get("data"))
    parsed = _parse_quote_items(diff)
    return {
        "name": "cn_key_etfs",
        "ok": raw["ok"],
        "url": raw["url"],
        "error": raw["error"],
        "parsed": parsed,
        "raw": raw["data"],
        "note": "ETF 代码可继续扩充（行业ETF、主题ETF、债券ETF、黄金ETF等）。",
    }


def collect_sources() -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for fn in [source_cn_broad_indices, source_cn_key_etfs]:
        try:
            sources.append(fn())
        except Exception as e:
            sources.append({"name": fn.__name__, "ok": False, "error": str(e), "url": "", "raw": None})
    return sources


def _pick_by_code(items: List[Dict[str, Any]], code: str) -> Optional[Dict[str, Any]]:
    for it in items or []:
        if str(it.get("code") or "") == str(code):
            return it
    return None


def build_digest(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    today = date.today().isoformat()
    digest: Dict[str, Any] = {
        "schema_version": 1,
        "updated_at": today,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cn_market": {},
        "market_state_hint": "unknown",
        "notes": [],
    }

    idx_src = next((s for s in sources if s.get("name") == "cn_broad_indices" and s.get("ok")), None)
    indices = (idx_src or {}).get("parsed") or []
    digest["cn_market"]["indices"] = indices

    # 用 code 精确定位（避免 name 变体）
    hs300 = _pick_by_code(indices, "000300")
    chinext = _pick_by_code(indices, "399006")
    sh = _pick_by_code(indices, "000001")

    pct_list = [x for x in [hs300, chinext, sh] if x and x.get("pct") is not None]
    if pct_list:
        avg = sum([float(x["pct"]) for x in pct_list]) / len(pct_list)
        if avg >= 1.0:
            digest["market_state_hint"] = "risk_on"
        elif avg <= -1.0:
            digest["market_state_hint"] = "risk_off"
        else:
            digest["market_state_hint"] = "neutral"
    else:
        digest["notes"].append("未能计算 market_state_hint（缺少关键指数涨跌幅）。")

    etf_src = next((s for s in sources if s.get("name") == "cn_key_etfs" and s.get("ok")), None)
    digest["cn_market"]["etfs"] = (etf_src or {}).get("parsed") or []

    return digest


def write_daily_files(sources: List[Dict[str, Any]], digest: Dict[str, Any]):
    os.makedirs(KB_DYNAMIC_DIR, exist_ok=True)
    today = date.today().isoformat()
    base_name = f"{today}_market_brief"
    md_path = os.path.join(KB_DYNAMIC_DIR, f"{base_name}.md")
    raw_path = os.path.join(KB_DYNAMIC_DIR, f"{base_name}.raw.json")

    raw_payload = {
        "digest": digest,
        "sources": [
            {
                "name": s.get("name"),
                "ok": s.get("ok"),
                "url": s.get("url"),
                "error": s.get("error"),
                # raw 可能很大，这里保留；如担心体积可只保留 s.get("raw") 的摘要
                "raw": s.get("raw"),
            }
            for s in sources
        ],
    }

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"UPDATED_AT: {today}",
        "TITLE: 每日市场简报（国内）",
        "CATEGORY: dynamic",
        "TAGS: 日更, 指数, ETF, 东方财富, 风险开关",
        "LANG: zh",
        "SCOPE: CN",
        "SOURCE: eastmoney_push2",
        "",
        "## 今日要点（可引用）",
        f"- 抓取时间：{digest.get('fetched_at')}",
        f"- 市场状态提示：{digest.get('market_state_hint')}",
        "",
        "## 指数快照",
    ]

    for it in digest.get("cn_market", {}).get("indices", [])[:20]:
        lines.append(
            f"- {it.get('name')}({it.get('code')}): "
            f"最新={it.get('last')} 涨跌幅(%)={it.get('pct')} 成交额={it.get('amt')}"
        )

    lines += [
        "",
        "## ETF 快照",
    ]
    for it in digest.get("cn_market", {}).get("etfs", [])[:30]:
        lines.append(
            f"- {it.get('name')}({it.get('code')}): "
            f"最新={it.get('last')} 涨跌幅(%)={it.get('pct')} 成交额={it.get('amt')}"
        )

    lines += [
        "",
        "## 原始数据（可追溯）",
        f"- raw 文件：{os.path.basename(raw_path)}",
        "",
        "## 时效口径提示（写给模型）",
        "- risk_off：强调现金/固收缓冲、分批、再平衡、风险预算；避免激进加仓话术。",
        "- risk_on：仍强调分散、回撤管理、核心-卫星；避免承诺收益。",
        "- 输出中应给出风险提示与免责声明；不提供具体个股/买卖点。",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[update_dynamic_kb] wrote: {md_path}")
    print(f"[update_dynamic_kb] wrote: {raw_path}")


def main():
    sources = collect_sources()
    digest = build_digest(sources)
    write_daily_files(sources, digest)


if __name__ == "__main__":
    main()