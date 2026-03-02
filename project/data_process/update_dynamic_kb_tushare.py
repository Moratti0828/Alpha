import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import tushare as ts

from tushare_client import TushareClient

DEFAULT_INDEX_CODES = [
    "000001.SH",  # 上证综指
    "399001.SZ",  # 深证成指
    "399006.SZ",  # 创业板指
    "000688.SH",  # 科创50
    "000300.SH",  # 沪深300
    "000905.SH",  # 中证500
    "000852.SH",  # 中证1000
    "000016.SH",  # 上证50 (A1)
    "000922.CSI", # 中证红利 (A2) —— 如你账号/接口不识别，可改成 000922.SH 再试
]


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _calc_vol(close: pd.Series, window: int) -> Optional[float]:
    r = close.pct_change().dropna()
    if len(r) < window:
        return None
    return float(r.tail(window).std() * (252 ** 0.5))


def _calc_mdd(close: pd.Series, window: int) -> Optional[float]:
    s = close.dropna()
    if len(s) < window:
        return None
    s = s.tail(window)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())


def _infer_risk_state(metrics: List[Dict[str, Any]]) -> str:
    # 非交易建议，仅做“风险偏好提示”
    # 规则：用沪深300/创业板/上证 的涨跌幅均值做分档
    key = {"000300.SH", "399006.SZ", "000001.SH"}
    pcts = []
    for m in metrics:
        if m.get("ts_code") in key and m.get("pct_chg") is not None:
            pcts.append(float(m["pct_chg"]))
    if not pcts:
        return "unknown"
    avg = sum(pcts) / len(pcts)
    if avg >= 1.0:
        return "risk_on"
    if avg <= -1.0:
        return "risk_off"
    return "neutral"


def main():
    # Use TushareClient to handle token loading (env var or file)
    client = TushareClient()
    pro = client.pro

    project_root = Path(__file__).resolve().parent
    kb_root = Path(os.getenv("KB_ROOT") or (project_root / "knowledge_base"))
    dyn_dir = kb_root / "dynamic"
    _ensure_dir(dyn_dir)

    today_iso = _today_iso()
    end = _today_yyyymmdd()

    # lookback：给 120 回撤 + 若干余量
    lookback_days = int(os.getenv("DYN_LOOKBACK_DAYS", "200"))
    start = (datetime.now() - pd.Timedelta(days=lookback_days * 2)).strftime("%Y%m%d")

    metrics: List[Dict[str, Any]] = []
    series_payload: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for code in DEFAULT_INDEX_CODES:
        try:
            df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
            if df is None or df.empty:
                errors.append({"ts_code": code, "error": "empty dataframe"})
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.sort_values("trade_date")
            latest = df.iloc[-1]

            close = df["close"]
            m = {
                "ts_code": code,
                "asof": latest["trade_date"].strftime("%Y-%m-%d"),
                "close": _safe_float(latest.get("close")),
                "pct_chg": _safe_float(latest.get("pct_chg")),  # %
                "amount": _safe_float(latest.get("amount")),
                "vol_20d": _safe_float(_calc_vol(close, 20)),
                "vol_60d": _safe_float(_calc_vol(close, 60)),
                "mdd_60d": _safe_float(_calc_mdd(close, 60)),
                "mdd_120d": _safe_float(_calc_mdd(close, 120)),
            }
            metrics.append(m)

            # 限制 raw 体积
            tail = df.tail(160)[["trade_date", "close", "pct_chg", "amount"]].copy()
            tail["trade_date"] = tail["trade_date"].dt.strftime("%Y-%m-%d")
            series_payload.append({"ts_code": code, "series": tail.to_dict(orient="records")})
        except Exception as e:
            errors.append({"ts_code": code, "error": str(e)})

    metrics = sorted(metrics, key=lambda x: x["ts_code"])
    risk_state = _infer_risk_state(metrics)

    raw = {
        "schema_version": 1,
        "date": today_iso,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "tushare",
        "risk_state": risk_state,
        "indices": {"metrics": metrics, "series": series_payload},
        "errors": errors,
        "notes": [
            "本文件为国内公开数据的日更快照，仅供信息参考，不构成投资建议或收益承诺。",
            "如部分指数代码无数据，请检查 TuShare 账号权限或调整 ts_code（如 000922.CSI vs 000922.SH）。",
        ],
    }

    raw_path = dyn_dir / f"{today_iso}_market_brief.raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    def fmt_pct(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{v:.2f}%"

    def fmt_num(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"

    lines: List[str] = []
    lines += [
        f"UPDATED_AT: {today_iso}",
        "TITLE: 每日市场简报（国内）",
        "CATEGORY: dynamic",
        "TAGS: 日更, 指数, 波动, 回撤, 成交额, 风险开关",
        "LANG: zh",
        "SCOPE: CN",
        "SOURCE: tushare",
        "",
        "## 今日要点（可引用）",
        f"- 抓取时间：{raw['fetched_at']}",
        f"- 风险状态提示：{risk_state}",
        "",
        "## 指数快照（数值）",
    ]

    for m in metrics:
        # mdd 是负数（例如 -0.12），输出为百分比
        mdd60 = None if m.get("mdd_60d") is None else float(m["mdd_60d"]) * 100
        mdd120 = None if m.get("mdd_120d") is None else float(m["mdd_120d"]) * 100
        lines.append(
            f"- {m['ts_code']}：收盘={fmt_num(m.get('close'))}；"
            f"涨跌幅={fmt_pct(m.get('pct_chg'))}；"
            f"20D波动(年化)={fmt_num(m.get('vol_20d'))}；"
            f"60D最大回撤={fmt_pct(mdd60)}；120D最大回撤={fmt_pct(mdd120)}"
        )

    if errors:
        lines += ["", "## 采集异常（如有）"]
        for e in errors[:20]:
            lines.append(f"- {e.get('ts_code')}: {e.get('error')}")

    lines += [
        "",
        "## 原始数据（可追溯）",
        f"- raw 文件：{raw_path.name}",
        "",
        "## 时效口径提示（写给模型）",
        "- risk_off：强调现金/固收缓冲、分批、再平衡、风险预算；避免激进加仓话术。",
        "- risk_on：仍强调分散、回撤管理、核心-卫星；避免承诺收益。",
        "- 输出中应给出风险提示与免责声明；不提供具体个股/买卖点。",
    ]

    md_path = dyn_dir / f"{today_iso}_market_brief.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[update_dynamic_kb_tushare] wrote: {md_path}")
    print(f"[update_dynamic_kb_tushare] wrote: {raw_path}")


if __name__ == "__main__":
    main()