import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import html
import json
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from siliconflow_client import SiliconFlowClient


def esc(s: str) -> str:
    return html.escape(s, quote=True)


DEFAULT_SYSTEM_PROMPT = """
你是一个中文智能投顾助手。你要基于给定的用户画像、风险预测概率、资产配置建议，生成合规、清晰、结构化、可执行的投资建议。
合规要求：
1) 不得承诺收益，不得出现“保证盈利”“稳赚不赔”等表述。
2) 避免给出具体个股/具体买卖点；可以给出资产类别、指数基金、ETF等方向性建议（可泛化，不要编造具体代码）。
3) 必须包含风险提示与免责声明。
4) 输出必须是严格 JSON，字段必须完整，便于程序渲染到HTML。
"""

JSON_SCHEMA_HINT = """
请输出如下 JSON（不要输出额外文本、不要输出Markdown代码块）：
{
  "risk_level": "进取型|稳健型|保守型",
  "summary": "一句话总结（20~40字）",
  "bullet_points": ["要点1", "要点2", "要点3", "要点4"],
  "action_plan": [
    {"title": "资金安排", "content": "..." },
    {"title": "组合执行", "content": "..." },
    {"title": "风险控制", "content": "..." }
  ],
  "disclaimer": "免责声明文本"
}
"""


def ensure_advisor_dir(base_dir: str) -> str:
    advisor_dir = os.path.join(base_dir, "advisor_design")
    os.makedirs(advisor_dir, exist_ok=True)
    return advisor_dir


def _load_optional(
    portfolio_csv: Optional[str],
    predictions_csv: Optional[str],
    advisor_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if portfolio_csv is None:
        candidate = os.path.join(advisor_dir, "portfolio_recs.csv")
        portfolio_csv = candidate if os.path.exists(candidate) else None
    if predictions_csv is None:
        candidate = os.path.join(advisor_dir, "predictions.csv")
        predictions_csv = candidate if os.path.exists(candidate) else None

    df_port = pd.read_csv(portfolio_csv) if portfolio_csv and os.path.exists(portfolio_csv) else None
    df_pred = pd.read_csv(predictions_csv) if predictions_csv and os.path.exists(predictions_csv) else None
    return df_port, df_pred


def _merge_data(profiles: pd.DataFrame, df_port: Optional[pd.DataFrame], df_pred: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = profiles.copy()
    if df_port is not None and {"user_id", "portfolio"}.issubset(df_port.columns):
        out = out.merge(df_port[["user_id", "portfolio"]], on="user_id", how="left")
    if df_pred is not None and "user_id" in df_pred.columns:
        cols = [c for c in ["risk_pred", "risk_prob"] if c in df_pred.columns]
        if cols:
            out = out.merge(df_pred[["user_id"] + cols], on="user_id", how="left")
    return out


def _risk_level_from_prob(prob: float) -> str:
    if prob >= 0.6:
        return "进取型"
    if prob <= 0.4:
        return "保守型"
    return "稳健型"


def _build_user_prompt(row: pd.Series) -> str:
    uid = int(row.get("user_id", 0))

    prob = row.get("risk_prob", None)
    try:
        prob_f = float(prob) if prob is not None and prob == prob else 0.5
    except Exception:
        prob_f = 0.5

    payload = {
        "user_id": uid,
        "profile": {
            "age": row.get("age", None),
            "education": row.get("education", None),
            "income10k": row.get("income10k", None),
            "asset10k": row.get("asset10k", None),
            "debt10k": row.get("debt10k", None),
            "children": row.get("children", None),
            "exp_years": row.get("exp_years", None),
        },
        "model_predictions": {
            "risk_prob": round(prob_f, 4),
            "risk_level_hint": _risk_level_from_prob(prob_f),
            "risk_pred": row.get("risk_pred", None),
        },
        "portfolio_recommendation": row.get("portfolio", None),
        "note": "risk_level_hint 是程序根据风险概率给出的提示，可参考但不必机械照搬。",
    }

    return (
        "你将为以下用户生成投顾建议。\n"
        "输入数据如下（JSON）：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"{JSON_SCHEMA_HINT}\n"
    )


def _render_txt(uid: int, advice: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"用户 {uid} 投顾建议")
    lines.append("=" * 44)
    lines.append(f"风险类型：{advice.get('risk_level','')}")
    lines.append(f"摘要：{advice.get('summary','')}")
    lines.append("")
    lines.append("关键要点：")
    for bp in (advice.get("bullet_points") or [])[:10]:
        lines.append(f"- {bp}")
    lines.append("")
    lines.append("行动计划：")
    for sec in (advice.get("action_plan") or [])[:10]:
        lines.append(f"* {sec.get('title','')}: {sec.get('content','')}")
    lines.append("")
    lines.append(f"免责声明：{advice.get('disclaimer','')}")
    return "\n".join(lines)


def _load_template(template_path: str, default: str) -> str:
    """Load HTML template if exists, otherwise return default.

    We keep it extremely lightweight on purpose: just a file read +
    simple string-return so we don't introduce extra dependencies
    (like Jinja2). Control flow (loops/ifs) stays in Python, the
    template only contains scalar placeholders such as {{USER_ID}}.
    """
    try:
        if template_path and os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        # In case of any IO error, silently fall back to default
        pass
    return default


def _subst_template(tpl: str, mapping: Dict[str, Any]) -> str:
    """Very small placeholder replacement helper.

    The template uses double-brace placeholders like {{KEY}}.
    We deliberately avoid any complex templating engine here.
    Values are converted to str with a best-effort fallback.
    """
    out = tpl
    for k, v in mapping.items():
        placeholder = f"{{{{{k}}}}}"
        try:
            out = out.replace(placeholder, str(v))
        except Exception:
            # In weird edge cases keep going so that one bad field
            # does not break the whole page.
            continue
    return out


def _render_html_card(uid: int, row: pd.Series, advice: Dict[str, Any], advisor_dir: str) -> str:
    # Default inline card template (kept for backward compatibility).
    # If an external template file exists, generate_text_advice will
    # load it and call this function with that content instead.
    prob = row.get("risk_prob", 0.5)
    try:
        prob_f = float(prob)
    except Exception:
        prob_f = 0.5

    risk_level = advice.get("risk_level") or _risk_level_from_prob(prob_f)
    summary = advice.get("summary", "")
    bullets = advice.get("bullet_points", []) or []
    plans = advice.get("action_plan", []) or []
    disclaimer = advice.get("disclaimer", "")

    watchlist = row.get("watchlist", "-")

    bullets_html = "".join(f"<li>{esc(str(b))}</li>" for b in bullets)
    plans_html = "".join(
        f"<div class='action-item'><div class='t'>{esc(str(p.get('title','')))}</div>"
        f"<div class='c'>{esc(str(p.get('content','')))}</div></div>"
        for p in plans
    )

    # Fallback inline card (old style) used if no external template
    # overrides it. The outer page controls card container styling.
    default_tpl = f"""
    <div class='card'>
      <h3>用户 {uid} <span class='badge'>{esc(str(risk_level))}</span></h3>
      <p><strong>关注：</strong>{esc(str(watchlist))}</p>
      <p><strong>风险概率：</strong>{prob_f:.1%}</p>
      <p><strong>摘要：</strong>{esc(str(summary))}</p>
      <div><strong>要点：</strong><ul>{bullets_html}</ul></div>
      <div><strong>行动计划：</strong>{plans_html}</div>
      <div class='disclaimer'>{esc(str(disclaimer))}</div>
    </div>
    """

    card_tpl_path = os.path.join(advisor_dir, "report_card.html")
    tpl = _load_template(card_tpl_path, default_tpl)

    ctx = {
        "USER_ID": uid,
        "RISK_LEVEL": risk_level,
        "RISK_PROB": f"{prob_f:.1%}",
        "SUMMARY": summary,
        "WATCHLIST": watchlist,
        "DISCLAIMER": disclaimer,
        # HTML fragments (already wrapped in appropriate tags)
        "BULLETS_HTML": bullets_html,
        "ACTION_PLAN_HTML": plans_html,
    }
    return _subst_template(tpl, ctx)


def generate_text_advice(
    profiles_csv: str,
    advisor_dir: str,
    *,
    max_users: int = 200,
    portfolio_csv: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    os.makedirs(advisor_dir, exist_ok=True)

    profiles = pd.read_csv(profiles_csv)
    df_port, df_pred = _load_optional(portfolio_csv, predictions_csv, advisor_dir)
    merged = _merge_data(profiles, df_port, df_pred)

    client = SiliconFlowClient(model=model or os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen3-VL-32B-Instruct")

    txt_out = os.path.join(advisor_dir, "advice_all.txt")
    html_out = os.path.join(advisor_dir, "advice_all.html")

    txt_blocks: List[str] = []
    cards: List[str] = []

    n = min(max_users, len(merged))
    for _, row in merged.head(n).iterrows():
        uid = int(row.get("user_id", 0))
        user_prompt = _build_user_prompt(row)

        advice = client.chat_json(
            system_prompt=DEFAULT_SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
            retries=int(os.getenv("LLM_RETRIES", "2")),
        )

        if not isinstance(advice, dict):
            advice = {}

        if not advice.get("risk_level"):
            prob = row.get("risk_prob", 0.5)
            try:
                prob = float(prob)
            except Exception:
                prob = 0.5
            advice["risk_level"] = _risk_level_from_prob(prob)

        advice.setdefault("summary", "")
        advice.setdefault("bullet_points", [])
        advice.setdefault("action_plan", [])
        advice.setdefault("disclaimer", "免责声明：本内容仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。")

        txt_blocks.append(_render_txt(uid, advice))
        cards.append(_render_html_card(uid, row, advice, advisor_dir))

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(txt_blocks))

    # Page template: allow overriding via external HTML file. If
    # no template is provided, fall back to the original inline
    # style so existing workflows still work.
    default_page_tpl = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>智能投顾建议（Qwen · SiliconFlow）</title>
  <style>
    body {{
      font-family: "Microsoft YaHei","SimHei", Arial, sans-serif;
      background: #f0f2f5;
      padding: 20px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .disclaimer {{
      font-size: 12px;
      color: #666;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <h1>智能投顾建议（Qwen · SiliconFlow）</h1>
  <p>共生成 {{N_USERS}} 位用户的建议。</p>
  {{CARDS_HTML}}
</body>
</html>
"""

    page_tpl_path = os.path.join(advisor_dir, "report_base.html")
    page_tpl = _load_template(page_tpl_path, default_page_tpl)

    # 统计不同风险等级数量
    risk_counts = {"aggressive": 0, "balanced": 0, "conservative": 0}
    for card_row in merged.head(n).itertuples(index=False):
        prob = getattr(card_row, "risk_prob", 0.5)
        try:
            prob_f = float(prob)
        except Exception:
            prob_f = 0.5
        lvl = _risk_level_from_prob(prob_f)
        if lvl == "进取型":
            risk_counts["aggressive"] += 1
        elif lvl == "稳健型":
            risk_counts["balanced"] += 1
        else:
            risk_counts["conservative"] += 1

    # 构造示例 payload JSON（前几位用户画像）
    try:
        sample_records = merged.head(min(5, len(merged))).to_dict(orient="records")
        sample_json = json.dumps(sample_records, ensure_ascii=False, indent=2)
    except Exception:
        sample_json = "[]"

    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    page_ctx = {
        "GENERATED_AT": generated_at,
        "N_USERS": len(cards),
        "N_AGGRESSIVE": risk_counts["aggressive"],
        "N_BALANCED": risk_counts["balanced"],
        "N_CONSERVATIVE": risk_counts["conservative"],
        "CARDS_HTML": "".join(cards),
        "SAMPLE_PAYLOAD_JSON": sample_json,
    }
    page = _subst_template(page_tpl, page_ctx)

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(page)

    return txt_out

