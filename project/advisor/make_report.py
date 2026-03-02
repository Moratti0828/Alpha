import os
import html
import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from qwen_client import QwenClient


def _risk_level_from_prob(prob: float) -> str:
    if prob > 0.6:
        return "进取型"
    if prob < 0.4:
        return "保守型"
    return "稳健型"


def _advisor_system_prompt() -> str:
    return (
        "你是一个中文智能投顾助手。你要基于给定的用户画像、风险预测概率、用户最近关注的股票/主题（watchlist），"
        "生成合规、清晰、结构化、可执行的投资建议。\n"
        "合规要求：\n"
        "1) 不得承诺收益，不得出现“保证盈利”“稳赚不赔”等表述。\n"
        "2) 避免给出具体个股/具体买卖点；可以给出资产类别、指数基金、ETF等方向性建议（可泛化，不要编造具体代码）。\n"
        "3) 必须包含风险提示与免责声明。\n"
        "4) 输出必须是严格 JSON，字段必须完整，便于程序渲染。\n"
    )


def _advisor_schema_hint() -> str:
    return (
        '请输出如下 JSON（不要输出额外文本、不要输出Markdown代码块）：\n'
        '{\n'
        '  "risk_level": "进取型|稳健型|保守型",\n'
        '  "summary": "一句话总结（20~40字）",\n'
        '  "bullet_points": ["要点1", "要点2", "要点3", "要点4"],\n'
        '  "action_plan": [\n'
        '    {"title": "资金安排", "content": "..." },\n'
        '    {"title": "组合执行", "content": "..." },\n'
        '    {"title": "风险控制", "content": "..." }\n'
        '  ],\n'
        '  "disclaimer": "免责声明文本"\n'
        '}\n'
    )


def _safe_parse_watchlist(val: Any):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            x = json.loads(s)
            if isinstance(x, list):
                return x
            return [str(x)]
        except Exception:
            # 不是 JSON 就当作单字符串
            return [s]
    return [str(val)]


def _normalize_advice(advice: Any, risk_hint: str) -> Dict[str, Any]:
    if not isinstance(advice, dict):
        advice = {}

    advice.setdefault("risk_level", risk_hint)
    advice.setdefault("summary", "")
    advice.setdefault("bullet_points", [])
    advice.setdefault("action_plan", [])
    advice.setdefault("disclaimer", "免责声明：本内容仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。")

    # 保证字段类型
    if not isinstance(advice.get("bullet_points"), list):
        advice["bullet_points"] = []
    if not isinstance(advice.get("action_plan"), list):
        advice["action_plan"] = []

    return advice


def generate_html_report(
    profiles_csv: str,
    predictions_csv: str,
    advisor_dir: str,
    max_users: int = 200,
):
    os.makedirs(advisor_dir, exist_ok=True)

    df = pd.read_csv(profiles_csv)
    pred_df = pd.read_csv(predictions_csv)

    # 合并 risk_prob
    if "risk_prob" in pred_df.columns:
        risk_map = dict(zip(pred_df["user_id"], pred_df["risk_prob"]))
        df["risk_prob"] = df["user_id"].map(risk_map)
    else:
        df["risk_prob"] = 0.5

    df["risk_prob"] = df["risk_prob"].fillna(0.5).astype(float)

    # watchlist 在 M1 已加入；这里仅兜底
    if "watchlist" not in df.columns:
        df["watchlist"] = '["沪深300ETF","中证500ETF","红利ETF"]'

    qwen = QwenClient(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL"),
        model=os.getenv("QWEN_MODEL"),
        timeout=int(os.getenv("QWEN_TIMEOUT", "90")),
    )

    # CSS and Header
    html_parts = [
        "<!doctype html>",
        "<html lang='zh-CN'>",
        "<head>",
        "  <meta charset='utf-8' />",
        "  <meta name='viewport' content='width=device-width,initial-scale=1' />",
        "  <title>AlphaMind | 智能投顾建议</title>",
        "  <style>",
        "    :root{",
        "      --bg: #fbf3ea;",
        "      --card: #ffffff;",
        "      --border: #eadfd3;",
        "      --text: #2b2b2b;",
        "      --muted: #6f6a63;",
        "      --primary: #a31818;",
        "      --shadow: 0 10px 30px rgba(25, 10, 10, .06);",
        "      --radius: 14px;",
        "      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;",
        "      --sans: -apple-system,BlinkMacSystemFont,\"Segoe UI\",\"PingFang SC\",\"Hiragino Sans GB\",\"Microsoft YaHei\",Arial,sans-serif;",
        "    }",
        "    *{ box-sizing: border-box; }",
        "    body{",
        "      margin:0;",
        "      font-family: var(--sans);",
        "      color: var(--text);",
        "      background: radial-gradient(1200px 600px at 20% 0%, #fff8f1 0%, var(--bg) 60%, var(--bg) 100%);",
        "      padding: 24px;",
        "    }",
        "",
        "    .topbar{",
        "      max-width: 1100px;",
        "      margin: 0 auto 18px auto;",
        "      display:flex;",
        "      align-items:center;",
        "      justify-content:space-between;",
        "      padding: 14px 18px;",
        "      background: rgba(255,255,255,.65);",
        "      backdrop-filter: blur(6px);",
        "      border: 1px solid var(--border);",
        "      border-radius: var(--radius);",
        "      box-shadow: var(--shadow);",
        "    }",
        "    .brand{",
        "      display:flex;",
        "      align-items:center;",
        "      gap: 12px;",
        "      min-width: 0;",
        "    }",
        "    .brand-logo{",
        "      width: 36px; height: 36px;",
        "      border-radius: 10px;",
        "      overflow:hidden;",
        "      box-shadow: 0 6px 14px rgba(163,24,24,.18);",
        "      flex: 0 0 auto;",
        "    }",
        "    .brand-logo img{",
        "      width:100%; height:100%; object-fit:cover; display:block;",
        "    }",
        "    .brand-title{ line-height: 1.1; min-width: 0; }",
        "    .brand-title .h1{",
        "      font-weight: 800;",
        "      font-size: 20px;",
        "      color: var(--primary);",
        "      white-space: nowrap;",
        "      overflow: hidden;",
        "      text-overflow: ellipsis;",
        "    }",
        "    .brand-title .sub{",
        "      font-size: 12px;",
        "      color: var(--muted);",
        "      margin-top: 4px;",
        "      white-space: nowrap;",
        "      overflow: hidden;",
        "      text-overflow: ellipsis;",
        "    }",
        "    .badge{",
        "      background: var(--primary);",
        "      color: #fff;",
        "      padding: 4px 10px;",
        "      border-radius: 999px;",
        "      font-size: 12px;",
        "      font-weight: 700;",
        "      box-shadow: 0 8px 18px rgba(163,24,24,.18);",
        "    }",
        "",
        "    .container{",
        "      max-width: 1100px;",
        "      margin: 0 auto;",
        "      display:grid;",
        "      grid-template-columns: 1fr 360px;",
        "      gap: 18px;",
        "      align-items:start;",
        "    }",
        "    @media (max-width: 1000px){",
        "      .container{ grid-template-columns: 1fr; }",
        "    }",
        "",
        "    .card{",
        "      background: var(--card);",
        "      border: 1px solid var(--border);",
        "      border-radius: var(--radius);",
        "      box-shadow: var(--shadow);",
        "      overflow:hidden;",
        "    }",
        "    .card-hd{",
        "      padding: 14px 16px;",
        "      border-bottom: 1px solid #f0e7dc;",
        "      display:flex;",
        "      align-items:center;",
        "      justify-content:space-between;",
        "      gap: 12px;",
        "      background: linear-gradient(180deg, rgba(251,243,234,.55) 0%, rgba(255,255,255,.85) 100%);",
        "    }",
        "    .card-title{",
        "      font-weight: 800;",
        "      color: #3a2a2a;",
        "      display:flex;",
        "      align-items:center;",
        "      gap: 10px;",
        "      min-width:0;",
        "    }",
        "    .dot{",
        "      width: 10px; height: 10px; border-radius: 50%;",
        "      background: var(--primary);",
        "      box-shadow: 0 0 0 4px rgba(163,24,24,.10);",
        "      flex: 0 0 auto;",
        "    }",
        "    .card-bd{ padding: 14px 16px; }",
        "",
        "    /* 顶部摘要 */",
        "    .summary-grid{",
        "      display:grid;",
        "      grid-template-columns: repeat(4, 1fr);",
        "      gap: 12px;",
        "    }",
        "    @media (max-width: 900px){",
        "      .summary-grid{ grid-template-columns: repeat(2, 1fr); }",
        "    }",
        "    .stat{",
        "      border: 1px solid #f0e7dc;",
        "      border-radius: 12px;",
        "      padding: 12px;",
        "      background: #fff;",
        "    }",
        "    .stat .k{ font-size: 12px; color: var(--muted); }",
        "    .stat .v{ font-size: 18px; font-weight: 900; margin-top: 4px; color: #2c1d1d; }",
        "",
        "    /* 用户建议列表容器 */",
        "    .user-grid{",
        "      display:grid;",
        "      grid-template-columns: 1fr;",
        "      gap: 14px;",
        "      padding: 14px;",
        "    }",
        "",
        "    /* 右侧 JSON / 合规卡片复用样式 */",
        "    .codebox{",
        "      font-family: var(--mono);",
        "      font-size: 12px;",
        "      background: #101010;",
        "      color: #eaeaea;",
        "      border-radius: 12px;",
        "      padding: 12px;",
        "      overflow:auto;",
        "      border: 1px solid rgba(255,255,255,.08);",
        "      box-shadow: 0 10px 20px rgba(0,0,0,.08);",
        "      white-space: pre;",
        "    }",
        "    .pill{",
        "      font-size: 12px;",
        "      padding: 4px 10px;",
        "      border-radius: 999px;",
        "      border: 1px solid #f0e7dc;",
        "      background: #fff;",
        "      color: #5c4e4e;",
        "      white-space: nowrap;",
        "    }",
        "    .pill.red{ border-color: rgba(163,24,24,.25); color: var(--primary); background: rgba(163,24,24,.06); }",
        "    .pill.green{ border-color: rgba(39,174,96,.25); color: #1e8449; background: rgba(39,174,96,.08); }",
        "    .btn{",
        "      display:inline-flex;",
        "      align-items:center;",
        "      justify-content:center;",
        "      gap: 8px;",
        "      padding: 10px 12px;",
        "      border-radius: 12px;",
        "      background: var(--primary);",
        "      color:#fff;",
        "      font-weight: 800;",
        "      border: 1px solid rgba(0,0,0,.06);",
        "      text-decoration:none;",
        "      box-shadow: 0 10px 20px rgba(163,24,24,.18);",
        "    }",
        "    .btn.secondary{",
        "      background: #fff;",
        "      color: var(--primary);",
        "      border: 1px solid rgba(163,24,24,.22);",
        "      box-shadow: none;",
        "    }",
        "    .muted{ color: var(--muted); font-size: 12px; }",
        "",
        "    /* Added styles for user card */",
        "    .user-card {",
        "      background: #fff;",
        "      border: 1px solid #f0e7dc;",
        "      border-radius: 12px;",
        "      margin-bottom: 14px;",
        "      overflow: hidden;",
        "    }",
        "    .user-hd {",
        "      padding: 12px 16px;",
        "      background: #fafafa;",
        "      border-bottom: 1px solid #eee;",
        "      display: flex;",
        "      justify-content: space-between;",
        "      align-items: center;",
        "    }",
        "    .user-hd .left {",
        "      display: flex;",
        "      align-items: center;",
        "      gap: 10px;",
        "    }",
        "    .user-id {",
        "      font-weight: bold;",
        "      font-size: 16px;",
        "    }",
        "    .user-bd {",
        "      padding: 16px;",
        "    }",
        "    .kv {",
        "      display: grid;",
        "      grid-template-columns: auto 1fr;",
        "      gap: 8px 16px;",
        "      margin-bottom: 16px;",
        "    }",
        "    .kv .k {",
        "      color: var(--muted);",
        "      font-size: 13px;",
        "      text-align: right;",
        "      min-width: 80px;",
        "    }",
        "    .kv .v {",
        "      font-size: 14px;",
        "    }",
        "    .section {",
        "      margin-top: 16px;",
        "    }",
        "    .sec-title {",
        "      font-weight: bold;",
        "      margin-bottom: 8px;",
        "      font-size: 14px;",
        "      color: var(--primary);",
        "    }",
        "    .actions .act-item {",
        "      margin-bottom: 8px;",
        "      font-size: 14px;",
        "    }",
        "  </style>",
        "</head>",
        "",
        "<body>",
        "  <!-- 顶部栏 -->",
        "  <div class='topbar'>",
        "    <div class='brand'>",
        "      <div class='brand-logo' aria-hidden='true'>",
        "        <img src='alphamind_logo.png' alt='logo' onerror=\"this.style.display='none'\" />",
        "      </div>",
        "      <div class='brand-title'>",
        "        <div class='h1'>AlphaMind 智能投顾建议</div>",
        "        <div class='sub'>画像 × 风险评估 × 资产配置 × 合规输出</div>",
        "      </div>",
        "    </div>",
        "    <div class='badge'>Beta</div>",
        "  </div>",
        "",
        "  <div class='container'>",
        "    <!-- 左侧：建议 -->",
        "    <div class='card'>",
        "      <div class='card-hd'>",
        "        <div class='card-title'><span class='dot'></span>建议总览</div>",
        "        <div class='muted'>生成时间：2025-12-20</div>",
        "      </div>",
        "      <div class='card-bd'>",
        "        <!-- 顶部摘要统计：可选 -->",
        "        <div class='summary-grid'>",
        f"          <div class='stat'><div class='k'>用户数</div><div class='v'>{min(max_users, len(df))}</div></div>",
        "          <div class='stat'><div class='k'>进取型</div><div class='v'>-</div></div>",
        "          <div class='stat'><div class='k'>稳健型</div><div class='v'>-</div></div>",
        "          <div class='stat'><div class='k'>保守型</div><div class='v'>-</div></div>",
        "        </div>",
        "      </div>",
        "",
        "      <!-- 用户卡片列表（由 Python 填充） -->",
        "      <div class='user-grid'>",
    ]

    for _, row in df.head(max_users).iterrows():
        uid = int(row.get("user_id", 0))
        prob = float(row.get("risk_prob", 0.5))
        if np.isnan(prob):
            prob = 0.5

        risk_hint = _risk_level_from_prob(prob)
        watchlist = _safe_parse_watchlist(row.get("watchlist"))

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
            "risk_prediction": {
                "risk_prob": round(float(prob), 4),
                "risk_level_hint": risk_hint,
            },
            "watchlist": watchlist,
            "note": "watchlist 是用户关注标的，不代表推荐；建议需合规且避免具体个股买卖点。",
        }

        user_prompt = (
            "你将为以下用户生成投顾建议。\n"
            "输入数据如下（JSON）：\n"
            f"{payload}\n\n"
            f"{_advisor_schema_hint()}"
        )

        try:
            advice = qwen.chat_json(
                system_prompt=_advisor_system_prompt(),
                user_prompt=user_prompt,
                temperature=float(os.getenv("QWEN_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "900")),
            )
        except Exception as e:
            advice = {
                "risk_level": risk_hint,
                "summary": "模型生成失败，建议以分散配置与风险控制为先。",
                "bullet_points": ["分散配置", "分批/定投", "控制仓位与回撤", "定期复盘再平衡"],
                "action_plan": [
                    {"title": "资金安排", "content": "先预留应急金，再按月定投或分批投入。"},
                    {"title": "组合执行", "content": "以宽基指数/债券基金为核心，主题方向少量卫星配置。"},
                    {"title": "风险控制", "content": "设定最大回撤阈值与仓位上限，避免情绪交易。"},
                ],
                "disclaimer": f"免责声明：本内容仅供参考，不构成投资建议（Qwen调用失败：{e}）。",
            }

        advice = _normalize_advice(advice, risk_hint)

        risk_level = html.escape(str(advice.get("risk_level", risk_hint)))
        summary = html.escape(str(advice.get("summary", "")))
        disclaimer = html.escape(str(advice.get("disclaimer", "")))

        bullets = advice.get("bullet_points", []) or []
        bullets_html = "".join([f"<li>{html.escape(str(x))}</li>" for x in bullets])

        plans = advice.get("action_plan", []) or []
        plans_html = "".join(
            [
                f"<div class='act-item'><strong>{html.escape(str(p.get('title','')))}:</strong> {html.escape(str(p.get('content','')))}</div>"
                for p in plans
                if isinstance(p, dict)
            ]
        )

        html_parts.append(
            f"""
        <div class="user-card">
          <div class="user-hd">
            <div class="left">
              <div class="user-id">用户 {uid}</div>
              <span class="pill ">{risk_level}</span>
              <span class="pill">风险概率：{prob:.1%}</span>
            </div>
            <a class="btn secondary" href="#user-{uid}">定位</a>
          </div>

          <div class="user-bd" id="user-{uid}">
            <div class="kv">
              <div class="k">一句话总结</div><div class="v">{summary}</div>
              <div class="k">用户关注</div><div class="v">{html.escape(str(watchlist))}</div>
              <div class="k">免责声明</div><div class="v">{disclaimer}</div>
            </div>

            <div class="section">
              <div class="sec-title">要点</div>
              <ul>
                {bullets_html}
              </ul>
            </div>

            <div class="section">
              <div class="sec-title">行动计划</div>
              <div class="actions">
                {plans_html}
              </div>
            </div>
          </div>
        </div>
            """
        )

    # Right column and footer
    html_parts.append("""
      </div>
    </div>

    <!-- 右侧：请求数据预览 + 字段映射（示意） -->
    <div>
      <div class="card" style="margin-bottom: 18px;">
        <div class="card-hd">
          <div class="card-title"><span class="dot"></span>请求数据预览</div>
          <span class="pill red">POST /advisor/generate</span>
        </div>
        <div class="card-bd">
          <div class="muted" style="margin-bottom:10px;">
            实际系统对接时，可直接 POST 下列 JSON 结构给后端，后端按模板渲染结构化建议。
          </div>
          <div class="codebox">{
  "users": [
    {
      "id": 0,
      "profile": {
        "age": 28,
        "education": "高中及以下",
        "income10k": 12.5,
        "asset10k": 50.0,
        "debt10k": 20.0,
        "children": 1,
        "exp_years": 3
      },
      "model_predictions": {
        "risk_prob": 0.45,
        "risk_level": "稳健型"
      }
    }
  ]
}</div>
        </div>
      </div>

      <div class="card" style="margin-bottom: 18px;">
        <div class="card-hd">
          <div class="card-title"><span class="dot"></span>字段与模型对应</div>
        </div>
        <div class="card-bd">
          <div class="codebox" style="background:#fff7ef;color:#3b2f2f;border:1px solid #f0e7dc;box-shadow:none;white-space:pre-wrap;">
age → profile.age (int)
education → profile.education (str)
income10k → profile.income10k (float)
asset10k → profile.asset10k (float)
debt10k → profile.debt10k (float)
children → profile.children (int)
exp_years → profile.exp_years (int)
risk_prob → model_predictions.risk_prob (float)
portfolio → portfolio_recommendation (str)
rag_evidence → sources (list)
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-hd">
          <div class="card-title"><span class="dot"></span>合规提示</div>
          <span class="pill green">必读</span>
        </div>
        <div class="card-bd">
          <ul>
            <li>不承诺收益，不出现“保证盈利/稳赚”等表述。</li>
            <li>不提供具体个股/买卖点；可给 ETF/指数基金/资产类别方向。</li>
            <li>每条建议包含风险提示与免责声明。</li>
            <li>sources 至少包含 1 条 compliance 类来源。</li>
          </ul>
          <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
            <a class="btn" href="#top">回到顶部</a>
            <a class="btn secondary" href="javascript:window.print()">打印/导出 PDF</a>
          </div>
        </div>
      </div>
    </div>
  </div>

  <a id="top"></a>
</body>
</html>
    """)

    out_path = os.path.join(advisor_dir, "advice_all.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"✅ HTML 报告已生成: {out_path}")


if __name__ == "__main__":
    # Determine project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    work_dir = os.path.join(project_root, "work")
    advisor_dir = os.path.join(project_root, "advisor_design")

    profiles_path = os.path.join(data_dir, "profiles.csv")
    predictions_path = os.path.join(work_dir, "predictions.csv")

    generate_html_report(profiles_path, predictions_path, advisor_dir, max_users=50)