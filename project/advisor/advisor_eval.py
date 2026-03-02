"""
离线评测：投顾文本生成（SiliconFlow vs HuggingFace Inference）

输入（建议准备好）：
- profiles_csv: outputs/profiles.csv
- portfolio_csv: work/portfolio_recs.csv   (可选；没有就空)
- predictions_csv: work/predictions.csv 或 work/predictions_hf_xxx.csv

输出：
- results/advisor_llm_eval_{ts}.csv                     (汇总评分)
- results/advisor_llm_outputs_{provider}_{ts}.jsonl      (逐用户原始 JSON)
- results/advisor_llm_errors_{provider}_{ts}.jsonl       (逐用户错误)

用法：
  # 评测 SiliconFlow（你现有）
  set LLM_PROVIDER=siliconflow
  set SILICONFLOW_API_KEY=...
  python advisor_eval.py --predictions_csv work/predictions.csv

  # 评测 Hugging Face Inference
  set LLM_PROVIDER=hf
  set HF_API_TOKEN=hf_xxx
  set HF_API_MODEL=xxx/yyy-instruct
  python advisor_eval.py --predictions_csv work/predictions.csv

  # 评测 HF 风险 baseline + HF LLM（组合对比）
  python advisor_eval.py --predictions_csv work/predictions_hf_macbert.csv
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from cache import TTLCache
from rag_service import RagService

# providers
from siliconflow_client import SiliconFlowClient
from hf_inference_client import HfInferenceClient

# 复用你项目里的风险等级提示
from advisor_llm import _risk_level_from_prob


# 简单敏感词检查（可扩充）
SENSITIVE_RE = re.compile(
    r"(稳赚|保证|必涨|无风险|买点|卖点|代码|个股|内幕|荐股|带单|抄底|翻倍|今天买|明天买|直接买|立刻买|现在买)",
    flags=re.IGNORECASE,
)


def load_portfolio_map(portfolio_csv: Optional[str]) -> Dict[int, str]:
    if not portfolio_csv or not os.path.exists(portfolio_csv):
        return {}
    df = pd.read_csv(portfolio_csv)
    if "user_id" not in df.columns or "portfolio" not in df.columns:
        return {}
    return {int(r["user_id"]): str(r["portfolio"]) for _, r in df.iterrows()}


def load_predictions_map(predictions_csv: str) -> Dict[int, Dict[str, Any]]:
    df = pd.read_csv(predictions_csv)
    required = {"user_id", "risk_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{predictions_csv} 缺少列: {sorted(missing)}")
    out = {}
    for _, r in df.iterrows():
        uid = int(r["user_id"])
        prob = float(r["risk_prob"])
        pred = int(r.get("risk_pred", 1 if prob >= 0.5 else 0))
        out[uid] = {"risk_prob": prob, "risk_pred": pred}
    return out


def advisor_system_prompt() -> str:
    return (
        "你是一个中文智能投顾助手。你要基于给定的用户画像、风险预测概率、资产配置建议，生成合规、清晰、结构化、可执行的投资建议。\n"
        "合规要求：\n"
        "1) 不得承诺收益，不得出现“保证盈利”“稳赚不赔”等表述。\n"
        "2) 避免给出具体个股/具体买卖点；可以给出资产类别、指数基金、ETF等方向性建议（可泛化，不要编造具体代码）。\n"
        "3) 必须包含风险提示与免责声明。\n"
        "4) 输出必须是严格 JSON，字段必须完整，便于程序渲染。\n"
        "5) 你会收到 rag_evidence（检索到的知识依据），请优先参考，并在 sources 字段中列出引用来源。\n"
        "6) sources 至少包含 1 条来自 compliance 类别的来源。\n"
    )


def advisor_schema_hint() -> str:
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
        '  "sources": [{"doc_id":"...", "title":"...", "category":"...", "source":"...", "updated_at":"..."}],\n'
        '  "disclaimer": "免责声明文本"\n'
        '}\n'
    )


def score_one(advice: Dict[str, Any]) -> Dict[str, Any]:
    """对单条建议 JSON 打分（尽量自动化、可解释）。"""
    s = {
        "json_parse_ok": True,
        "has_all_fields": False,
        "sources_has_compliance": False,
        "sensitive_hit": False,
        "bullet_points_len": 0,
        "action_plan_len": 0,
    }

    required = {"risk_level", "summary", "bullet_points", "action_plan", "sources", "disclaimer"}
    s["has_all_fields"] = required.issubset(set(advice.keys()))

    bp = advice.get("bullet_points")
    if isinstance(bp, list):
        s["bullet_points_len"] = len(bp)

    ap = advice.get("action_plan")
    if isinstance(ap, list):
        s["action_plan_len"] = len(ap)

    # sources 至少一条 compliance
    sources = advice.get("sources")
    if isinstance(sources, list):
        for it in sources:
            if isinstance(it, dict) and str(it.get("category") or "").lower() == "compliance":
                s["sources_has_compliance"] = True
                break

    # 敏感词：在 summary + bullet_points + disclaimer 中扫一下
    txt = ""
    txt += str(advice.get("summary") or "") + "\n"
    if isinstance(bp, list):
        txt += "\n".join([str(x) for x in bp]) + "\n"
    txt += str(advice.get("disclaimer") or "")
    s["sensitive_hit"] = bool(SENSITIVE_RE.search(txt))

    return s


def build_payload(profile_row: pd.Series, risk_prob: float, portfolio: Optional[str], rag: RagService) -> Dict[str, Any]:
    risk_hint = _risk_level_from_prob(float(risk_prob))

    payload = {
        "user_id": int(profile_row["user_id"]),
        "profile": {
            "user_id": int(profile_row["user_id"]),
            "age": None if pd.isna(profile_row.get("age")) else int(profile_row.get("age")),
            "education": None if pd.isna(profile_row.get("education")) else str(profile_row.get("education")),
            "income10k": None if pd.isna(profile_row.get("income10k")) else float(profile_row.get("income10k")),
            "asset10k": None if pd.isna(profile_row.get("asset10k")) else float(profile_row.get("asset10k")),
            "debt10k": None if pd.isna(profile_row.get("debt10k")) else float(profile_row.get("debt10k")),
            "children": None if pd.isna(profile_row.get("children")) else int(profile_row.get("children")),
            "exp_years": None if pd.isna(profile_row.get("exp_years")) else int(profile_row.get("exp_years")),
        },
        "model_predictions": {
            "risk_prob": round(float(risk_prob), 4),
            "risk_level_hint": risk_hint,
        },
        "portfolio_recommendation": portfolio,
        "note": "risk_level_hint 是程序根据风险概率给出的提示，可参考但不必机械照搬。",
    }

    # rag evidence
    query = (
        "中国市场 投顾 合规 资产配置 风险管理 再平衡 定投 "
        f"风险等级:{risk_hint} 风险概率:{float(risk_prob):.2f} "
        f"年龄:{payload['profile'].get('age')} 收入:{payload['profile'].get('income10k')} 资产:{payload['profile'].get('asset10k')} 负债:{payload['profile'].get('debt10k')} "
        f"子女:{payload['profile'].get('children')} 经验:{payload['profile'].get('exp_years')} 配置:{portfolio}"
    )
    payload["rag_evidence"] = rag.retrieve(query, top_k=6)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles_csv", default=os.path.join("outputs", "profiles.csv"))
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--portfolio_csv", default=os.path.join("work", "portfolio_recs.csv"))
    p.add_argument("--max_users", type=int, default=50)
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "siliconflow"), choices=["siliconflow", "hf"])
    p.add_argument("--kb_root", default=os.getenv("KB_ROOT") or os.path.join(os.path.dirname(__file__), "knowledge_base"))
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)

    # load data
    profiles = pd.read_csv(args.profiles_csv)
    pred_map = load_predictions_map(args.predictions_csv)
    port_map = load_portfolio_map(args.portfolio_csv)

    rag = RagService(kb_root=args.kb_root)

    # init llm client
    if args.provider == "siliconflow":
        client = SiliconFlowClient(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1",
            model=os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen3-VL-32B-Instruct",
            timeout=int(os.getenv("SILICONFLOW_TIMEOUT", "90")),
        )
    else:
        client = HfInferenceClient(
            api_token=os.getenv("HF_API_TOKEN"),
            model=os.getenv("HF_API_MODEL"),
            base_url=os.getenv("HF_API_BASE_URL") or "https://api-inference.huggingface.co",
            timeout=int(os.getenv("HF_API_TIMEOUT", "90")),
        )

    cache = TTLCache(ttl_seconds=3600, max_items=4096)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join("results", f"advisor_llm_outputs_{args.provider}_{ts}.jsonl")
    err_jsonl = os.path.join("results", f"advisor_llm_errors_{args.provider}_{ts}.jsonl")
    out_csv = os.path.join("results", f"advisor_llm_eval_{args.provider}_{ts}.csv")

    rows = []
    n = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out, open(err_jsonl, "w", encoding="utf-8") as f_err:
        for _, r in profiles.iterrows():
            uid = int(r["user_id"])
            if uid not in pred_map:
                continue
            risk_prob = float(pred_map[uid]["risk_prob"])
            portfolio = port_map.get(uid)

            payload = build_payload(r, risk_prob, portfolio, rag)
            cache_key = cache.make_key({"provider": args.provider, "payload": payload})
            cached = cache.get(cache_key)

            start = time.time()
            try:
                if cached is not None:
                    advice = cached
                    latency = 0.0
                    from_cache = True
                else:
                    user_prompt = (
                        "你将为以下用户生成投顾建议。\n"
                        "输入数据如下（JSON）：\n"
                        f"{payload}\n\n"
                        f"{advisor_schema_hint()}"
                    )
                    advice = client.chat_json(
                        system_prompt=advisor_system_prompt(),
                        user_prompt=user_prompt,
                        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
                        retries=int(os.getenv("LLM_RETRIES", "2")),
                    )
                    cache.set(cache_key, advice)
                    latency = time.time() - start
                    from_cache = False

                # score
                sc = score_one(advice)

                record = {
                    "provider": args.provider,
                    "user_id": uid,
                    "risk_prob": risk_prob,
                    "latency_sec": round(float(latency), 4),
                    "from_cache": bool(from_cache),
                    **sc,
                }
                rows.append(record)

                f_out.write(json.dumps({"user_id": uid, "input": payload, "advice": advice}, ensure_ascii=False) + "\n")
                n += 1
            except Exception as e:
                f_err.write(json.dumps({"user_id": uid, "error": str(e)}, ensure_ascii=False) + "\n")
                rows.append(
                    {
                        "provider": args.provider,
                        "user_id": uid,
                        "risk_prob": risk_prob,
                        "latency_sec": round(float(time.time() - start), 4),
                        "from_cache": False,
                        "json_parse_ok": False,
                        "has_all_fields": False,
                        "sources_has_compliance": False,
                        "sensitive_hit": False,
                        "bullet_points_len": 0,
                        "action_plan_len": 0,
                    }
                )
                n += 1

            if n >= args.max_users:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 打印汇总
    def rate(col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return 0.0
        return float(df[col].mean())

    print("Saved:")
    print(" - outputs jsonl:", out_jsonl)
    print(" - errors  jsonl:", err_jsonl)
    print(" - eval csv:", out_csv)
    print("\nSummary:")
    print(f" - json_parse_ok rate: {rate('json_parse_ok'):.1%}")
    print(f" - has_all_fields rate: {rate('has_all_fields'):.1%}")
    print(f" - sources_has_compliance rate: {rate('sources_has_compliance'):.1%}")
    print(f" - sensitive_hit rate (lower is better): {rate('sensitive_hit'):.1%}")
    if "latency_sec" in df.columns:
        print(f" - avg latency_sec: {df['latency_sec'].mean():.3f}")


if __name__ == "__main__":
    main()