"""
api_server.py
用途：加载训练好的 PyTorch LSTM 模型，提供 RESTful API 接口给小程序调用。
运行方式：uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from siliconflow_client import SiliconFlowClient
from cache import TTLCache
from advisor_llm import _risk_level_from_prob

# NEW: RAG 服务（根目录文件）
from rag_service import RagService  # NEW

# ================= 1. 定义模型结构 (必须与训练时一致) =================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.out(out)
        return out

# ================= 2. 初始化 API 应用 =================

model_bin = None
device = torch.device("cpu")

_llm_client: Optional[SiliconFlowClient] = None
_llm_cache = TTLCache(ttl_seconds=int(os.getenv("LLM_CACHE_TTL", "600")))

# NEW: RAG 全局对象
_rag: Optional[RagService] = None  # NEW

# ================= 3. 加载模型 (启动时运行) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bin, _llm_client, _rag  # NEW: 加 _rag

    model_path = "work/best_model_binary.pth"

    if not os.path.exists(model_path):
        print(f"⚠️ 警告: 找不到模型文件 {model_path}，请先上传！")
    else:
        model_bin = LSTMClassifier(input_dim=9, hidden_dim=128, layer_dim=2, output_dim=2)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model_bin.load_state_dict(state_dict)
            model_bin.to(device)
            model_bin.eval()
            print("✅ 二分类 LSTM 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            model_bin = None

    # 初始化 LLM client：缺 key 不影响服务启动
    try:
        _llm_client = SiliconFlowClient(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1",
            model=os.getenv("SILICONFLOW_MODEL") or "Qwen/Qwen3-VL-32B-Instruct",
            timeout=int(os.getenv("SILICONFLOW_TIMEOUT", "90")),
        )
        print("✅ LLM Client 初始化成功（SiliconFlow）")
    except Exception as e:
        _llm_client = None
        print(f"⚠️ LLM Client 未启用: {e}")

    # NEW: 初始化 RAG（知识库）
    try:
        kb_root = os.getenv("KB_ROOT") or os.path.join(os.path.dirname(__file__), "knowledge_base")
        _rag = RagService(kb_root=kb_root)
        print(f"✅ RAG 初始化成功: {kb_root}")
    except Exception as e:
        _rag = None
        print(f"⚠️ RAG 未启用: {e}")

    yield
    print("🛑 Server shutting down...")

app = FastAPI(title="智能投顾 AI 核心接口", lifespan=lifespan)

# ================= 4. 定义请求数据格式 =================
class MarketData(BaseModel):
    user_id: int


class UserProfile(BaseModel):
    user_id: int
    age: Optional[int] = None
    education: Optional[str] = None
    income10k: Optional[float] = None
    asset10k: Optional[float] = None
    debt10k: Optional[float] = None
    children: Optional[int] = None
    exp_years: Optional[int] = None


class AdvisorRequest(BaseModel):
    profile: UserProfile
    risk_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    portfolio: Optional[str] = None


# ================= 5. LLM 投顾生成（结构化 JSON） =================
def _advisor_system_prompt() -> str:
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


def _advisor_schema_hint() -> str:
    # NEW: 增加 sources 字段
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


def _call_llm_advice(req: AdvisorRequest) -> Dict[str, Any]:
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="LLM 未启用（缺少 SILICONFLOW_API_KEY 或初始化失败）")

    prob = req.risk_prob if req.risk_prob is not None else 0.5
    risk_hint = _risk_level_from_prob(float(prob))

    payload = {
        "user_id": req.profile.user_id,
        "profile": req.profile.model_dump(),
        "model_predictions": {
            "risk_prob": round(float(prob), 4),
            "risk_level_hint": risk_hint,
        },
        "portfolio_recommendation": req.portfolio,
        "note": "risk_level_hint 是程序根据风险概率给出的提示，可参考但不必机械照搬。",
    }

    # NEW: RAG 检索并注入证据（混合检索 + 强制合规）
    if _rag is not None:
        query = (
            "中国市场 投顾 合规 资产配置 风险管理 再平衡 定投 "
            f"风险等级:{risk_hint} 风险概率:{float(prob):.2f} "
            f"年龄:{req.profile.age} 收入:{req.profile.income10k} 资产:{req.profile.asset10k} 负债:{req.profile.debt10k} "
            f"子女:{req.profile.children} 经验:{req.profile.exp_years} 配置:{req.portfolio}"
        )
        payload["rag_evidence"] = _rag.retrieve(query, top_k=6)
    else:
        payload["rag_evidence"] = []

    cache_key = _llm_cache.make_key(payload)
    cached = _llm_cache.get(cache_key)
    if cached is not None:
        return cached

    user_prompt = (
        "你将为以下用户生成投顾建议。\n"
        "输入数据如下（JSON）：\n"
        f"{payload}\n\n"
        f"{_advisor_schema_hint()}"
    )

    advice = _llm_client.chat_json(
        system_prompt=_advisor_system_prompt(),
        user_prompt=user_prompt,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
        retries=int(os.getenv("LLM_RETRIES", "2")),
    )

    if not isinstance(advice, dict):
        advice = {}

    advice.setdefault("risk_level", risk_hint)
    advice.setdefault("summary", "")
    advice.setdefault("bullet_points", [])
    advice.setdefault("action_plan", [])

    # NEW: sources 默认回填（来自 rag_evidence，至少保证有来源可解释）
    evidence = payload.get("rag_evidence") or []
    advice.setdefault(
        "sources",
        [
            {
                "doc_id": e.get("doc_id"),
                "title": e.get("title"),
                "category": e.get("category"),
                "source": e.get("source"),
                "updated_at": e.get("updated_at"),
            }
            for e in evidence
        ],
    )

    advice.setdefault("disclaimer", "免责声明：本内容仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。")

    _llm_cache.set(cache_key, advice)
    return advice


@app.post("/advisor/generate")
async def advisor_generate(req: AdvisorRequest):
    return {
        "status": "success",
        "user_id": req.profile.user_id,
        "llm_advice": _call_llm_advice(req),
    }


# NEW: 手动刷新 RAG 索引（配合 Windows 定时任务更新动态库）
@app.post("/rag/refresh")
async def rag_refresh():
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG 未启用")
    _rag.refresh()
    return {"status": "success", "message": "RAG index refreshed"}


# ================= 6. 风险预测接口 =================
@app.post("/predict/risk")
async def predict_risk(data: MarketData, with_llm: bool = Query(default=False)):
    if model_bin is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    dummy_input = torch.randn(1, 60, 9).to(device)
    with torch.no_grad():
        output = model_bin(dummy_input)
        probs = torch.softmax(output, dim=1)
        buy_prob = probs[0, 1].item()

    if buy_prob > 0.6:
        advice = "进取型：建议关注科技成长板块"
    elif buy_prob < 0.4:
        advice = "保守型：建议持有现金或国债"
    else:
        advice = "稳健型：建议定投沪深300"

    resp: Dict[str, Any] = {
        "user_id": data.user_id,
        "buy_probability": round(buy_prob, 4),
        "advice": advice,
        "status": "success",
    }

    if with_llm:
        req = AdvisorRequest(
            profile=UserProfile(user_id=data.user_id),
            risk_prob=float(buy_prob),
            portfolio=None,
        )
        resp["llm_advice"] = _call_llm_advice(req)

    return resp


@app.get("/")
def read_root():
    return {"message": "GXB AI Server is Running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)