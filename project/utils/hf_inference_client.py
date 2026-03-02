"""
Hugging Face Inference API client (text-generation / chat style)
用途：在不占用本地显存的情况下，对比不同 HF 模型的投顾 JSON 输出质量。

环境变量：
  - HF_API_TOKEN: required
  - HF_API_MODEL: required (模型 repo id)
  - HF_API_BASE_URL: optional, default https://api-inference.huggingface.co

说明：
- Inference API 对不同模型的“chat模板”支持不一致，因此这里采用“拼接 system+user prompt”并请求 text-generation。
- 为了得到 JSON，使用强约束提示 + 后处理提取 JSON 对象。
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests


_JSON_RE = re.compile(r"\{[\s\S]*\}$")  # 从末尾抓一个尽量大的 JSON（较保守）
_FIRST_OBJ_RE = re.compile(r"\{[\s\S]*\}")  # 从文本中抓第一个 {...}


class HfInferenceClient:
    def __init__(
        self,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
    ):
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
        self.model = model or os.getenv("HF_API_MODEL")
        self.base_url = (base_url or os.getenv("HF_API_BASE_URL") or "https://api-inference.huggingface.co").rstrip("/")
        self.timeout = timeout

        if not self.api_token:
            raise RuntimeError("Missing HF_API_TOKEN")
        if not self.model:
            raise RuntimeError("Missing HF_API_MODEL")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}

    def _endpoint(self) -> str:
        # 通用 text-generation endpoint
        return f"{self.base_url}/models/{self.model}"

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 900,
        temperature: float = 0.7,
        top_p: float = 0.9,
        retries: int = 2,
    ) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
            },
            "options": {
                # wait_for_model=True：若模型冷启动，API会等待加载
                "wait_for_model": True,
            },
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(self._endpoint(), headers=self._headers(), json=payload, timeout=self.timeout)
                # 503 时模型可能在加载；重试
                if resp.status_code in (429, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()

                # 常见格式：list[{"generated_text": "..."}]
                if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                    return str(data[0]["generated_text"])
                # 也可能直接返回 dict
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"])
                return json.dumps(data, ensure_ascii=False)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"HF Inference generate_text failed: {last_err}")

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
        retries: int = 2,
    ) -> Dict[str, Any]:
        # 统一格式：system+user 合并成单 prompt
        prompt = (
            f"System:\n{system_prompt}\n\n"
            f"User:\n{user_prompt}\n\n"
            "Assistant:\n"
            "注意：只输出严格JSON对象，不要输出Markdown代码块，不要输出任何解释文字。\n"
        )

        txt = self.generate_text(prompt, max_new_tokens=max_tokens, temperature=temperature, retries=retries).strip()

        # 常见清理：去掉 ```json / ```
        txt = txt.strip()
        txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        # 尝试直接 parse
        try:
            return json.loads(txt)
        except Exception:
            pass

        # 尝试抓第一个 { ... } 段
        m = _FIRST_OBJ_RE.search(txt)
        if m:
            snippet = m.group(0).strip()
            try:
                return json.loads(snippet)
            except Exception:
                pass

        # 尝试抓“以 { 开头到结尾 }”的段
        m2 = _JSON_RE.search(txt)
        if m2:
            snippet = m2.group(0).strip()
            try:
                return json.loads(snippet)
            except Exception:
                pass

        raise ValueError(f"Model did not return valid JSON. Raw output head: {txt[:300]}")