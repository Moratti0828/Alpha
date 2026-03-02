import os
import json
import time
from typing import Any, Dict, Optional, List

import requests


class SiliconFlowClient:
    """
    SiliconFlow OpenAI-compatible chat client.

    Env:
      - SILICONFLOW_API_KEY: required
      - SILICONFLOW_BASE_URL: optional, default "https://api.siliconflow.cn/v1"
      - SILICONFLOW_MODEL: optional, default "Qwen/Qwen3-VL-32B-Instruct"

    Notes:
      - Try response_format=json_object if supported.
      - If unsupported, fallback to strict JSON prompting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")

        if not self.api_key:
            # Try reading from local file if not found
            try:
                key_file = os.path.join(os.path.dirname(__file__), "api_key.txt")
                if os.path.exists(key_file):
                    with open(key_file, "r", encoding="utf-8") as f:
                        self.api_key = f.read().strip()
            except Exception:
                pass

        if not self.api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY env var for SiliconFlow API.")

        # Auto-detect DashScope key
        default_base_url = "https://api.siliconflow.cn/v1"
        default_model = "Qwen/Qwen3-VL-32B-Instruct"

        # if self.api_key == "sk-9cad32ea69024b71a79374e609abf0ca":
        #      default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        #      default_model = "qwen-plus"
        #      # If the model is the SiliconFlow default, switch to DashScope default
        #      if model == "Qwen/Qwen3-VL-32B-Instruct":
        #          model = default_model

        self.base_url = (base_url or os.getenv("SILICONFLOW_BASE_URL") or default_base_url).rstrip("/")
        self.model = model or os.getenv("SILICONFLOW_MODEL") or default_model
        self.timeout = timeout

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
        retries: int = 2,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Try with response_format first
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 400:
                    break
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                last_err = e
                time.sleep(1.2 * (attempt + 1))

        # Fallback: no response_format
        fallback_payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt + "\n\n注意：只输出严格JSON，不要输出```、解释或任何额外字符。",
                },
            ],
        }

        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=fallback_payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                return json.loads(content)
            except Exception as e:
                last_err = e
                time.sleep(1.2 * (attempt + 1))

        raise RuntimeError(f"SiliconFlow chat_json failed after retries: {last_err}")