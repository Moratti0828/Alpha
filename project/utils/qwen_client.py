import os
import json
from typing import Any, Dict, Optional

import requests


class QwenClient:
    """
    Qwen OpenAI-compatible client（DashScope compatible-mode）。
    Env:
      - QWEN_API_KEY: required
      - QWEN_BASE_URL: optional, default "https://dashscope.aliyuncs.com/compatible-mode/v1"
      - QWEN_MODEL: optional, default "qwen-plus"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")

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
            raise RuntimeError("Missing QWEN_API_KEY env var.")

        self.base_url = (base_url or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip(
            "/"
        )
        self.model = model or os.getenv("QWEN_MODEL") or "qwen-plus"
        self.timeout = timeout

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Remove markdown code block markers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

if __name__ == "__main__":
    try:
        client = QwenClient()
        print(f"Client initialized with model: {client.model}")

        system_prompt = "You are a helpful assistant."
        user_prompt = "Generate a JSON object with a key 'message' and value 'Hello Qwen'."

        print("Sending request...")
        response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        print("Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
