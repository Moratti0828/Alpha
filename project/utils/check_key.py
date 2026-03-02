import sys
import requests
import os

with open("check_key_result.txt", "w") as f_out:
    def log(msg):
        f_out.write(str(msg) + "\n")
        f_out.flush()

    log("Starting check_key.py...")

    api_key = "sk-9cad32ea69024b71a79374e609abf0ca"

    def test_siliconflow():
        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            log(f"SiliconFlow: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            log(f"SiliconFlow Error: {e}")

    def test_dashscope():
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            log(f"DashScope: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            log(f"DashScope Error: {e}")

    log(f"Testing key: {api_key}")
    test_siliconflow()
    test_dashscope()
