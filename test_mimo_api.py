
"""MiMo API 连通性测试脚本。

用法：
    1. 确保 .env 中已配置 MIMO_API_KEY
    2. 运行: python test_mimo_api.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402


def test_api():
    """测试 MiMo API 连通性。"""
    if not config.MIMO_API_KEY:
        print("[SKIP] MIMO_API_KEY not set. Cannot test API.")
        print("       Set it in .env file first.")
        return False

    try:
        import requests
    except ImportError:
        print("[FAIL] requests not installed.")
        return False

    url = f"{config.MIMO_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.MIMO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.MIMO_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "ping"},
        ],
        "temperature": 0,
        "max_tokens": 10,
    }

    print(f"[INFO] Testing API at: {config.MIMO_BASE_URL}")
    print(f"[INFO] Model: {config.MIMO_MODEL_NAME}")
    print(f"[INFO] API Key: ***{config.MIMO_API_KEY[-4:]}")

    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=30
        )
        print(f"[INFO] HTTP Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"[PASS] API response: {content}")
            return True
        elif resp.status_code == 401:
            print("[FAIL] Authentication failed. Check MIMO_API_KEY.")
            return False
        elif resp.status_code == 429:
            print("[FAIL] Rate limited. Try again later.")
            return False
        else:
            # 脱敏输出错误信息
            body = resp.text
            if config.MIMO_API_KEY and config.MIMO_API_KEY in body:
                body = body.replace(config.MIMO_API_KEY, "***REDACTED***")
            print(f"[FAIL] API error: {body}")
            return False

    except requests.exceptions.ConnectTimeout:
        print("[FAIL] Connect timeout - cannot reach server.")
        print(f"       Tried: {config.MIMO_BASE_URL}")
        return False
    except requests.exceptions.ReadTimeout:
        print("[FAIL] Read timeout (30s) - server accepted but no response.")
        return False
    except requests.exceptions.ConnectionError as e:
        err_str = str(e)
        if config.MIMO_API_KEY and config.MIMO_API_KEY in err_str:
            err_str = err_str.replace(config.MIMO_API_KEY, "***REDACTED***")
        print(f"[FAIL] Connection error: {err_str}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
