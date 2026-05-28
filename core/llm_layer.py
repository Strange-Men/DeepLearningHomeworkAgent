"""LLM问答层模块 - 封装 MiMo API 调用、上下文管理、重试与降级。"""

import time
import logging
from collections import deque

import requests

try:
    import config
except ImportError:
    import os
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    import config

from core.i18n import t

logger = logging.getLogger(__name__)


def _sanitize_error(error_msg: str) -> str:
    """Remove API key from error messages."""
    key = config.MIMO_API_KEY
    if key and len(key) > 4 and key in error_msg:
        return error_msg.replace(key, "***REDACTED***")
    return error_msg


class LLMLayer:
    """MiMo API LLM 问答层。"""

    def __init__(self):
        self._history = deque(maxlen=config.AGENT_LLM_MAX_HISTORY * 2)
        self._circuit_breaker_until = 0.0
        self._consecutive_failures = 0

    def _is_circuit_open(self):
        """检查熔断器是否打开。"""
        if self._circuit_breaker_until > 0:
            if time.time() < self._circuit_breaker_until:
                return True
            # 熔断恢复
            self._circuit_breaker_until = 0.0
            self._consecutive_failures = 0
        return False

    def _record_failure(self):
        """记录一次失败，达到阈值触发熔断。"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._circuit_breaker_until = time.time() + 60
            logger.warning(
                "Circuit breaker opened for 60s after 3 consecutive failures"
            )

    def _record_success(self):
        """记录成功，重置失败计数。"""
        self._consecutive_failures = 0

    def _build_messages(self, question, lang=None):
        """构建 API 请求的 messages 列表。"""
        system_content = t("llm.system_prompt")
        if lang:
            lang_name = config.LANG_NAMES.get(lang, lang)
            lang_instruction = t("llm.must_respond_in", lang=lang_name)
            system_content = f"{system_content}\n\n{lang_instruction}"
        messages = [{"role": "system", "content": system_content}]
        # 加入对话历史
        messages.extend(list(self._history))
        # 加入当前问题
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _is_valid_context(context):
        """判断检索上下文是否包含有效数据。

        仅检查是否为空字符串或纯空白。
        不做关键词过滤——RAG prompt 已指导 LLM 在上下文不足时
        用通用知识回答，因此即使上下文很短也应传入，由 LLM 自行判断。
        """
        if not context or not context.strip():
            return False
        return True

    def _build_rag_messages(self, question, context, lang=None):
        """构建带检索上下文的 RAG messages 列表。

        Args:
            question: 用户问题
            context: 检索工具返回的上下文文本
            lang: 目标回答语言代码

        Returns:
            list: API messages 列表
        """
        # 检索上下文为空或无效时，使用普通模式（让 LLM 用通用知识回答）
        if not self._is_valid_context(context):
            logger.debug("Context empty or invalid, using normal mode")
            return self._build_messages(question, lang=lang)

        # 截断过长的上下文
        if len(context) > config.AGENT_RAG_MAX_CONTEXT_LEN:
            context = (
                context[:config.AGENT_RAG_MAX_CONTEXT_LEN]
                + t("llm.context_truncated")
            )

        system_content = (
            f"{t('llm.rag_system_prompt')}\n\n"
            f"{t('llm.context_header')}\n{context}\n"
            f"{t('llm.context_footer')}"
        )
        if lang:
            lang_name = config.LANG_NAMES.get(lang, lang)
            lang_instruction = t("llm.must_respond_in", lang=lang_name)
            system_content = f"{system_content}\n\n{lang_instruction}"
        messages = [{"role": "system", "content": system_content}]
        messages.extend(list(self._history))
        messages.append({"role": "user", "content": question})
        return messages

    def _call_api(self, messages):
        """单次 API 调用。"""
        url = f"{config.MIMO_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.MIMO_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.MIMO_MODEL_NAME,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
        }

        # 调试日志：请求 URL（不打印 API Key）
        safe_url = url
        logger.debug("API request -> POST %s", safe_url)
        logger.debug(
            "API request headers -> Content-Type: %s, "
            "Authorization: Bearer ***",
            headers["Content-Type"],
        )

        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=config.AGENT_LLM_TIMEOUT,
        )

        logger.debug("API response <- status=%d", resp.status_code)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = min(int(retry_after), 5) if retry_after else 2
            raise RateLimitError(f"Rate limited, retry after {wait}s", wait)

        if resp.status_code == 401:
            raise APIError("Authentication failed (401)")

        if resp.status_code >= 500:
            raise APIError(f"Server error ({resp.status_code})")

        if resp.status_code != 200:
            body = _sanitize_error(resp.text)
            raise APIError(f"API error {resp.status_code}: {body}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    def ask(self, question, context=None, lang=None):
        """调用 LLM 回答问题，含重试和熔断逻辑。

        Args:
            question: 用户问题
            context: 可选的检索上下文，传入时使用 RAG 模式
            lang: 目标回答语言代码

        Returns:
            str: LLM 回答，失败时返回 None
        """
        if not config.MIMO_API_KEY:
            return None

        if self._is_circuit_open():
            logger.debug("Circuit breaker is open, skipping LLM call")
            return None

        if context and config.AGENT_RAG_ENABLED:
            messages = self._build_rag_messages(question, context, lang=lang)
        else:
            messages = self._build_messages(question, lang=lang)
        last_error = None

        for attempt in range(config.AGENT_LLM_MAX_RETRIES + 1):
            try:
                answer = self._call_api(messages)
                self._record_success()
                # 更新对话历史
                self._history.append({"role": "user", "content": question})
                self._history.append({"role": "assistant", "content": answer})
                return answer

            except RateLimitError as e:
                last_error = e
                wait = getattr(e, "wait_seconds", 2)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)

            except requests.exceptions.ConnectTimeout:
                last_error = "connect_timeout"
                logger.warning(
                    "LLM connect timeout (attempt %d) - "
                    "cannot reach server at %s",
                    attempt + 1, config.MIMO_BASE_URL,
                )
                if attempt < config.AGENT_LLM_MAX_RETRIES:
                    time.sleep(2 * (attempt + 1))

            except requests.exceptions.ReadTimeout:
                last_error = "read_timeout"
                logger.warning(
                    "LLM read timeout (attempt %d) - "
                    "server accepted connection but no response in %ds",
                    attempt + 1, config.AGENT_LLM_TIMEOUT,
                )
                if attempt < config.AGENT_LLM_MAX_RETRIES:
                    time.sleep(1 * (attempt + 1))

            except requests.exceptions.ConnectionError as e:
                last_error = "connection_error"
                safe_msg = _sanitize_error(str(e))
                logger.warning(
                    "LLM connection error (attempt %d): %s",
                    attempt + 1, safe_msg,
                )
                if attempt < config.AGENT_LLM_MAX_RETRIES:
                    time.sleep(2)

            except APIError as e:
                last_error = e
                safe_msg = _sanitize_error(str(e))
                logger.warning("LLM API error: %s", safe_msg)
                if "401" in str(e):
                    # 认证失败不重试
                    self._record_failure()
                    return None
                if attempt < config.AGENT_LLM_MAX_RETRIES:
                    time.sleep(1)

            except Exception as e:
                last_error = e
                logger.warning("LLM unexpected error: %s", type(e).__name__)
                break

        self._record_failure()
        safe_err = _sanitize_error(str(last_error))
        logger.warning("LLM call failed after retries: %s", safe_err)
        return None

    def clear_history(self):
        """清空对话历史。"""
        self._history.clear()

    def get_history(self):
        """获取当前对话历史（只读副本）。"""
        return list(self._history)


class APIError(Exception):
    """MiMo API 调用异常。"""
    pass


class RateLimitError(APIError):
    """API 限流异常。"""

    def __init__(self, message="Rate limited", wait_seconds=2):
        super().__init__(message)
        self.wait_seconds = wait_seconds


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    from core.i18n import init as i18n_init
    i18n_init()

    print(f"[INFO] MIMO_BASE_URL = {config.MIMO_BASE_URL}")
    print(f"[INFO] MIMO_MODEL_NAME = {config.MIMO_MODEL_NAME}")
    key_display = (
        f"***{config.MIMO_API_KEY[-4:]}"
        if config.MIMO_API_KEY and len(config.MIMO_API_KEY) > 4
        else "(not set)"
    )
    print(f"[INFO] MIMO_API_KEY = {key_display}")
    print(f"[INFO] Timeout = {config.AGENT_LLM_TIMEOUT}s")
    print()

    llm = LLMLayer()
    test_question = "Hello, briefly introduce yourself."
    print(f"[TEST] Asking: {test_question}")
    answer = llm.ask(test_question)

    if answer:
        print(f"[PASS] Got response:\n{answer}")
        sys.exit(0)
    else:
        print("[FAIL] No response received. Check logs above for details.")
        sys.exit(1)
