"""LLM 通信层 - 封装 MiMo API 调用、重试、熔断。

提供两种接口：
1. LLM 类：直接 API 调用（用于 L2 降级路径）
2. create_langchain_llm()：返回 LangChain ChatOpenAI 实例（用于 Agent）
"""

import logging
import time
from collections import deque

import requests

import config

logger = logging.getLogger(__name__)


def _sanitize_error(error_msg: str) -> str:
    """从错误消息中移除 API Key。"""
    key = config.MIMO_API_KEY
    if key and len(key) > 4 and key in error_msg:
        return error_msg.replace(key, "***REDACTED***")
    return error_msg


def create_langchain_llm():
    """创建 LangChain 兼容的 LLM 实例。

    使用 ChatOpenAI 指向 MiMo API（OpenAI 兼容接口）。
    若 langchain-openai 不可用，返回 None。

    Returns:
        ChatOpenAI 实例，或 None
    """
    if not config.MIMO_API_KEY:
        logger.warning("MIMO_API_KEY not set, LangChain LLM disabled")
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.MIMO_MODEL_NAME,
            openai_api_key=config.MIMO_API_KEY,
            openai_api_base=config.MIMO_BASE_URL,
            temperature=0,
            max_tokens=512,
            request_timeout=config.AGENT_LLM_TIMEOUT,
        )
        return llm
    except ImportError:
        logger.warning(
            "langchain-openai not installed, "
            "LangChain LLM unavailable"
        )
        return None
    except Exception as e:
        logger.error("Failed to create LangChain LLM: %s", e)
        return None


def create_followup_llm():
    """创建 followup 专用的 LangChain LLM（低 token、短超时）。

    用于工具调用后的 followup 回答，只需基于事实简短回答。
    max_tokens=256, request_timeout=10s。

    Returns:
        ChatOpenAI 实例，或 None
    """
    if not config.MIMO_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.MIMO_MODEL_NAME,
            openai_api_key=config.MIMO_API_KEY,
            openai_api_base=config.MIMO_BASE_URL,
            temperature=0,
            max_tokens=config.AGENT_FOLLOWUP_MAX_TOKENS,
            request_timeout=config.AGENT_FOLLOWUP_TIMEOUT,
        )
    except Exception as e:
        logger.error("Failed to create followup LLM: %s", e)
        return None


class LLM:
    """MiMo API LLM 通信层（直接调用，用于 L2 降级路径）。"""

    def __init__(self):
        self._history = deque(maxlen=config.AGENT_LLM_MAX_HISTORY * 2)
        self._circuit_breaker_until = 0.0
        self._consecutive_failures = 0

    def _is_circuit_open(self) -> bool:
        if self._circuit_breaker_until > 0:
            if time.time() < self._circuit_breaker_until:
                return True
            self._circuit_breaker_until = 0.0
            self._consecutive_failures = 0
        return False

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._circuit_breaker_until = time.time() + 60
            logger.warning(
                "Circuit breaker opened for 60s "
                "after 3 consecutive failures"
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    @property
    def is_available(self) -> bool:
        return bool(config.MIMO_API_KEY) and not self._is_circuit_open()

    def _call_api(self, messages: list) -> str:
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

        logger.debug("API request -> POST %s", url)
        logger.debug("_call_api: HTTP POST START | url=%s | timeout=%ds", url, config.AGENT_LLM_TIMEOUT)
        t0 = time.time()

        resp = requests.post(
            url, json=payload, headers=headers,
            timeout=config.AGENT_LLM_TIMEOUT,
        )

        elapsed = time.time() - t0
        logger.debug("_call_api: HTTP POST DONE | status=%d | %.3fs", resp.status_code, elapsed)
        logger.debug("API response <- status=%d", resp.status_code)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = min(int(retry_after), 5) if retry_after else 2
            raise RateLimitError(
                f"Rate limited, retry after {wait}s", wait
            )

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

    def ask(self, question: str, context: str = None,
            lang: str = None) -> str:
        """简单问答接口（L2 降级路径使用）。"""
        from core.i18n import t as i18n_t

        system_content = i18n_t("llm.system_prompt")
        if lang:
            lang_name = config.LANG_NAMES.get(lang, lang)
            lang_instruction = i18n_t(
                "llm.must_respond_in", lang=lang_name
            )
            system_content = f"{system_content}\n\n{lang_instruction}"

        if context and context.strip() and config.AGENT_RAG_ENABLED:
            max_len = config.AGENT_RAG_MAX_CONTEXT_LEN
            if len(context) > max_len:
                context = (
                    context[:max_len]
                    + i18n_t("llm.context_truncated")
                )
            system_content = (
                f"{i18n_t('llm.rag_system_prompt')}\n\n"
                f"{i18n_t('llm.context_header')}\n{context}\n"
                f"{i18n_t('llm.context_footer')}"
            )
            if lang:
                lang_name = config.LANG_NAMES.get(lang, lang)
                lang_instruction = i18n_t(
                    "llm.must_respond_in", lang=lang_name
                )
                system_content = (
                    f"{system_content}\n\n{lang_instruction}"
                )

        # MiMo API 不支持 system 角色，将 system prompt 合并到 user 消息中
        user_content = question
        if system_content:
            user_content = f"{system_content}\n\n{question}"
        messages = []
        messages.extend(list(self._history))
        messages.append({"role": "user", "content": user_content})

        answer = self._ask_messages(messages)
        if answer:
            self._history.append({"role": "user", "content": question})
            self._history.append(
                {"role": "assistant", "content": answer}
            )
        return answer

    def _ask_messages(self, messages: list) -> str:
        """发送 messages 列表，返回 LLM 输出。"""
        if not config.MIMO_API_KEY:
            logger.debug("_ask_messages: no API key, return None")
            return None
        if self._is_circuit_open():
            logger.debug("_ask_messages: circuit breaker OPEN, return None")
            return None

        max_retries = config.AGENT_LLM_MAX_RETRIES
        t_total = time.time()
        logger.debug("_ask_messages: START | max_retries=%d", max_retries)

        last_error = None
        for attempt in range(max_retries + 1):
            logger.debug("_ask_messages: attempt %d/%d", attempt+1, max_retries+1)
            try:
                answer = self._call_api(messages)
                self._record_success()
                elapsed = time.time() - t_total
                logger.debug("_ask_messages: SUCCESS on attempt %d | total=%.3fs", attempt+1, elapsed)
                return answer
            except RateLimitError as e:
                last_error = e
                wait = getattr(e, "wait_seconds", 2)
                logger.debug("_ask_messages: RateLimitError, sleeping %ds", wait)
                time.sleep(wait)
            except requests.exceptions.ConnectTimeout:
                last_error = "connect_timeout"
                logger.debug("_ask_messages: ConnectTimeout on attempt %d", attempt+1)
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
            except requests.exceptions.ReadTimeout:
                last_error = "read_timeout"
                logger.debug("_ask_messages: ReadTimeout on attempt %d", attempt+1)
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
            except requests.exceptions.ConnectionError:
                last_error = "connection_error"
                logger.debug("_ask_messages: ConnectionError on attempt %d", attempt+1)
                if attempt < max_retries:
                    time.sleep(2)
            except APIError as e:
                last_error = e
                logger.debug("_ask_messages: APIError on attempt %d: %s", attempt+1, e)
                if "401" in str(e):
                    self._record_failure()
                    return None
                if attempt < max_retries:
                    time.sleep(1)
            except Exception as e:
                last_error = f"unexpected: {e}"
                logger.debug("_ask_messages: UnexpectedException: %s", e)
                break

        self._record_failure()
        elapsed = time.time() - t_total
        logger.debug("_ask_messages: ALL ATTEMPTS FAILED | total=%.3fs | last_error=%s", elapsed, last_error)
        logger.warning(
            "LLM call failed: %s",
            _sanitize_error(str(last_error)),
        )
        return None

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> list:
        return list(self._history)


class APIError(Exception):
    pass


class RateLimitError(APIError):
    def __init__(self, message="Rate limited", wait_seconds=2):
        super().__init__(message)
        self.wait_seconds = wait_seconds
