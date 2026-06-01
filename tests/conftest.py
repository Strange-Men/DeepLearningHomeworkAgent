"""共享测试 fixtures。"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockLLM:
    """Mock LLM，按顺序返回预设响应。"""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.call_count = 0
        self.is_available = True
        self._history = []

    def ask_messages(self, messages):
        if self.call_count >= len(self.responses):
            return None
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    def ask(self, question, context=None, lang=None):
        return self.ask_messages([])

    def clear_history(self):
        self._history.clear()


class FailingLLM:
    """总是失败的 Mock LLM。"""

    is_available = False

    def ask_messages(self, messages):
        return None

    def ask(self, question, context=None, lang=None):
        return None

    def clear_history(self):
        pass
