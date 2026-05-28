"""意图路由器模块 - 决定用户问题走规则层还是LLM层。"""

import logging

import config

logger = logging.getLogger(__name__)

# 降级提示信息
FALLBACK_HINT = (
    "\n\n> [注：当前智能问答服务暂不可用，"
    "以上回答来自本地知识库，可能不够完整。]"
)


class IntentRouter:
    """意图路由器：根据规则层置信度决定走规则层还是LLM层。"""

    def __init__(self, rule_layer, llm_layer):
        self.rule = rule_layer
        self.llm = llm_layer

    def route(self, question):
        """路由决策。

        Args:
            question: 用户输入的问题

        Returns:
            tuple: (answer, source)
                source: "rule" | "llm" | "fallback"
        """
        if not question or not question.strip():
            return "Please enter your question.", "rule"

        question = question.strip()

        # 1. 规则层匹配
        rule_score, rule_answer = self.rule.match(question)

        # 2. 精确匹配或高置信度 -> 直接返回
        if rule_score >= config.AGENT_RULE_HIGH_THRESHOLD:
            return rule_answer, "rule"

        # 3. LLM 层未启用 -> 仅用规则层
        if not config.AGENT_LLM_ENABLED or not config.MIMO_API_KEY:
            if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
                return rule_answer, "rule"
            return self.rule.default_response, "fallback"

        # 4. 低置信度 -> 直接走 LLM
        if rule_score < config.AGENT_RULE_LOW_THRESHOLD:
            llm_answer = self.llm.ask(question)
            if llm_answer:
                return llm_answer, "llm"
            # LLM 失败 -> 降级到规则层
            return self._fallback(rule_score, rule_answer)

        # 5. 模糊区间 -> 双路并行
        llm_answer = self.llm.ask(question)
        if llm_answer:
            # LLM 成功 -> 优先使用 LLM 回答
            return llm_answer, "llm"
        # LLM 失败 -> 使用规则层结果
        if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
            return rule_answer, "rule"
        return self._fallback(rule_score, rule_answer)

    def _fallback(self, rule_score, rule_answer):
        """降级处理。"""
        if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
            return rule_answer + FALLBACK_HINT, "fallback"
        return self.rule.default_response + FALLBACK_HINT, "fallback"
