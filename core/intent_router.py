"""意图路由器模块 - 决定用户问题走规则层还是LLM层。"""

import logging

import config
from core.i18n import t

logger = logging.getLogger(__name__)


class IntentRouter:
    """意图路由器：根据规则层置信度决定走规则层还是LLM层。"""

    def __init__(self, rule_layer, llm_layer):
        self.rule = rule_layer
        self.llm = llm_layer

    def route(self, question, context=None, lang=None):
        """路由决策。

        Args:
            question: 用户输入的问题
            context: 可选的检索上下文，传递给 LLM 层
            lang: 目标回答语言代码

        Returns:
            tuple: (answer, source)
                source: "rule" | "llm" | "fallback"
        """
        if not question or not question.strip():
            return t("intent.enter_question"), "rule"

        question = question.strip()

        # 1. 规则层匹配
        rule_score, rule_answer = self.rule.match(question)
        logger.debug("Rule layer score: %.4f for: %s", rule_score,
                     question[:50])

        # 2. 精确匹配或高置信度 -> 直接返回
        if rule_score >= config.AGENT_RULE_HIGH_THRESHOLD:
            logger.debug("Route: rule (high confidence %.4f)", rule_score)
            return rule_answer, "rule"

        # 3. LLM 层未启用 -> 仅用规则层
        if not config.AGENT_LLM_ENABLED or not config.MIMO_API_KEY:
            if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
                return rule_answer, "rule"
            return t("intent.unified_fallback"), "fallback"

        # 4. 低置信度 -> 直接走 LLM（带上下文）
        if rule_score < config.AGENT_RULE_LOW_THRESHOLD:
            logger.debug("Route: trying LLM (low rule score %.4f)",
                         rule_score)
            llm_answer = self.llm.ask(question, context=context, lang=lang)
            if llm_answer:
                return llm_answer, "llm"
            # LLM 失败 -> 降级到统一兜底
            logger.debug("Route: fallback (LLM failed, low rule score)")
            return self._fallback(rule_score, rule_answer)

        # 5. 模糊区间 -> 双路并行（LLM 带上下文）
        logger.debug("Route: trying LLM (fuzzy zone %.4f)", rule_score)
        llm_answer = self.llm.ask(question, context=context, lang=lang)
        if llm_answer:
            return llm_answer, "llm"
        # LLM 失败 -> 使用规则层结果
        if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
            return rule_answer, "rule"
        return self._fallback(rule_score, rule_answer)

    def _fallback(self, rule_score, rule_answer):
        """降级处理 - 使用统一兜底模板，保证一致性。"""
        if rule_score >= config.AGENT_SIMILARITY_THRESHOLD:
            return rule_answer + t("intent.fallback_hint"), "fallback"
        return t("intent.unified_fallback"), "fallback"
