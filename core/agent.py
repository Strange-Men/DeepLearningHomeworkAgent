"""Agent问答引擎模块 - 双层混合架构（规则层 + LLM层）。"""

import json
import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from core.i18n import t
from core.text_preprocessor import TextPreprocessor
from core.llm_layer import LLMLayer
from core.intent_router import IntentRouter
from core.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class GestureAgent:
    """规则层：基于TF-IDF + 关键词融合的问答Agent。"""

    # 融合权重
    ALPHA = 0.6   # TF-IDF 权重
    BETA = 0.3    # 关键词匹配权重
    GAMMA = 0.1   # 精确匹配加分

    def __init__(self, knowledge_base_path=None, lang=None):
        self._lang = lang or config.DEFAULT_LANGUAGE
        self.kb_path = knowledge_base_path or self._resolve_kb_path(self._lang)
        self.qa_pairs = []
        self.default_response = ""
        self.preprocessor = TextPreprocessor(lang=self._lang)
        self._load_knowledge_base()
        self._build_tfidf_index()

    def _resolve_kb_path(self, lang):
        """根据语言解析知识库路径。"""
        return config.KNOWLEDGE_BASE_PATHS.get(
            lang, config.KNOWLEDGE_BASE_PATH
        )

    def _load_knowledge_base(self):
        """加载知识库。"""
        if not os.path.exists(self.kb_path):
            return
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.qa_pairs = data.get("qa_pairs", [])
        self.default_response = data.get(
            "default_response",
            t("agent.default_response"),
        )
        if self.qa_pairs:
            examples = "\n".join(
                f"  {i}. {qa.get('question', '')}"
                for i, qa in enumerate(self.qa_pairs, 1)
                if qa.get("question")
            )
            self.default_response = (
                f"{t('agent.default_response')}\n"
                f"{t('agent.try_these')}\n\n"
                f"{examples}\n\n"
                f"{t('agent.input_related')}"
            )

    def _get_expanded_keywords(self, qa):
        """合并 keywords + synonyms 为扩展关键词列表。"""
        keywords = qa.get("keywords", [])
        synonyms = qa.get("synonyms", [])
        return keywords + synonyms

    def _build_tfidf_index(self):
        """构建 TF-IDF 索引。"""
        if not self.qa_pairs:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        corpus = []
        for qa in self.qa_pairs:
            question = qa.get("question", "")
            expanded = self._get_expanded_keywords(qa)
            text = question + " " + " ".join(expanded)
            processed = self.preprocessor.process(text)
            corpus.append(processed if processed else question)

        self._tfidf_vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def _calculate_keyword_score(self, question, qa):
        """计算问题与扩展关键词的匹配分数。"""
        expanded = self._get_expanded_keywords(qa)
        if not expanded:
            return 0.0
        question_lower = question.lower()
        matched = 0
        for keyword in expanded:
            if keyword.lower() in question_lower:
                matched += 1
        if matched == 0:
            return 0.0
        return matched / max(len(expanded) ** 0.5, 1)

    def _check_exact_match(self, question, qa):
        """检查是否精确匹配。"""
        example = qa.get("question", "")
        if not example:
            return False
        q = question.strip().rstrip("?！!。，,.")
        e = example.rstrip("?！!。，,.")
        if q == e:
            return True
        if q in e or e in q:
            return True
        return False

    def _calculate_tfidf_score(self, question):
        """计算问题与知识库的 TF-IDF 相似度，返回最高分及索引。"""
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return 0.0, -1
        processed = self.preprocessor.process(question)
        if not processed:
            return 0.0, -1
        q_vec = self._tfidf_vectorizer.transform([processed])
        similarities = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])
        return best_score, best_idx

    def match(self, question):
        """核心匹配方法，返回 (best_score, best_answer)。"""
        if not question or not question.strip():
            return 0.0, ""

        question = question.strip()

        # 精确匹配优先
        for qa in self.qa_pairs:
            if self._check_exact_match(question, qa):
                return 1.0, qa["answer"]

        # TF-IDF 相似度
        tfidf_score, tfidf_idx = self._calculate_tfidf_score(question)

        best_score = 0.0
        best_answer = self.default_response

        for i, qa in enumerate(self.qa_pairs):
            keyword_score = self._calculate_keyword_score(question, qa)
            exact_bonus = (
                1.0 if self._check_exact_match(question, qa) else 0.0
            )

            qa_tfidf = 0.0
            if tfidf_idx == i:
                qa_tfidf = tfidf_score
            elif tfidf_idx >= 0:
                processed = self.preprocessor.process(question)
                if processed and self._tfidf_vectorizer is not None:
                    q_vec = self._tfidf_vectorizer.transform([processed])
                    qa_tfidf = float(
                        cosine_similarity(
                            q_vec, self._tfidf_matrix[i]
                        ).flatten()[0]
                    )

            final_score = (
                self.ALPHA * qa_tfidf
                + self.BETA * keyword_score
                + self.GAMMA * exact_bonus
            )

            if final_score > best_score:
                best_score = final_score
                best_answer = qa["answer"]

        return best_score, best_answer

    def get_quick_questions(self):
        """获取快捷问题列表。"""
        return [qa["question"] for qa in self.qa_pairs[:6]]


class HybridAgent:
    """双层混合Agent：规则层 + LLM层，通过意图路由器调度。"""

    def __init__(self, knowledge_base_path=None, lang=None):
        self._lang = lang or config.DEFAULT_LANGUAGE
        self.rule_agent = GestureAgent(knowledge_base_path, lang=self._lang)
        self.llm_layer = LLMLayer()
        self.router = IntentRouter(self.rule_agent, self.llm_layer)
        self.tools = TOOL_REGISTRY

    def reload(self, lang):
        """切换语言时重新加载对应知识库和重建索引。

        Args:
            lang: 新的语言代码
        """
        self._lang = lang
        self.rule_agent = GestureAgent(lang=lang)
        self.router = IntentRouter(self.rule_agent, self.llm_layer)
        # 切换语言时清空 LLM 对话历史，避免旧语言上下文干扰
        self.llm_layer.clear_history()
        # 重新加载工具注册表（关键词可能随语言变化）
        from core.tools import build_tool_registry
        self.tools = build_tool_registry()

    def _select_and_run_tools(self, question):
        """根据问题关键词选择合适的工具并执行。

        Args:
            question: 用户问题

        Returns:
            str: 检索到的上下文文本，失败时返回空字符串
        """
        if not question or not config.AGENT_RAG_ENABLED:
            return ""

        question_lower = question.lower()

        # 代码查询特殊处理：提取函数/类名
        code_keywords = t("keywords.code").split(",")
        code_keywords = [kw.strip() for kw in code_keywords]
        if any(kw in question for kw in code_keywords):
            import re
            func_match = re.search(
                r"[a-zA-Z_][a-zA-Z0-9_]*", question
            )
            if func_match:
                func_name = func_match.group(0)
                skip_words = {"how", "what", "is", "the", "function",
                              "class", "code", "def", "this"}
                if func_name.lower() not in skip_words:
                    try:
                        code_tool = next(
                            (t2 for t2 in self.tools
                             if t2["name"] == "query_code"), None
                        )
                        if code_tool is None:
                            return ""
                        result = code_tool["func"](func_name)
                        if t("tools.func_not_found", name="") not in result:
                            return result
                    except Exception as e:
                        logger.warning("query_code failed: %s", e)

        # 按关键词匹配工具（跳过最后的兜底工具）
        for tool in self.tools[:-1]:
            for kw in tool["keywords"]:
                if kw in question_lower:
                    try:
                        result = tool["func"](question)
                        if result and not self._is_tool_error(result):
                            return result
                    except Exception as e:
                        logger.warning("Tool %s failed: %s",
                                       tool["name"], e)

        # 兜底：使用知识库检索
        try:
            result = self.tools[-1]["func"](question)
            if result and not self._is_tool_error(result):
                return result
            return ""
        except Exception as e:
            logger.warning("search_knowledge_base failed: %s", e)
            return ""

    @staticmethod
    def _is_tool_error(result):
        """判断工具返回结果是否为错误/空结果。"""
        if not result or not result.strip():
            return True
        error_markers = [
            t("tools.query_error", error=""),
            t("tools.kb_not_found"),
            t("tools.kb_empty"),
            t("tools.kb_not_found_generic"),
            t("tools.kb_query_error", error=""),
            t("tools.query_model_error", error=""),
            t("tools.model_data_not_found"),
        ]
        for marker in error_markers:
            if marker and marker in result:
                return True
        return False

    def answer(self, question, lang=None):
        """回答用户问题。

        Args:
            question: 用户输入的问题字符串
            lang: 目标回答语言代码，默认使用实例当前语言

        Returns:
            str: 回答内容
        """
        target_lang = lang or self._lang
        try:
            # 先执行工具检索获取上下文
            context = self._select_and_run_tools(question)
            # 将上下文传递给路由器，由路由器决定走规则层还是LLM层
            answer_text, _source = self.router.route(
                question, context=context, lang=target_lang
            )
            return answer_text
        except Exception:
            return t("agent.answer_gen_failed")

    def get_quick_questions(self):
        """获取快捷问题列表。"""
        return self.rule_agent.get_quick_questions()

    def clear_history(self):
        """清空LLM对话历史。"""
        self.llm_layer.clear_history()
