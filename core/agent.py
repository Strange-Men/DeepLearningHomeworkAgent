"""Agent问答引擎模块 - 双层混合架构（规则层 + LLM层）。"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from core.text_preprocessor import TextPreprocessor
from core.llm_layer import LLMLayer
from core.intent_router import IntentRouter


class GestureAgent:
    """规则层：基于TF-IDF + 关键词融合的问答Agent。"""

    # 融合权重
    ALPHA = 0.6   # TF-IDF 权重
    BETA = 0.3    # 关键词匹配权重
    GAMMA = 0.1   # 精确匹配加分

    def __init__(self, knowledge_base_path=None):
        self.kb_path = knowledge_base_path or config.KNOWLEDGE_BASE_PATH
        self.qa_pairs = []
        self.default_response = ""
        self.preprocessor = TextPreprocessor()
        self._load_knowledge_base()
        self._build_tfidf_index()

    def _load_knowledge_base(self):
        """加载知识库。"""
        if not os.path.exists(self.kb_path):
            return
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.qa_pairs = data.get("qa_pairs", [])
        self.default_response = data.get(
            "default_response",
            "Sorry, I cannot understand your question."
        )
        if self.qa_pairs:
            examples = "\n".join(
                f"  {i}. {qa.get('question', '')}"
                for i, qa in enumerate(self.qa_pairs, 1)
                if qa.get("question")
            )
            self.default_response = (
                "Sorry, I cannot understand your question. "
                "Try these:\n\n"
                f"{examples}\n\n"
                "Type a related question and I will do my best!"
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

    def __init__(self, knowledge_base_path=None):
        self.rule_agent = GestureAgent(knowledge_base_path)
        self.llm_layer = LLMLayer()
        self.router = IntentRouter(self.rule_agent, self.llm_layer)

    def answer(self, question):
        """回答用户问题。

        Args:
            question: 用户输入的问题字符串

        Returns:
            str: 回答内容
        """
        try:
            answer_text, source = self.router.route(question)
            return answer_text
        except Exception:
            return "Failed to generate answer. Please try again later."

    def get_quick_questions(self):
        """获取快捷问题列表。"""
        return self.rule_agent.get_quick_questions()

    def clear_history(self):
        """清空LLM对话历史。"""
        self.llm_layer.clear_history()
