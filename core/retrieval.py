"""知识库检索模块 - 提供 TF-IDF / 向量检索的统一抽象接口。"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from core.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果数据类。"""
    score: float
    question: str
    answer: str
    category: str = ""
    source: str = "knowledge_base"


class BaseRetriever(ABC):
    """检索器抽象基类。"""

    @abstractmethod
    def load(self, kb_path: str, lang: str = None) -> None:
        """加载知识库。"""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list:
        """检索知识库，返回按相关性排序的 SearchResult 列表。"""
        ...

    @abstractmethod
    def get_quick_questions(self, count: int = 6) -> list:
        """获取快捷问题列表。"""
        ...

    @abstractmethod
    def get_default_response(self) -> str:
        """获取默认兜底回答。"""
        ...


class TFIDFRetriever(BaseRetriever):
    """基于 TF-IDF + 关键词融合的检索器。"""

    ALPHA = 0.6   # TF-IDF 权重
    BETA = 0.3    # 关键词匹配权重
    GAMMA = 0.1   # 精确匹配加分

    def __init__(self):
        self.qa_pairs: list = []
        self.default_response: str = ""
        self.preprocessor: TextPreprocessor = None
        self._tfidf_vectorizer: TfidfVectorizer = None
        self._tfidf_matrix = None
        self._lang: str = config.DEFAULT_LANGUAGE

    def load(self, kb_path: str, lang: str = None) -> None:
        """加载知识库并构建 TF-IDF 索引。"""
        self._lang = lang or config.DEFAULT_LANGUAGE
        self.preprocessor = TextPreprocessor(lang=self._lang)

        if not os.path.exists(kb_path):
            logger.warning("Knowledge base not found: %s", kb_path)
            self.qa_pairs = []
            self._build_index()
            return

        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.qa_pairs = data.get("qa_pairs", [])
        self.default_response = data.get("default_response", "")
        self._build_index()

        logger.info("Loaded %d QA pairs from %s", len(self.qa_pairs), kb_path)

    def _build_index(self) -> None:
        """构建 TF-IDF 索引。"""
        if not self.qa_pairs:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        corpus = []
        for qa in self.qa_pairs:
            question = qa.get("question", "")
            expanded = qa.get("keywords", []) + qa.get("synonyms", [])
            text = question + " " + " ".join(expanded)
            processed = self.preprocessor.process(text)
            corpus.append(processed if processed else question)

        self._tfidf_vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def _calculate_keyword_score(self, question: str, qa: dict) -> float:
        """计算问题与扩展关键词的匹配分数。"""
        expanded = qa.get("keywords", []) + qa.get("synonyms", [])
        if not expanded:
            return 0.0
        question_lower = question.lower()
        matched = sum(1 for kw in expanded if kw.lower() in question_lower)
        if matched == 0:
            return 0.0
        return matched / max(len(expanded) ** 0.5, 1)

    def _check_exact_match(self, question: str, qa: dict) -> bool:
        """检查是否精确匹配。"""
        example = qa.get("question", "")
        if not example:
            return False
        q = question.strip().rstrip("?！!。，,.")
        e = example.rstrip("?！!。，,.")
        return q == e or q in e or e in q

    def _calculate_tfidf_score(self, question: str):
        """计算问题与知识库的 TF-IDF 相似度，返回 (最高分, 索引)。"""
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return 0.0, -1
        processed = self.preprocessor.process(question)
        if not processed:
            return 0.0, -1
        q_vec = self._tfidf_vectorizer.transform([processed])
        similarities = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        return best_score, best_idx

    def search(self, query: str, top_k: int = 3) -> list:
        """检索知识库，返回按相关性排序的 SearchResult 列表。"""
        if not query or not query.strip():
            return []

        query = query.strip()

        # 精确匹配优先
        for qa in self.qa_pairs:
            if self._check_exact_match(query, qa):
                return [SearchResult(
                    score=1.0,
                    question=qa.get("question", ""),
                    answer=qa["answer"],
                    category=qa.get("category", ""),
                    source="knowledge_base",
                )]

        # TF-IDF + 关键词融合评分
        tfidf_score, tfidf_idx = self._calculate_tfidf_score(query)
        results = []

        for i, qa in enumerate(self.qa_pairs):
            keyword_score = self._calculate_keyword_score(query, qa)
            exact_bonus = 1.0 if self._check_exact_match(query, qa) else 0.0

            qa_tfidf = 0.0
            if tfidf_idx == i:
                qa_tfidf = tfidf_score
            elif tfidf_idx >= 0:
                processed = self.preprocessor.process(query)
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

            if final_score > 0:
                results.append(SearchResult(
                    score=final_score,
                    question=qa.get("question", ""),
                    answer=qa["answer"],
                    category=qa.get("category", ""),
                    source="knowledge_base",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def match(self, question: str):
        """兼容旧接口：返回 (best_score, best_answer)。"""
        results = self.search(question, top_k=1)
        if results:
            return results[0].score, results[0].answer
        return 0.0, self.default_response

    def get_quick_questions(self, count: int = 6) -> list:
        """获取快捷问题列表。"""
        return [qa["question"] for qa in self.qa_pairs[:count]]

    def get_default_response(self) -> str:
        """获取默认兜底回答。"""
        return self.default_response


class VectorRetriever(BaseRetriever):
    """基于 ChromaDB 向量检索的适配器（包装 ChromaRetriever）。"""

    def __init__(self):
        from core.rag_retriever import ChromaRetriever
        self._retriever = ChromaRetriever()
        self._default_response = ""

    def load(self, kb_path: str, lang: str = None) -> None:
        """构建向量索引。"""
        self._retriever.build_index(lang=lang)
        self._default_response = self._retriever.get_default_response()

    def search(self, query: str, top_k: int = 3) -> list:
        """向量检索，返回 SearchResult 列表。"""
        results = self._retriever.search(query, top_k=top_k)
        return [
            SearchResult(
                score=r["score"],
                question="",
                answer=r["text"],
                source="vector",
            )
            for r in results
        ]

    def match(self, question: str):
        """兼容旧接口：返回 (best_score, best_answer)。"""
        results = self.search(question, top_k=1)
        if results and results[0].score >= 0.2:
            return results[0].score, results[0].answer
        return 0.0, self._default_response

    def get_quick_questions(self, count: int = 6) -> list:
        return self._retriever.get_quick_questions(count=count)

    def get_default_response(self) -> str:
        return self._default_response


def create_retriever(backend: str = None):
    """检索器工厂函数。

    Args:
        backend: 检索后端，可选 "tfidf" / "vector"。默认从 config 读取。

    Returns:
        BaseRetriever 实例
    """
    backend = backend or getattr(config, "RETRIEVER_BACKEND", "tfidf")
    if backend == "vector":
        return VectorRetriever()
    return TFIDFRetriever()
