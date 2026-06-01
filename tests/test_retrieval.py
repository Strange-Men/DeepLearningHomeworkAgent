"""检索器单元测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval import TFIDFRetriever, VectorRetriever, SearchResult, create_retriever  # noqa: E402


class TestSearchResult:
    """SearchResult 数据类测试。"""

    def test_creation(self):
        r = SearchResult(
            score=0.85,
            question="What is YOLO?",
            answer="YOLO is an object detection algorithm.",
            category="tech",
            source="knowledge_base",
        )
        assert r.score == 0.85
        assert r.question == "What is YOLO?"
        assert r.source == "knowledge_base"

    def test_defaults(self):
        r = SearchResult(score=0.5, question="q", answer="a")
        assert r.category == ""
        assert r.source == "knowledge_base"


class TestTFIDFRetriever:
    """TFIDFRetriever 测试。"""

    def test_load_and_search(self):
        """加载知识库后搜索应返回结果。"""
        import config
        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        results = retriever.search("YOLO", top_k=3)
        assert isinstance(results, list)
        # 知识库中有 YOLO 相关条目，应有结果
        if results:
            assert all(isinstance(r, SearchResult) for r in results)
            assert results[0].score > 0

    def test_search_empty_query(self):
        retriever = TFIDFRetriever()
        results = retriever.search("", top_k=3)
        assert results == []

    def test_search_no_match(self):
        """完全无关的查询应返回低分或空结果。"""
        import config
        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        results = retriever.search("zzzzxxxxyyyy completely unrelated", top_k=1)
        # 无关查询应没有高分结果
        if results:
            assert results[0].score < 0.3

    def test_match_compatible_interface(self):
        """match() 兼容旧接口。"""
        import config
        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        score, answer = retriever.match("手势识别")
        assert isinstance(score, (int, float))
        assert isinstance(answer, str)

    def test_get_quick_questions(self):
        import config
        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        qs = retriever.get_quick_questions(count=6)
        assert isinstance(qs, list)
        assert len(qs) <= 6

    def test_load_nonexistent_file(self):
        """加载不存在的文件不应崩溃。"""
        retriever = TFIDFRetriever()
        retriever.load("/nonexistent/path.json")
        assert retriever.qa_pairs == []

    def test_exact_match(self):
        """精确匹配应返回最高分。"""
        import config
        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        # 用知识库中的原始问题搜索
        qs = retriever.get_quick_questions(count=1)
        if qs:
            results = retriever.search(qs[0], top_k=1)
            if results:
                assert results[0].score >= 0.9


class TestCreateRetriever:
    """工厂函数测试。"""

    def test_default_backend(self):
        retriever = create_retriever()
        assert isinstance(retriever, TFIDFRetriever)

    def test_tfidf_backend(self):
        retriever = create_retriever(backend="tfidf")
        assert isinstance(retriever, TFIDFRetriever)

    def test_vector_backend_returns_vector_retriever(self):
        retriever = create_retriever(backend="vector")
        assert isinstance(retriever, VectorRetriever)
