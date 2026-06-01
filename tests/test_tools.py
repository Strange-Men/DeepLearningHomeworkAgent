"""工具函数单元测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools import (  # noqa: E402
    query_history,
    query_model_metrics,
    query_code,
    build_tool_registry,
    TOOL_SCHEMAS,
)


class TestQueryHistory:
    """query_history 测试。"""

    def test_returns_string(self):
        result = query_history("today")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_time_keywords(self):
        """不同时间关键词不应崩溃。"""
        for keyword in ["今天", "yesterday", "hier", "本周"]:
            result = query_history(keyword)
            assert isinstance(result, str)


class TestQueryModelMetrics:
    """query_model_metrics 测试。"""

    def test_returns_string(self):
        result = query_model_metrics("model info")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_metrics(self):
        """应包含模型相关关键词。"""
        result = query_model_metrics("精度")
        # 结果中应有 Model 或 mAP 等关键词
        assert any(
            kw in result
            for kw in ["Model", "mAP", "model", "未找到"]
        )


class TestQueryCode:
    """query_code 测试。"""

    def test_existing_function(self):
        """搜索存在的函数应返回结果。"""
        result = query_code("detect_image")
        assert isinstance(result, str)
        assert "detect_image" in result

    def test_nonexistent_function(self):
        """搜索不存在的函数应返回未找到提示。"""
        result = query_code("nonexistent_function_xyz")
        assert isinstance(result, str)

    def test_empty_input(self):
        result = query_code("")
        assert isinstance(result, str)

    def test_class_search(self):
        """搜索类名应返回结果。"""
        result = query_code("GestureDetector")
        assert isinstance(result, str)
        if "GestureDetector" in result:
            assert "class" in result


class TestToolSchemas:
    """工具 Schema 测试。"""

    def test_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_schema_count(self):
        assert len(TOOL_SCHEMAS) == 4


class TestBuildToolRegistry:
    """build_tool_registry 测试。"""

    def test_returns_list(self):
        registry = build_tool_registry()
        assert isinstance(registry, list)
        assert len(registry) == 4

    def test_registry_items_have_keys(self):
        registry = build_tool_registry()
        for item in registry:
            assert "name" in item
            assert "func" in item
            assert "schema" in item
            assert callable(item["func"])

    def test_search_kb_with_retriever(self):
        """传入 retriever 时 search_knowledge_base 应可调用。"""
        from core.retrieval import TFIDFRetriever
        import config

        retriever = TFIDFRetriever()
        retriever.load(config.KNOWLEDGE_BASE_PATH, lang="zh-CN")

        registry = build_tool_registry(retriever=retriever)
        kb_tool = next(t for t in registry if t["name"] == "search_knowledge_base")

        result = kb_tool["func"]("YOLO")
        assert isinstance(result, str)

    def test_search_kb_without_retriever(self):
        """不传 retriever 时 search_knowledge_base 应返回空字符串。"""
        registry = build_tool_registry(retriever=None)
        kb_tool = next(t for t in registry if t["name"] == "search_knowledge_base")

        result = kb_tool["func"]("YOLO")
        assert result == ""
