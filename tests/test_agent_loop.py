"""LangChainAgent 单元测试 - 使用 Mock 验证四级降级链逻辑。"""

from unittest.mock import MagicMock

from tests.conftest import MockLLM, FailingLLM
from core.agent import LangChainAgent, GestureAgent, _FallbackAgent


# ── 辅助工厂 ─────────────────────────────────────────────


def _make_agent(llm=None, lc_llm=None, followup_llm=None,
                tfidf_retriever=None, rag_retriever=None):
    """创建带 Mock 组件的 LangChainAgent（跳过 __init__）。"""
    agent = LangChainAgent.__new__(LangChainAgent)
    agent._lang = "zh-CN"
    agent._llm = llm or MockLLM(["mock answer"])
    agent._lc_llm = lc_llm
    agent._followup_llm = followup_llm or lc_llm
    agent._tools = []
    agent._tool_map = {}
    agent._tfidf_retriever = tfidf_retriever or MagicMock()
    agent._rag_retriever = rag_retriever or MagicMock()
    return agent


def _mock_lc_llm_response(text):
    """创建返回固定文本的 Mock LangChain LLM。"""
    mock = MagicMock()
    resp = MagicMock()
    resp.content = text
    mock.invoke.return_value = resp
    return mock


# ── LangChainAgent 四级降级链测试 ────────────────────────


class TestLangChainAgentFallback:
    """测试 LangChainAgent 的四级降级链行为。"""

    def test_l1_langchain_agent_success(self):
        """L1: LLM 直接返回 Final Answer。"""
        lc_llm = _mock_lc_llm_response(
            "Thought: 我来回答\nFinal Answer: This is the L1 answer."
        )
        agent = _make_agent(lc_llm=lc_llm)

        result = agent.answer("What gestures are supported?")

        assert result == "This is the L1 answer."

    def test_l1_fails_falls_to_l2(self):
        """L1 异常 → 降级到 L2 纯 LLM 回答。"""
        lc_llm = MagicMock()
        lc_llm.invoke.side_effect = Exception("LLM failed")
        mock_llm = MockLLM(["L2 plain answer"])
        agent = _make_agent(llm=mock_llm, lc_llm=lc_llm)

        result = agent.answer("test question")

        assert result == "L2 plain answer"

    def test_l1_returns_empty_falls_to_l2(self):
        """L1 返回空字符串 → 降级到 L2。"""
        lc_llm = _mock_lc_llm_response("  ")
        mock_llm = MockLLM(["L2 answer"])
        agent = _make_agent(llm=mock_llm, lc_llm=lc_llm)

        result = agent.answer("test")

        assert result == "L2 answer"

    def test_l1_none_falls_to_l2(self):
        """L1 lc_llm 为 None → 降级到 L2。"""
        mock_llm = MockLLM(["L2 fallback"])
        agent = _make_agent(llm=mock_llm, lc_llm=None)

        result = agent.answer("test")

        assert result == "L2 fallback"

    def test_l2_unavailable_falls_to_l3(self):
        """L1 失败 + L2 不可用 → 降级到 L3 TF-IDF 匹配。"""
        lc_llm = MagicMock()
        lc_llm.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.8, "L3 TF-IDF answer")
        agent = _make_agent(
            llm=failing_llm,
            lc_llm=lc_llm,
            tfidf_retriever=mock_retriever,
        )

        result = agent.answer("test question")

        assert "L3 TF-IDF answer" in result

    def test_l3_low_score_falls_to_l4(self):
        """L1/L2 失败 + L3 分数过低 → 降级到 L4 默认兜底。"""
        lc_llm = MagicMock()
        lc_llm.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.05, "weak match")
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = ["Q1", "Q2"]
        agent = _make_agent(
            llm=failing_llm,
            lc_llm=lc_llm,
            tfidf_retriever=mock_retriever,
            rag_retriever=mock_rag,
        )

        result = agent.answer("completely unrelated gibberish")

        assert result is not None
        assert "Q1" in result or "default" in result.lower() \
            or "抱歉" in result or "Sorry" in result

    def test_l3_high_score_returns_with_hint(self):
        """L3 匹配成功时追加降级提示。"""
        lc_llm = MagicMock()
        lc_llm.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.9, "KB answer here")
        agent = _make_agent(
            llm=failing_llm,
            lc_llm=lc_llm,
            tfidf_retriever=mock_retriever,
        )

        result = agent.answer("gesture question")

        assert "KB answer here" in result
        assert "degradation_hint" in result \
            or "注" in result \
            or "Note" in result

    def test_all_levels_fail_returns_default(self):
        """所有级别都失败 → 返回默认兜底回答。"""
        lc_llm = MagicMock()
        lc_llm.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        failing_llm.is_available = False
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.01, "")
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = []
        agent = _make_agent(
            llm=failing_llm,
            lc_llm=lc_llm,
            tfidf_retriever=mock_retriever,
            rag_retriever=mock_rag,
        )

        result = agent.answer("zzz unrelated")

        assert result is not None
        assert len(result) > 0


# ── LangChainAgent 接口测试 ─────────────────────────────


class TestLangChainAgentInterface:
    """测试 LangChainAgent 的公共接口。"""

    def test_get_quick_questions_from_rag(self):
        """优先从 RAG retriever 获取快捷问题。"""
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = ["Q1", "Q2", "Q3"]
        agent = _make_agent(rag_retriever=mock_rag)

        qs = agent.get_quick_questions()

        assert qs == ["Q1", "Q2", "Q3"]
        mock_rag.get_quick_questions.assert_called_once_with(count=6)

    def test_get_quick_questions_fallback_to_tfidf(self):
        """RAG 无结果时降级到 TF-IDF retriever。"""
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = []
        mock_tfidf = MagicMock()
        mock_tfidf.get_quick_questions.return_value = ["TF1", "TF2"]
        agent = _make_agent(
            rag_retriever=mock_rag, tfidf_retriever=mock_tfidf
        )

        qs = agent.get_quick_questions()

        assert qs == ["TF1", "TF2"]

    def test_clear_history(self):
        """clear_history 调用 LLM 的 clear_history。"""
        mock_llm = MagicMock()
        agent = _make_agent(llm=mock_llm)

        agent.clear_history()

        mock_llm.clear_history.assert_called_once()

    def test_reload_rebuilds_components(self):
        """reload 重建索引和工具集。"""
        agent = _make_agent()
        agent._build_rag_index = MagicMock()
        agent._load_tfidf_kb = MagicMock()
        agent._build_tools = MagicMock(
            return_value=([], {})
        )
        agent._llm = MagicMock()

        agent.reload("en")

        assert agent._lang == "en"
        agent._build_rag_index.assert_called_once()
        agent._load_tfidf_kb.assert_called_once()
        agent._build_tools.assert_called_once()
        agent._llm.clear_history.assert_called_once()

    def test_answer_lang_parameter(self):
        """answer() 传递 lang 参数到 L1。"""
        lc_llm = _mock_lc_llm_response(
            "Thought: ok\nFinal Answer: English answer"
        )
        agent = _make_agent(lc_llm=lc_llm)

        result = agent.answer("What is this?", lang="en")

        assert result == "English answer"


# ── GestureAgent 规则层测试 ─────────────────────────────


class TestGestureAgent:
    """测试规则层 TF-IDF 匹配。"""

    def test_match_returns_tuple(self):
        agent = GestureAgent()
        score, answer = agent.match("test question")
        assert isinstance(score, (int, float))
        assert isinstance(answer, str)

    def test_get_quick_questions(self):
        agent = GestureAgent()
        qs = agent.get_quick_questions()
        assert isinstance(qs, list)


# ── _FallbackAgent 测试 ─────────────────────────────────


class TestFallbackAgent:
    """测试初始化失败兜底 Agent。"""

    def test_answer_returns_string(self):
        fb = _FallbackAgent()
        result = fb.answer("test")
        assert result is not None
        assert isinstance(result, str)

    def test_get_quick_questions_returns_empty(self):
        fb = _FallbackAgent()
        assert fb.get_quick_questions() == []

    def test_reload_noop(self):
        fb = _FallbackAgent()
        fb.reload("en")  # Should not raise

    def test_clear_history_noop(self):
        fb = _FallbackAgent()
        fb.clear_history()  # Should not raise
