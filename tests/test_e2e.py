"""E2E 验证脚本 - 逐项测试 LangChainAgent 链路（Mock LLM，无真实 API）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock  # noqa: E402

from tests.conftest import MockLLM, FailingLLM  # noqa: E402
from core.agent import LangChainAgent, _FallbackAgent  # noqa: E402


# ── 辅助函数 ─────────────────────────────────────────────

_RESULTS = []


def _report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    _RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def _make_agent(llm=None, tfidf_retriever=None,
                rag_retriever=None, lang="zh-CN"):
    """创建带 Mock 组件的 LangChainAgent。"""
    agent = LangChainAgent.__new__(LangChainAgent)
    agent._lang = lang
    agent._llm = llm or MockLLM(["mock answer"])
    agent._tfidf_retriever = tfidf_retriever or MagicMock()
    agent._rag_retriever = rag_retriever or MagicMock()
    return agent


# ── 场景 1: Agent 初始化成功 ─────────────────────────────


def test_1_agent_init():
    """LangChainAgent 能正常创建，公共接口可用。"""
    name = "Agent 初始化成功"
    try:
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = ["Q1", "Q2"]
        agent = _make_agent(rag_retriever=mock_rag)

        # Verify public interface exists
        assert hasattr(agent, "answer")
        assert hasattr(agent, "reload")
        assert hasattr(agent, "get_quick_questions")
        assert hasattr(agent, "clear_history")

        qs = agent.get_quick_questions()
        assert isinstance(qs, list)

        _report(name, True, f"interfaces OK, {len(qs)} quick questions")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 2: L1 LangChain Agent 工具调用 ──────────────────


def test_2_langchain_agent_tool_call():
    """L1: LangChain Agent 调用工具并返回结果。"""
    name = "L1 LangChain Agent 工具调用"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "系统支持 10 类手势：A, D, I, L, V, W, Y, 5, 7, ILY。"
        }
        agent = _make_agent(agent_executor=mock_executor)

        result = agent.answer("支持哪些手势？")

        if result and "手势" in result:
            _report(name, True, "L1 returned gesture info")
        else:
            _report(name, False, f"unexpected: {result!r}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 3: L2 纯 LLM 回答 ──────────────────────────────


def test_3_plain_llm_answer():
    """L1 失败 → L2 纯 LLM 回答。"""
    name = "L2 纯 LLM 回答"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("Agent error")
        mock_llm = MockLLM(["这是一个手势识别系统。"])
        agent = _make_agent(llm=mock_llm, agent_executor=mock_executor)

        result = agent.answer("这是什么系统？")

        if result and "手势" in result:
            _report(name, True, "L2 returned answer")
        else:
            _report(name, False, f"unexpected: {result!r}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 4: L3 TF-IDF 规则层匹配 ────────────────────────


def test_4_tfidf_fallback():
    """L1/L2 失败 → L3 TF-IDF 匹配知识库。"""
    name = "L3 TF-IDF 规则层"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (
            0.85, "系统支持 A, D, I, L, V, W, Y 等手势。"
        )
        agent = _make_agent(
            llm=failing_llm,
            agent_executor=mock_executor,
            tfidf_retriever=mock_retriever,
        )

        result = agent.answer("支持哪些手势")

        if result and "手势" in result:
            _report(name, True, "L3 matched knowledge base")
        else:
            _report(name, False, f"unexpected: {result!r}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 5: L4 默认兜底 ─────────────────────────────────


def test_5_default_fallback():
    """所有级别失败 → L4 默认兜底回答。"""
    name = "L4 默认兜底"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.01, "")
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = [
            "支持哪些手势？", "如何使用摄像头？"
        ]
        agent = _make_agent(
            llm=failing_llm,
            agent_executor=mock_executor,
            tfidf_retriever=mock_retriever,
            rag_retriever=mock_rag,
        )

        result = agent.answer("zzzzxxxxyyyy gibberish")

        if result is not None and len(result) > 0:
            _report(name, True, "got default fallback answer")
        else:
            _report(name, False, "no default fallback answer")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 6: LLM 直接回答（无工具调用）────────────────────


def test_6_direct_answer():
    """L1 Agent 直接返回回答，不调用工具。"""
    name = "LLM 直接回答"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "你好！我是手势识别助手。"
        }
        agent = _make_agent(agent_executor=mock_executor)

        result = agent.answer("你好")

        if result and "助手" in result:
            _report(name, True, "direct answer returned")
        else:
            _report(name, False, f"unexpected: {result!r}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 7: 降级提示追加 ─────────────────────────────────


def test_7_degradation_hint():
    """L3 匹配成功时追加降级提示。"""
    name = "降级提示追加"
    try:
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("fail")
        failing_llm = FailingLLM()
        mock_retriever = MagicMock()
        mock_retriever.match.return_value = (0.9, "知识库回答内容")
        agent = _make_agent(
            llm=failing_llm,
            agent_executor=mock_executor,
            tfidf_retriever=mock_retriever,
        )

        result = agent.answer("test")

        has_hint = "注" in result or "Note" in result
        if "知识库回答内容" in result and has_hint:
            _report(name, True, "answer + degradation hint present")
        else:
            _report(name, False, f"hint missing: {result!r}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 8: 快捷问题获取 ─────────────────────────────────


def test_8_quick_questions():
    """get_quick_questions 从 RAG 降级到 TF-IDF。"""
    name = "快捷问题获取"
    try:
        # RAG has questions
        mock_rag = MagicMock()
        mock_rag.get_quick_questions.return_value = ["Q1", "Q2", "Q3"]
        agent1 = _make_agent(rag_retriever=mock_rag)
        qs1 = agent1.get_quick_questions()

        # RAG empty, fallback to TF-IDF
        mock_rag2 = MagicMock()
        mock_rag2.get_quick_questions.return_value = []
        mock_tfidf = MagicMock()
        mock_tfidf.get_quick_questions.return_value = ["TF1", "TF2"]
        agent2 = _make_agent(
            rag_retriever=mock_rag2, tfidf_retriever=mock_tfidf
        )
        qs2 = agent2.get_quick_questions()

        if qs1 == ["Q1", "Q2", "Q3"] and qs2 == ["TF1", "TF2"]:
            _report(name, True, f"RAG:{len(qs1)}, TF-IDF:{len(qs2)}")
        else:
            _report(name, False, f"qs1={qs1}, qs2={qs2}")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 9: 语言切换 ─────────────────────────────────────


def test_9_language_reload():
    """reload() 切换语言不报错。"""
    name = "语言切换"
    all_passed = True
    details = []

    for lang in ["en", "fr", "zh-CN", "zh-TW"]:
        try:
            agent = _make_agent(lang=lang)
            agent._build_rag_index = MagicMock()
            agent._load_tfidf_kb = MagicMock()
            agent._llm = MagicMock()
            agent.reload(lang)
            details.append(f"{lang}:OK")
        except Exception as e:
            all_passed = False
            details.append(f"{lang}:FAIL({e})")

    _report(name, all_passed, ", ".join(details))


# ── 场景 10: clear_history ───────────────────────────────


def test_10_clear_history():
    """clear_history 调用 LLM 清空历史。"""
    name = "清空对话历史"
    try:
        mock_llm = MagicMock()
        agent = _make_agent(llm=mock_llm)
        agent.clear_history()
        mock_llm.clear_history.assert_called_once()
        _report(name, True, "clear_history called")
    except Exception as e:
        _report(name, False, str(e))


# ── 场景 11: _FallbackAgent ──────────────────────────────


def test_11_fallback_agent():
    """_FallbackAgent 返回初始化失败消息。"""
    name = "_FallbackAgent 兜底"
    try:
        fb = _FallbackAgent()
        result = fb.answer("test")
        if result is not None and len(result) > 0:
            _report(name, True, "fallback agent returned message")
        else:
            _report(name, False, "empty response")
    except Exception as e:
        _report(name, False, str(e))


# ── 主入口 ───────────────────────────────────────────────


def run_all():
    print("=" * 60)
    print("E2E 验证报告 (LangChainAgent v3)")
    print("=" * 60)

    test_1_agent_init()
    test_2_langchain_agent_tool_call()
    test_3_plain_llm_answer()
    test_4_tfidf_fallback()
    test_5_default_fallback()
    test_6_direct_answer()
    test_7_degradation_hint()
    test_8_quick_questions()
    test_9_language_reload()
    test_10_clear_history()
    test_11_fallback_agent()

    print("\n" + "-" * 60)
    passed = sum(1 for _, s, _ in _RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in _RESULTS if s == "FAIL")
    print(f"Total: {len(_RESULTS)} | PASS: {passed} | FAIL: {failed}")
    print("-" * 60)

    if failed:
        print("\nFailed tests:")
        for name, status, detail in _RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {detail}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
