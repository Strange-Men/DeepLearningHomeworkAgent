"""Agent 问答引擎模块 - LangChain ChatOpenAI + 手动 ReAct 解析 + 四级降级链。

核心组件：
- LangChainAgent: LangChain ChatOpenAI 调用 + 手动工具调度 + 四级降级链
- GestureAgent: 规则层降级（TF-IDF 知识库匹配）
- _FallbackAgent: 初始化失败时的兜底
"""

import logging
import re
import time

import config
from core.i18n import t
from core.llm import LLM, create_langchain_llm, create_followup_llm
from core.rag_retriever import create_rag_retriever
from core.retrieval import create_retriever
from core.tools import build_langchain_tools, query_history

logger = logging.getLogger(__name__)


# ── 直接 LLM Agent ───────────────────────────────────────


class LangChainAgent:
    """LangChain ChatOpenAI + 手动 ReAct 解析 + 多轮工具调用 + 四级降级链。

    L1: LLM + Tools（ReAct 格式，手动解析，支持多轮工具调用）
    L2: LLM without Tools（纯 LLM 回答）
    L3: 规则层 TF-IDF 匹配
    L4: 默认兜底回答
    """

    # 快速路径关键词：命中则跳过 Agent，直接走 L2
    _SIMPLE_PATTERNS = [
        r"系统是什么|这个系统|什么系统|系统介绍|系统功能",
        r"what\s+is\s+this\s+system|describe\s+the\s+system",
        r"^(你好|hello|hi|hey|嗨|您好)",
        r"谢谢|感谢|thank|再见|bye|拜拜",
    ]
    # 工具相关关键词：出现则不走快速路径
    _TOOL_KEYWORDS = [
        "历史", "识别", "记录", "指标", "精度", "map", "代码",
        "函数", "检测", "查询", "训练", "模型", "loss",
    ]
    # 时间相关关键词：命中时预取工具结果
    _TIME_KEYWORDS = [
        "昨天", "今天", "最近", "本周", "昨日", "今日",
        "yesterday", "today", "recent", "this week",
        "hier", "aujourd", "cette semaine",
    ]

    def _is_simple_question(self, question: str) -> bool:
        """判断是否为无需工具的简单问题。"""
        q = question.strip().lower()
        # 含工具关键词 → 不是简单问题
        if any(kw in q for kw in self._TOOL_KEYWORDS):
            return False
        # 命中简单问题模式
        return any(re.search(p, q) for p in self._SIMPLE_PATTERNS)

    def _is_time_query(self, question: str) -> bool:
        """判断是否为包含时间关键词的查询。"""
        q = question.strip().lower()
        return any(kw in q for kw in self._TIME_KEYWORDS)

    def _prefetch_tool_context(self, question: str) -> str:
        """为时间相关查询预取工具结果，拼成上下文字符串。"""
        parts = []
        history_result = query_history(question)
        if history_result:
            parts.append(f"检测历史数据:\n{history_result}")
        return "\n\n".join(parts) if parts else ""

    def __init__(self, lang=None):
        self._lang = lang or config.DEFAULT_LANGUAGE
        self._llm = LLM()
        self._rag_retriever = create_rag_retriever()
        self._tfidf_retriever = create_retriever()

        # 构建 RAG 索引
        self._build_rag_index()

        # 加载 TF-IDF 知识库（L3 降级用）
        self._load_tfidf_kb()

        # 构建工具集和 LangChain LLM（用于 L1）
        self._tools, self._tool_map = self._build_tools()
        self._lc_llm = create_langchain_llm()
        self._followup_llm = create_followup_llm()

    def _resolve_kb_path(self, lang):
        return config.KNOWLEDGE_BASE_PATHS.get(
            lang, config.KNOWLEDGE_BASE_PATH
        )

    def _build_rag_index(self):
        """构建 Chroma 向量索引。"""
        try:
            self._rag_retriever.build_index(lang=self._lang)
            logger.info("RAG index built for lang=%s", self._lang)
        except Exception as e:
            logger.warning("RAG index build failed: %s", e)

    def _load_tfidf_kb(self):
        """加载 TF-IDF 知识库（L3 降级）。"""
        path = self._resolve_kb_path(self._lang)
        self._tfidf_retriever.load(path, lang=self._lang)

    def _build_tools(self):
        """构建工具集和工具名称→函数映射。"""
        tools = build_langchain_tools(retriever=self._rag_retriever)
        tool_map = {}
        for tool in tools:
            tool_map[tool.name] = tool.func
        return tools, tool_map

    def _build_tool_prompt(self) -> str:
        """构建工具描述文本，供 LLM 参考。"""
        lines = []
        for tool in self._tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    # ── 对外接口 ────────────────────────────────────────

    def answer(self, question: str, lang: str = None) -> str:
        """回答用户问题（四级降级链）。"""
        target_lang = lang or self._lang
        t_answer_start = time.time()
        logger.debug("answer() START | question=%r | lang=%s", question, target_lang)

        try:
            # 快速路径：简单问题跳过 Agent，直接走 L2
            t0 = time.time()
            is_simple = self._is_simple_question(question)
            logger.debug("_is_simple_question() => %s | %.3fs", is_simple, time.time()-t0)
            if is_simple:
                t0 = time.time()
                result = self._llm_ask_plain(question, target_lang)
                logger.debug("FAST PATH L2 (plain LLM) => %s | %.3fs", 'HIT' if result else 'MISS', time.time()-t0)
                if result is not None:
                    logger.debug("answer() TOTAL => %.3fs (fast path L2)", time.time()-t_answer_start)
                    return result

            # 快速路径 L2+：时间相关查询 → 预取工具结果 + LLM 直接回答
            t0 = time.time()
            is_time = self._is_time_query(question)
            logger.debug("_is_time_query() => %s | %.3fs", is_time, time.time()-t0)
            if is_time:
                t0 = time.time()
                tool_ctx = self._prefetch_tool_context(question)
                logger.debug("_prefetch_tool_context() => %s | %.3fs", 'OK' if tool_ctx else 'EMPTY', time.time()-t0)
                if tool_ctx:
                    t0 = time.time()
                    result = self._llm_ask_with_context(
                        question, tool_ctx, target_lang
                    )
                    hit = 'HIT' if result else 'MISS'
                    logger.debug("FAST PATH L2+ (prefetch+LLM) => %s | %.3fs", hit, time.time()-t0)
                    if result is not None:
                        logger.debug("answer() TOTAL => %.3fs (fast path L2+)", time.time()-t_answer_start)
                        return result

            # L1: 直接 LLM + 手动工具调度
            t0 = time.time()
            result = self._run_direct_llm_with_tools(question, target_lang)
            logger.debug("L1 (direct LLM+tools) => %s | %.3fs", 'HIT' if result else 'MISS', time.time()-t0)
            if result is not None:
                logger.debug("answer() TOTAL => %.3fs (L1)", time.time()-t_answer_start)
                return result

            # L2: 纯 LLM 回答
            t0 = time.time()
            result = self._llm_ask_plain(question, target_lang)
            logger.debug("L2 (plain LLM) => %s | %.3fs", 'HIT' if result else 'MISS', time.time()-t0)
            if result is not None:
                logger.debug("answer() TOTAL => %.3fs (L2)", time.time()-t_answer_start)
                return result

            # L3: TF-IDF 规则层
            t0 = time.time()
            score, answer = self._tfidf_retriever.match(question)
            logger.debug("L3 (TF-IDF) => score=%.3f | %.3fs", score, time.time()-t0)
            if score >= config.AGENT_SIMILARITY_THRESHOLD:
                logger.debug("answer() TOTAL => %.3fs (L3)", time.time()-t_answer_start)
                return answer + "\n\n" + t("agent.degradation_hint")

            # L4: 默认兜底
            logger.debug("answer() TOTAL => %.3fs (L4 fallback)", time.time()-t_answer_start)
            return self._build_default_response()

        except Exception as e:
            logger.error("answer() unexpected error: %s", e)
            logger.debug("answer() EXCEPTION after %.3fs: %s", time.time()-t_answer_start, e)
            return self._build_default_response()

    def reload(self, lang: str) -> None:
        """切换语言时重新加载。"""
        self._lang = lang
        self._build_rag_index()
        self._load_tfidf_kb()
        self._tools, self._tool_map = self._build_tools()
        self._lc_llm = create_langchain_llm()
        self._followup_llm = create_followup_llm()
        self._llm.clear_history()

    def get_quick_questions(self) -> list:
        """获取快捷问题列表。"""
        questions = self._rag_retriever.get_quick_questions(count=6)
        if questions:
            return questions
        return self._tfidf_retriever.get_quick_questions(count=6)

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._llm.clear_history()

    # ── L1: 直接 LLM + 手动工具调度 ────────────────────

    def _run_direct_llm_with_tools(self, question: str, lang: str) -> str:
        """直接调用 LLM，手动解析工具调用并执行（支持多轮工具调用）。"""
        if self._lc_llm is None:
            logger.debug("_run_direct_llm_with_tools: lc_llm is None, skip")
            return None

        t0 = time.time()
        lang_name = config.LANG_NAMES.get(lang, lang)
        tool_desc = self._build_tool_prompt()
        tool_names = ", ".join(self._tool_map.keys())

        system_prompt = (
            "你是一个手势识别系统的智能助手。请用简洁的中文回答。\n\n"
            f"可用工具:\n{tool_desc}\n\n"
            f"工具名称: [{tool_names}]\n\n"
            "**重要规则：**\n"
            "- 打招呼、闲聊、询问系统功能/简介等通用问题，"
            "直接给出 Final Answer，不要调用任何工具\n"
            "- 只有当问题明确涉及检测历史、模型指标、"
            "代码查询、知识库内容时，才使用工具\n"
            "- 如果需要使用工具，必须严格按以下格式输出，"
            "否则解析器会崩溃：\n"
            "Thought: [你的思考]\n"
            "Action: [工具名]\n"
            "Action Input: [输入]\n\n"
            "- 如果不需要工具，直接按以下格式输出：\n"
            "Thought: [你的思考]\n"
            "Final Answer: [最终回答]\n\n"
            f"- 回答语言：{lang_name}"
        )

        user_msg = f"Question: {question}\nThought:"

        try:
            from langchain_core.messages import HumanMessage

            # 合并 system prompt 到 user message（MiMo 不支持 system 角色）
            combined_msg = f"{system_prompt}\n\n{user_msg}"
            messages = [HumanMessage(content=combined_msg)]

            max_turns = config.AGENT_MAX_TURNS
            tool_results_context = []

            for turn in range(max_turns):
                logger.debug(
                    "_run_direct_llm_with_tools: LLM call turn %d/%d",
                    turn + 1, max_turns
                )
                resp = self._lc_llm.invoke(messages)
                output = resp.content.strip()
                logger.debug(
                    "_run_direct_llm_with_tools: LLM output | %.3fs | %.200r",
                    time.time()-t0, output
                )

                # 尝试解析 Final Answer
                final = self._extract_final_answer(output)
                if final:
                    logger.debug(
                        "_run_direct_llm_with_tools: Final Answer | turn=%d | total=%.3fs",
                        turn + 1, time.time()-t0
                    )
                    return final

                # 尝试解析工具调用
                tool_name, tool_input = self._extract_tool_call(output)
                if tool_name and tool_name in self._tool_map:
                    logger.debug(
                        "_run_direct_llm_with_tools: tool call => %s(%r)",
                        tool_name, tool_input
                    )
                    t_tool = time.time()
                    tool_result = self._tool_map[tool_name](tool_input)
                    logger.debug(
                        "_run_direct_llm_with_tools: tool result | %.3fs | len=%d",
                        time.time()-t_tool, len(str(tool_result))
                    )
                    tool_results_context.append(
                        f"[{tool_name}] {tool_result}"
                    )

                    # 将工具结果注入下一轮对话
                    observation = (
                        f"Observation: {tool_result}\nThought:"
                    )
                    messages = [
                        HumanMessage(
                            content=f"{combined_msg}\n\n"
                            + "\n\n".join(tool_results_context)
                            + f"\n\n{observation}"
                        ),
                    ]
                    continue

                # 没有工具调用也没有 Final Answer → 不再循环
                break

            # 循环结束：用所有工具结果做最终 followup
            if tool_results_context:
                return self._followup_with_tool_results(
                    question, tool_results_context, t0
                )

            # 无工具调用，返回原始输出
            if output and output.strip():
                cleaned = re.sub(
                    r"^Thought:\s*", "", output.strip()
                )
                if cleaned:
                    logger.debug(
                        "_run_direct_llm_with_tools: raw cleaned | total=%.3fs",
                        time.time()-t0
                    )
                    return cleaned

            logger.debug(
                "_run_direct_llm_with_tools: EMPTY | total=%.3fs",
                time.time()-t0
            )
            return None

        except Exception as e:
            logger.debug(
                "_run_direct_llm_with_tools: FAILED | total=%.3fs | %s",
                time.time()-t0, e
            )
            logger.warning("Direct LLM+tools failed: %s", e)
            return None

    def _followup_with_tool_results(
        self, question: str, tool_results: list, t0: float
    ) -> str:
        """用 followup LLM 基于工具结果生成最终回答。"""
        from langchain_core.messages import HumanMessage

        context = "\n\n".join(tool_results)
        followup_prompt = (
            "请直接基于以下检索结果回答用户问题，"
            "不要添加额外信息，用 2-3 句话完成。\n\n"
            f"用户问题: {question}\n\n"
            f"检索结果:\n{context}\n\n"
            "回答:"
        )
        llm = self._followup_llm or self._lc_llm
        logger.debug("_followup_with_tool_results: LLM call START")
        t1 = time.time()
        try:
            resp = llm.invoke([HumanMessage(content=followup_prompt)])
            output = resp.content.strip()
            logger.debug(
                "_followup_with_tool_results: DONE | %.3fs",
                time.time()-t1
            )
            if output:
                final = self._extract_final_answer(output)
                if final:
                    return final
                return output
        except Exception as e:
            logger.debug(
                "_followup_with_tool_results: FAILED | %.3fs | %s",
                time.time()-t1, e
            )
            logger.warning("Followup LLM failed: %s", e)

        # followup 失败 → 直接用工具结果兜底
        fallback = f"根据知识库查询结果：\n\n{context}"
        logger.debug(
            "_followup_with_tool_results: fallback | total=%.3fs",
            time.time()-t0
        )
        return fallback

    def _extract_final_answer(self, text: str) -> str:
        """从 LLM 输出中提取 Final Answer。"""
        match = re.search(
            r"Final\s*Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return ""

    def _extract_tool_call(self, text: str) -> tuple:
        """从 LLM 输出中提取 Action 和 Action Input。"""
        action_match = re.search(
            r"Action:\s*(\S+)", text, re.IGNORECASE
        )
        input_match = re.search(
            r"Action\s*Input:\s*(.+?)(?:\n|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = (
                input_match.group(1).strip() if input_match else ""
            )
            return tool_name, tool_input
        return None, None

    # ── L2: 纯 LLM 回答 ────────────────────────────────

    def _llm_ask_plain(self, question: str, lang: str) -> str:
        if not self._llm.is_available:
            logger.debug("_llm_ask_plain: LLM not available, skip")
            return None
        t0 = time.time()
        logger.debug("_llm_ask_plain: LLM.ask() START")
        result = self._llm.ask(question, lang=lang)
        logger.debug("_llm_ask_plain: LLM.ask() => %s | %.3fs", 'OK' if result else 'None', time.time()-t0)
        return result

    def _llm_ask_with_context(
        self, question: str, context: str, lang: str
    ) -> str:
        """带预取上下文的 LLM 回答（L2+ 快速路径）。"""
        if not self._llm.is_available:
            logger.debug("_llm_ask_with_context: LLM not available, skip")
            return None
        t0 = time.time()
        logger.debug("_llm_ask_with_context: LLM.ask(context) START")
        result = self._llm.ask(question, context=context, lang=lang)
        ok = 'OK' if result else 'None'
        logger.debug("_llm_ask_with_context: LLM.ask(context) => %s | %.3fs", ok, time.time()-t0)
        return result

    # ── 默认回答 ────────────────────────────────────────

    def _build_default_response(self) -> str:
        questions = self.get_quick_questions()
        if questions:
            examples = "\n".join(
                f"  {i}. {q}" for i, q in enumerate(questions, 1)
            )
            return (
                f"{t('agent.default_response')}\n"
                f"{t('agent.try_these')}\n\n"
                f"{examples}\n\n"
                f"{t('agent.input_related')}"
            )
        return t("agent.default_response")


# ── 规则层降级 Agent ────────────────────────────────────


class GestureAgent:
    """规则层降级 Agent（基于 TF-IDF 知识库匹配）。"""

    def __init__(self, knowledge_base_path=None, lang=None):
        self._lang = lang or config.DEFAULT_LANGUAGE
        self._retriever = create_retriever()
        path = knowledge_base_path or config.KNOWLEDGE_BASE_PATHS.get(
            self._lang, config.KNOWLEDGE_BASE_PATH
        )
        self._retriever.load(path, lang=self._lang)

    def match(self, question: str):
        return self._retriever.match(question)

    def get_quick_questions(self):
        return self._retriever.get_quick_questions(count=6)


class _FallbackAgent:
    """初始化失败时的兜底 Agent。"""

    def answer(self, question, lang=None):
        return t("agent.init_failed")

    def get_quick_questions(self):
        return []

    def reload(self, lang):
        pass

    def clear_history(self):
        pass
