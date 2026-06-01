# Agent 重构架构设计文档

> 最后更新：2026-06-01
> 版本：v3.0 — LangChain ChatOpenAI + 手动 ReAct 解析 + Chroma RAG 架构

---

## 一、架构总览

### 1.1 核心理念

```
旧架构 (v1):  规则匹配(主力) + LLM(兜底) + 关键词选工具
旧架构 (v2):  自建 Agent Loop + TF-IDF 检索
新架构 (v3):  LangChain ChatOpenAI + 手动 ReAct 解析 + Chroma 向量 RAG
```

使用 LangChain ChatOpenAI 调用 MiMo API，LLM 输出 ReAct 格式（Thought/Action/Action Input/Final Answer），代码正则解析并手动调度工具，支持多轮工具调用。知识库检索从 TF-IDF 升级为 ChromaDB 向量数据库 + sentence-transformers 嵌入模型，实现真正的语义检索。

### 1.2 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    app.py (UI 薄壳)                      │
│   Gradio 界面 → 调用 agent.invoke()                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│              core/agent.py (LangChainAgent)              │
│                                                         │
│   ┌─────────────────────────────────────────────┐       │
│   │     手动 ReAct 解析 (正则提取)               │       │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │       │
│   │  │search_kb │ │query_hist│ │query_model│    │       │
│   │  └──────────┘ └──────────┘ └──────────┘    │       │
│   │  ┌──────────┐                               │       │
│   │  │query_code│                               │       │
│   │  └──────────┘                               │       │
│   └─────────────────────────────────────────────┘       │
│                         │                               │
│   ┌─────────────────────v───────────────────────┐       │
│   │     多轮工具调用循环 (for turn in max_turns) │       │
│   │   1. LLM 接收问题 + 工具描述                 │       │
│   │   2. 正则解析 Action/Action Input            │       │
│   │   3. 执行工具，结果注入下一轮，循环           │       │
│   │   4. 解析到 Final Answer 则返回              │       │
│   └─────────────────────────────────────────────┘       │
│                                                         │
│   ┌─────────────────────────────────────────────┐       │
│   │         Fallback Chain (四级降级链)           │       │
│   │   L1: LangChain Agent + Tools               │       │
│   │   L2: LLM without Tools (纯 LLM 回答)       │       │
│   │   L3: 规则层 TF-IDF 匹配知识库               │       │
│   │   L4: 默认兜底回答                           │       │
│   └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
                         │
     ┌───────────┬───────┴────────┬───────────────┐
     v           v                v               v
┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│core/llm  │ │core/tools│ │core/rag_     │ │core/retrieval│
│.py       │ │.py       │ │retriever.py  │ │.py          │
│(MiMo API)│ │(LangChain│ │(Chroma RAG)  │ │(TF-IDF降级) │
│          │ │ Tools)   │ │              │ │              │
└──────────┘ └──────────┘ └──────────────┘ └──────────────┘
```

### 1.3 数据流（用户请求完整生命周期）

```
用户: "我昨天识别了几次手势？精度怎么样？"
  │
  v
[1] LangChainAgent.answer(question, lang)
  │
  v
[2] 快速路径判断 → 非简单问题，走 L1
  │
  v
[3] L1 _run_direct_llm_with_tools: 构建 System Prompt
    - 角色定义 + 工具描述 + ReAct 格式约束
  │
  v
[4] 多轮工具调用 第 1 轮:
    LLM → "Thought: 需要查询历史\nAction: query_history\nAction Input: 昨天"
    正则解析 → 执行 query_history("昨天") → "昨天共识别 5 次..."
  │
  v
[5] 多轮工具调用 第 2 轮:
    LLM → "Thought: 需要查精度\nAction: query_model_metrics\nAction Input: 精度"
    正则解析 → 执行 query_model_metrics("精度") → "mAP50: 99.5%..."
  │
  v
[6] 多轮工具调用 第 3 轮:
    LLM → "Thought: 已有足够信息\nFinal Answer: 昨天您共识别了5次手势..."
    正则解析 → 返回最终回答
  │
  v
返回给用户
```

---

## 二、模块拆分与职责

### 2.1 文件清单

| 文件 | 职责 | 变更类型 |
|------|------|----------|
| `app.py` | UI 薄壳，仅 Gradio 布局 + 调用 Agent 接口 | 适配 |
| `core/agent.py` | LangChainAgent（ReAct Agent + 四级降级链） | **重写** |
| `core/tools.py` | LangChain Tool 封装（history/model/code/search_kb） | **重写** |
| `core/llm.py` | MiMo API 的 LangChain 兼容 LLM 封装 | **重写** |
| `core/rag_retriever.py` | Chroma 向量 RAG 检索器 | **新建** |
| `core/retrieval.py` | TF-IDF 检索器（保留作为 L3 降级） | 保留 |
| `core/text_preprocessor.py` | 分词、去停用词、归一化 | 保留 |
| `core/history.py` | SQLite 历史记录管理 | 保留不变 |
| `core/detector.py` | YOLOv8 手势检测 | 保留不变 |
| `core/i18n.py` | 国际化翻译 | 保留不变 |

### 2.2 调用关系

```
app.py
  └─ core/agent.py (LangChainAgent)
       ├─ core/llm.py (LLM)               ← API 通信
       ├─ core/tools.py (ToolRegistry)     ← 工具定义 + 执行
       │    └─ core/retrieval.py           ← 知识库检索
       │         └─ core/text_preprocessor.py
       └─ core/retrieval.py (GestureAgent) ← 规则层降级
```

### 2.3 app.py 变更范围

**移出 app.py 的内容：**
- `_FallbackAgent` 类 → 移入 `core/agent.py`
- Agent 初始化逻辑 → `LangChainAgent` 统一管理
- 快捷问题按钮逻辑 → 通过 `agent.get_quick_questions()` 调用
- 对话历史管理 → `agent.clear_history()` 统一接口

**app.py 保留：**
- Gradio UI 布局定义（5 个 Tab）
- 各 Tab 的事件绑定（图片/视频/摄像头/Agent/历史）
- 语言切换事件 → 调用 `agent.reload(lang)`

**对外接口不变：**
```python
agent.answer(question: str, lang: str = None) -> str
agent.get_quick_questions() -> list[str]
agent.reload(lang: str) -> None
agent.clear_history() -> None
```

---

## 三、Agent 设计（LangChain ReAct Agent）

### 3.1 Agent 架构

使用 LangChain ChatOpenAI 调用 LLM，LLM 输出 ReAct 格式文本（Thought/Action/Action Input/Final Answer），代码通过正则表达式解析并手动调度工具。多轮工具调用通过 `for turn in range(max_turns)` 循环实现，工具结果注入下一轮消息上下文。

### 3.2 工具定义

工具使用 LangChain 的 `Tool` 类封装：

```python
from langchain.tools import Tool

Tool(
    name="query_history",
    func=query_history_fn,
    description="查询手势检测历史记录，支持时间范围查询"
)
```

### 3.3 终止条件

由 config.py 中的安全守卫参数控制：
- `AGENT_MAX_TURNS=3`：最大 LLM 调用轮次
- `AGENT_MAX_TOOL_CALLS=6`：工具总调用上限
- `AGENT_MAX_SAME_CALLS=2`：相同工具+输入最大重复次数
- `AGENT_MAX_PARSE_FAILS=2`：连续解析失败上限，触发后用 followup LLM 追问
- 超限后自动用已收集的工具结果做 followup 生成最终回答

### 3.4 快速路径

在 Agent 入口处增加关键词预判，对无需工具的简单问题直接走 L2（纯 LLM 回答），跳过 LangChain Agent 的工具调用循环：

- **命中条件**: 匹配系统介绍模式（"系统是什么"、"what is this system"等）、打招呼、闲聊
- **排除条件**: 问题含工具相关关键词（"历史"、"识别"、"指标"、"mAP"等）时不走快速路径
- **效果**: 简单问题从 50-60s 降至 5-10s

### 3.5 工具调用策略

LLM 输出 ReAct 格式文本，代码正则解析：
- `_extract_final_answer(text)` → 提取 `Final Answer: ...`
- `_extract_tool_call(text)` → 提取 `Action: tool_name` + `Action Input: input`
- 解析失败时通过 `_ask_followup_clarify()` 用 followup LLM 追问要求正确格式
- 相同工具+输入重复调用超过上限时，提示 LLM 基于已有结果回答

---

## 四、四级降级链

```
Level 1: LangChain ReAct Agent + Tools
   ↓ 失败（AgentExecutor 异常 / 循环超限 / LLM 报错）
Level 2: LLM without Tools (单次纯 LLM 调用)
   ↓ 失败（API 不可用 / 熔断中）
Level 3: 规则层 TF-IDF 匹配知识库
   ↓ 无匹配（score < 0.2）
Level 4: 默认兜底回答
```

### 各级触发条件

| 级别 | 触发条件 | 实现方式 |
|------|----------|----------|
| L1 | 默认路径 | LangChain AgentExecutor.invoke() |
| L2 | L1 异常或循环超限 | `llm.ask(question, lang=lang)` |
| L3 | L2 失败（API 不可用 / 熔断） | `retriever.match(question)` |
| L4 | L3 score < 0.2 | 返回默认兜底文本 |

### 降级提示

- L2 成功 → 无提示
- L3 成功 → 追加 `[注：当前智能服务暂不可用，以上回答来自本地知识库]`
- L4 → 追加 `[注：智能服务暂不可用，无法匹配相关知识]`

降级提示仅在首次降级时追加，避免重复提示。

---

## 五、RAG 检索设计（Chroma 向量检索）

### 5.1 架构

使用 `core/rag_retriever.py` 实现基于 ChromaDB 的向量检索：
- **嵌入模型**: sentence-transformers (all-MiniLM-L6-v2)
- **向量数据库**: ChromaDB（持久化存储）
- **索引源**: Essay.docx、.md 文档、代码函数、knowledge_base.json

### 5.2 索引构建

```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("knowledge")

# 分块策略：按段落分块，每块 500 字符，重叠 50 字符
# 来源：Essay.docx、README.md、CLAUDE.md、AgentMaker.md、knowledge_base.json
```

### 5.3 降级保留

`core/retrieval.py` 中的 TFIDFRetriever 保留作为 L3 降级路径。

---

## 六、LLM 通信层

### 6.1 MiMo API 调用

- 格式：OpenAI 兼容 Chat Completion（`POST {base_url}/chat/completions`）
- 通过 LangChain `ChatOpenAI` 兼容接口调用（base_url 指向 MiMo API）
- temperature=0, max_tokens=512

### 6.2 容错机制

| 机制 | 参数 |
|------|------|
| 超时 | 单次 15s |
| 重试 | 最多 2 次，指数退避 |
| 熔断 | 连续 3 次失败 → 60 秒冷却 |
| 错误脱敏 | 日志中过滤 API Key |
| 对话历史 | deque(maxlen=10)，5 轮 |

### 6.3 Tool Calling 方式

使用 ReAct prompt 模式（Thought/Action/Action Input/Final Answer），不依赖 MiMo API 的原生 tool_calls 支持。System Prompt 中明确定义输出格式约束，代码正则解析 LLM 输出并手动调度工具执行。

---

## 七、测试策略

### 7.1 测试分层

```
tests/
├── test_agent_loop.py      # LangChainAgent 单元测试（Mock LLM）
├── test_tools.py           # 工具函数测试
├── test_retrieval.py       # 检索器测试
├── test_e2e.py             # E2E 链路验证（Mock LLM）
└── conftest.py             # 共享 fixtures
```

### 7.2 Agent Loop 测试场景

| 场景 | Mock 行为 | 验证 |
|------|-----------|------|
| 单工具调用成功 | 第1轮 tool_call, 第2轮 final_answer | 返回含工具结果的回答 |
| 多工具串行 | 2 次 tool_call + 1 次 final_answer | 返回综合回答 |
| LLM 直接回答 | 第1轮即 final_answer | 不调用工具 |
| 解析失败重试 | 第1轮非法 JSON, 第2轮合法 | 重试后成功返回 |
| 最大轮次终止 | 连续 5+ 轮 tool_call | 不 hang，有返回 |
| 相同工具循环 | 连续 2 轮相同 tool+args | 检测循环，强制终止 |
| LLM 完全不可用 | API 总是报错 | 降级到规则层 |
| 规则层无匹配 | LLM 不可用 + 无关问题 | 返回默认兜底 |

### 7.3 工具测试

独立测试每个工具函数：正常输入、空输入、异常输入。

### 7.4 集成测试（可选）

需要真实 API Key，标记 `@pytest.mark.skipif`。

---

## 八、配置变更

### config.py 新增项

```python
# Agent Loop 配置
AGENT_MAX_TURNS = 3          # 从 5 降至 3，避免简单问题超时
AGENT_MAX_TOOL_CALLS = 6
AGENT_MAX_SAME_CALLS = 2
AGENT_MAX_PARSE_FAILS = 2

# LLM 超时配置
AGENT_LLM_TIMEOUT = 15       # 从 30s 降至 15s，单次 LLM 调用超时

# 检索器后端
RETRIEVER_BACKEND = "tfidf"  # "tfidf" | "vector"

# RAG 配置
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
```

### 新增项

```python
# 检测参数
VIDEO_CONF_THRESHOLD = 0.5       # 视频识别置信度阈值
CAMERA_TIMER_INTERVAL = 0.033    # 摄像头帧获取间隔（~30fps）

# 模型训练结果路径
TRAIN_RESULTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "train4", "results.csv")
TRAIN_ARGS_PATH = os.path.join(BASE_DIR, "runs", "detect", "train4", "args.yaml")
```

### 移除项

- `AGENT_LLM_ENABLED`：未使用的功能标志，已删除
- `AGENT_RULE_HIGH_THRESHOLD` / `AGENT_RULE_LOW_THRESHOLD`：未使用的阈值，已删除

---

## 九、依赖清单

**新增依赖：**

| 包 | 用途 |
|----|------|
| `langchain` | Agent 框架（LangChain ChatOpenAI, Tool） |
| `langchain-community` | LangChain 社区集成 |
| `langchain-openai` | LangChain OpenAI 兼容接口（ChatOpenAI） |
| `chromadb` | 向量数据库（存储文档嵌入） |
| `sentence-transformers` | 文本嵌入模型（all-MiniLM-L6-v2） |

**保留依赖：**

| 包 | 用途 |
|----|------|
| `requests` | MiMo API 调用 |
| `scikit-learn` | TF-IDF 向量化（L3 降级） |
| `jieba` | 中文分词 |
| `python-dotenv` | .env 加载 |
| `ultralytics` | YOLOv8 检测 |
| `gradio` | UI |

---

## 十、风险与缓解

| 风险 | 缓解 |
|------|------|
| MiMo 不稳定输出格式 | 正则解析 + followup LLM 追问 + 四级降级 |
| 工具调用死循环 | AGENT_MAX_TURNS=3 + AGENT_MAX_TOOL_CALLS=6 + AGENT_MAX_SAME_CALLS=2 三重保障 |
| LLM 调用延迟高 | 快速路径跳过 Agent + MAX_TURNS=3 + 15s 超时 |
| 简单问题被误判需工具 | 快速路径关键词预判，打招呼/系统介绍直接走 L2 |
| 降级到规则层质量低 | Chroma 向量 RAG 语义检索 + TF-IDF 兜底 |
| 重构引入 Bug | 对外接口不变，逐模块替换，保留 fallback，56 个测试用例 |

---

## 十一、实施步骤

### Phase 1: 文档更新
- [x] 更新 CLAUDE.md（技术栈、依赖批准、目录结构）
- [x] 更新 AgentMaker.md（架构方案、依赖清单）

### Phase 2: 安装依赖
- [x] pip install langchain langchain-community chromadb sentence-transformers
- [x] 更新 requirements.txt

### Phase 3: 新建 core/rag_retriever.py
- [x] 实现 Chroma 向量检索器
- [x] 索引 Essay.docx、.md 文档、knowledge_base.json
- [x] sentence-transformers 嵌入

### Phase 4: 重写 core/llm.py
- [x] 封装 LangChain 兼容的 LLM 类（ChatOpenAI 兼容 MiMo API）

### Phase 5: 重写 core/tools.py
- [x] 将工具封装为 LangChain Tool 对象
- [x] search_knowledge_base 切换为向量检索

### Phase 6: 重写 core/agent.py
- [x] 使用 LangChain create_react_agent
- [x] 集成四级降级链

### Phase 7: 适配 app.py
- [x] Agent Tab 调用新 Agent 的 invoke() 方法
- [x] 确保 5 个 Tab 功能不变

### Phase 8: 验证
- [x] flake8 全量通过
- [x] app.py 启动正常
- [x] Agent 功能验证
