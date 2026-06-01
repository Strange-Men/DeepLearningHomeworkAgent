# CLAUDE.md - 项目总控规则

## 项目目标
基于YOLOv8n构建实时手势识别系统，支持图片/视频/摄像头识别，集成交互式问答Agent。

---

## 技术栈（已锁定）

| 类别 | 技术 | 版本/路径 |
|------|------|----------|
| 运行环境 | Conda | `E:\Conda\envs\yolo` |
| 深度学习 | PyTorch + Ultralytics | 2.x / 8.x |
| 前端框架 | Gradio | >= 4.0 |
| 图像处理 | OpenCV | >= 4.8 |
| 数据库 | SQLite | Python内置 |
| LLM API | MiMo API (通过 LangChain ChatOpenAI 兼容接口调用) | langchain |
| 文本检索 | LangChain + ChromaDB + sentence-transformers (all-MiniLM-L6-v2) | 最新稳定版 |
| 代码规范 | flake8 + black | 最新稳定版 |

---

## 架构原则

### 分层架构
```
┌─────────────────────────────────┐
│       前端展示层 (Gradio UI)     │
├─────────────────────────────────┤
│       业务逻辑层 (core/)        │
├─────────────────────────────────┤
│       基础设施层 (模型/DB/工具)   │
└─────────────────────────────────┘
```

### 设计原则
1. **单一职责**：每个模块只做一件事
2. **依赖倒置**：业务层不依赖具体实现
3. **配置外置**：所有配置集中到 config.py
4. **异常兜底**：所有外部调用必须有异常处理

---

## 代码组织规则

### 目录结构
```
├── app.py              # Gradio UI 薄壳（仅布局+事件绑定）
├── config.py           # 全局配置
├── core/               # 核心业务
│   ├── agent.py        # LangChainAgent (手动 ReAct 解析 + 四级降级链) + GestureAgent (规则层降级)
│   ├── llm.py          # LLM 通信层（MiMo API、重试、熔断）
│   ├── tools.py        # 工具注册表 + 工具函数
│   ├── retrieval.py    # 知识库检索（TF-IDF / 向量检索抽象接口）
│   ├── rag_retriever.py # Chroma 向量 RAG 检索器（LangChain + sentence-transformers）
│   ├── text_preprocessor.py  # 分词、去停用词
│   ├── detector.py     # YOLOv8 手势检测
│   ├── history.py      # SQLite 历史记录
│   └── i18n.py         # 国际化
├── data/               # 数据文件（知识库/数据库/词典）
├── utils/              # 工具函数（图像/视频转换）
├── tests/              # 测试用例
├── assets/             # 静态资源
└── specs/              # 规格文档
```

### Agent 架构（v3: LangChain ChatOpenAI + 手动 ReAct 解析 + Chroma RAG）
- **Agent**: LangChain ChatOpenAI 调用 + 手动 ReAct 解析（正则提取 Thought/Action/Final Answer）+ 多轮工具调用
- **RAG 检索**: ChromaDB 向量数据库 + sentence-transformers (all-MiniLM-L6-v2) 嵌入
- **四级降级链**: LLM+Tools → LLM纯回答 → TF-IDF规则层 → 默认兜底
- **工具集**: LangChain Tool 封装（query_history, query_model_metrics, query_code, search_knowledge_base）
- **快速路径**: 简单问题（打招呼、系统介绍等）在 Agent 入口预判，直接走 L2 纯 LLM 回答
- **安全守卫**: AGENT_MAX_TURNS=3, AGENT_MAX_TOOL_CALLS=6, AGENT_MAX_SAME_CALLS=2, AGENT_MAX_PARSE_FAILS=2
- 详见 `AgentMaker.md`

### 命名规范
- 文件名：小写下划线（`gesture_detector.py`）
- 类名：大驼峰（`GestureDetector`）
- 函数名：小写下划线（`detect_image()`）
- 常量：全大写（`MODEL_PATH`）

---

## 禁区（绝对不允许）

1. ❌ 不提交密钥、token到代码
2. ❌ 不在业务层直接操作数据库
3. ❌ 不跳过异常处理
4. ❌ 不硬编码路径和参数
5. ❌ 不引入未经批准的新依赖
   > 已批准：langchain, langchain-community, chromadb, sentence-transformers

---

## 开发流程铁律

```
需求 → 计划(需批准) → 编码 → Lint → 测试 → 完成
```

1. **新功能必须先出计划**：写明实现方案，经用户批准后才动手
2. **Lint必须通过**：代码写完运行 `flake8`，零error零warning
3. **测试必须通过**：核心功能必须有测试用例
4. **提交前自检**：检查是否有遗留的print/debug代码
