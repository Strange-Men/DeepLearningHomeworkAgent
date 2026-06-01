# YOLOv8 手势识别系统

基于 YOLOv8n 构建的实时手势识别系统，支持 10 类手势（ASL 字母 A/D/I/L/V/W/Y、数字 5/7、I love you）的检测与识别。系统提供图片识别、视频识别、摄像头实时识别三种模式，并集成了基于知识库的智能问答 Agent 和历史记录管理功能。

---

## 已完成功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 图片识别 | 已完成 | 上传图片，返回标注结果、手势类别与置信度 |
| 视频识别 | 已完成 | 逐帧检测，输出带标注框的视频（自动转码 H.264 MP4） |
| 摄像头实时识别 | 已完成 | 实时捕获摄像头画面，显示检测框和 FPS |
| Agent 问答 | 已完成 | LangChain ReAct Agent 架构，LLM 自主决策调用工具，四级降级链，集成 4 个 RAG 工具 |
| 历史记录 | 已完成 | SQLite 存储，支持查看列表、查看详情、删除单条、清空全部 |
| 多语言支持 | 已完成 | 国际化（i18n）支持简体中文、繁體中文、English、Français，UI 和知识库按语言切换 |
| RAG 工具检索 | 已完成 | Agent 集成 4 个 LangChain 工具：查询历史记录、查询模型指标、搜索源码、搜索知识库（Chroma 向量检索） |

---

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 运行环境 | Conda | `E:\Conda\envs\yolo` |
| 深度学习 | PyTorch + Ultralytics YOLOv8n | 2.x / 8.x |
| 前端界面 | Gradio | >= 4.0 |
| 图像处理 | OpenCV | >= 4.8 |
| 数据库 | SQLite | Python 内置 |
| Agent 框架 | LangChain ChatOpenAI + 手动 ReAct 解析 | >= 0.3.0 |
| LLM 接口 | LangChain ChatOpenAI (兼容 MiMo API) | >= 0.2.0 |
| 向量数据库 | ChromaDB | >= 0.5.0 |
| 文本嵌入 | sentence-transformers (all-MiniLM-L6-v2) | >= 2.7.0 |
| 中文分词 | jieba | >= 0.42 |
| 文本向量化 | scikit-learn (TF-IDF, L3 降级备用) | >= 1.3 |
| 环境变量 | python-dotenv | >= 1.0 |
| HTTP 请求 | requests | >= 2.31 |
| 代码规范 | flake8 + black | 最新稳定版 |

---

## 项目结构

```
├── app.py                   # 主入口（Gradio 界面，5 个 Tab 页）
├── config.py                # 全局配置（模型路径、阈值、端口等）
├── core/                    # 核心业务逻辑
│   ├── detector.py          #   YOLO 检测引擎（detect_image / detect_frame）
│   ├── agent.py             #   LangChain ReAct Agent + 四级降级链
│   ├── llm.py               #   LLM 通信层（MiMo API via LangChain ChatOpenAI、重试、熔断）
│   ├── rag_retriever.py     #   Chroma 向量 RAG 检索器（sentence-transformers 嵌入）
│   ├── retrieval.py         #   TF-IDF 检索器（L3 降级备用）
│   ├── text_preprocessor.py #   文本预处理（jieba 分词 + 去停用词）
│   ├── i18n.py              #   国际化模块（多语言 UI 翻译）
│   ├── tools.py             #   LangChain Tool 封装（历史/模型指标/源码/知识库查询）
│   └── history.py           #   历史记录管理（SQLite CRUD）
├── data/                    # 数据文件
│   ├── knowledge_base.json       #  中文问答知识库
│   ├── knowledge_base_en.json    #  英文问答知识库
│   ├── knowledge_base_fr.json    #  法文问答知识库
│   ├── knowledge_base_zh-TW.json #  繁体中文问答知识库
│   ├── chroma_db/                #  Chroma 向量数据库（自动创建）
│   ├── hf_models/                #  HuggingFace 模型缓存
│   ├── stopwords.txt             #  中文停用词表
│   ├── custom_dict.txt           #  jieba 自定义词典（手势识别领域）
│   ├── history.db                #  SQLite 数据库（自动创建）
│   └── history_videos/           #  视频识别结果存储
├── locales/                 # UI 翻译文件
│   ├── zh-CN.json           #   简体中文
│   ├── zh-TW.json           #   繁體中文
│   ├── en.json              #   English
│   └── fr.json              #   Français
├── utils/                   # 工具函数
│   ├── image_utils.py       #   PIL ↔ OpenCV 格式转换
│   └── video_utils.py       #   视频信息获取
├── results/                 # 图片识别结果存储
├── runs/                    # 模型训练产物
│   └── detect/train4/weights/best.pt  # 训练好的 YOLOv8n 权重
├── libs/                    # 第三方动态库（OpenH264）
├── specs/                   # 规格文档
│   └── SPEC.md              #   功能规格书（Gherkin 用户故事）
├── tests/                   # 测试用例
│   ├── test_agent_loop.py   #   LangChainAgent 单元测试（Mock LLM）
│   ├── test_tools.py        #   工具函数测试
│   ├── test_retrieval.py    #   检索器测试
│   ├── test_e2e.py          #   E2E 链路验证（Mock LLM）
│   └── conftest.py          #   共享 Mock 组件
├── Essay.docx               # RAG 索引源文档（课程论文）
├── requirements.txt         # Python 依赖
├── .env.example             # API 密钥模板（提交到 Git）
├── .env                     # 实际密钥（不提交，本地使用）
├── .gitignore               # Git 排除规则
├── .flake8                  # flake8 配置
├── test_mimo_api.py         # MiMo API 连通性测试脚本
├── CLAUDE.md                # 项目总控规则
└── README.md                # 本文件
```

---

## 快速开始

```bash
# 1. 激活 conda 环境
conda activate E:\Conda\envs\yolo

# 2. 安装依赖（首次）
pip install -r requirements.txt

# 3. 配置 API 密钥（可选，启用 LLM 问答层）
cp .env.example .env
# 编辑 .env，填入真实的 MIMO_API_KEY

# 4. 测试 API 连通性（可选）
python test_mimo_api.py

# 5. 运行测试
pytest tests/ -v

# 6. 启动应用
python app.py
```

启动后访问 `http://127.0.0.1:7860` 即可使用。

---

## 支持的手势类别

| 序号 | 手势 | 类别名 |
|------|------|--------|
| 1 | 字母 A | A |
| 2 | 字母 D | D |
| 3 | 字母 I | I |
| 4 | 字母 L | L |
| 5 | 字母 V | V |
| 6 | 字母 W | W |
| 7 | 字母 Y | Y |
| 8 | 数字 5 | number 5 |
| 9 | 数字 7 | number 7 |
| 10 | 我爱你 | I love you |

---

## 已知问题与待优化项

- **相似手势混淆**：部分手势在视觉上相似（如 I 和 L、V 和 W），在低置信度或遮挡场景下可能出现误判
- **视频编码兼容性**：视频识别结果默认输出 H.264 MP4，但部分浏览器可能不支持 AVI 回退格式，需确保系统已安装 ffmpeg
- **摄像头独占问题**：摄像头被其他程序占用时自动重试 3 次（间隔 0.5s），仍失败则提示错误
- **历史记录详情展示**：摄像头模式下的识别结果未保存图片/视频到历史记录，详情页无预览

---

## 开发进度

**v3.0 已完成**，升级为 LangChain ChatOpenAI + 手动 ReAct 解析 + Chroma RAG 架构：

- 图片/视频/摄像头三种识别模式均已跑通
- Agent 问答采用手动 ReAct 解析架构：LLM 输出 Thought/Action/Action Input 格式，代码正则解析并执行工具，支持多轮工具调用
- 四级降级链：L1（LLM + 手动工具调度）→ L2（纯 LLM）→ L3（TF-IDF 规则层）→ L4（默认兜底）
- Agent 集成 4 个 LangChain 工具：查询历史记录、模型训练指标、源码搜索、知识库搜索
- 多轮工具调用安全守卫：最大轮次、工具调用上限、重复调用检测、解析失败追问
- RAG 检索升级为 ChromaDB 向量数据库 + sentence-transformers (all-MiniLM-L6-v2) 嵌入，实现语义检索
- TF-IDF 检索器保留作为 L3 降级路径
- LLM 通信层：通过 LangChain ChatOpenAI 兼容接口调用 MiMo API，指数退避重试、熔断器（3 次失败 → 60s 冷却）
- 国际化（i18n）支持 4 种语言：简体中文、繁體中文、English、Français
- 每种语言配有独立知识库和 UI 翻译文件，支持运行时切换
- API 密钥通过 `.env` 文件管理，代码中无硬编码
- 历史记录完整实现 CRUD 操作（SQLite 连接安全关闭）
- 视频输出自动转码为浏览器可播放格式
- 摄像头自动重试（启动 3 次 + 读帧 3 次）+ 进程退出自动释放
- 测试覆盖：Agent Loop、检索器、工具函数、E2E 链路测试（56 个用例全部通过）
