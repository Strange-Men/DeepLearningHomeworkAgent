# Agent 双层混合架构优化框架设计

## 一、现状分析

### 当前问题

| 问题 | 根因 | 影响 |
|------|------|------|
| 匹配率低 | 仅靠关键词子串匹配 + 简单计数 | 用户换个说法就匹配不上 |
| 无法处理开放问题 | 纯规则系统，无生成能力 | 知识库外的问题一律返回默认回答 |
| 同义词缺失 | keywords 字段手动维护，覆盖面窄 | "怎样提升精度"匹配不到"如何提高识别准确率" |
| 相似度算法粗糙 | `matched / sqrt(len(keywords))`，忽略词频和语义 | 高频词泛匹配导致误命中 |

### 当前架构

```
用户问题 --> 关键词子串匹配 --> 命中？--> 返回知识库答案
                                    |
                                    +--> 未命中 --> 返回默认回答
```

---

## 二、目标架构

### 整体架构图

```
+--------------------------------------------------------------+
|                         用户输入                              |
+------------------------------+-------------------------------+
                               v
                      +----------------+
                      |   预处理模块     |  去停用词、分词、归一化
                      +-------+--------+
                              v
                      +----------------+
                      |   意图路由器     |  判断走规则层 or LLM层
                      +---+--------+---+
                          |        |
                +---------v--+  +--v----------+
                |  规则层      |  |  LLM 层      |
                |  (优化后)    |  |  (MiMo API)  |
                +-----+------+  +------+-------+
                      |                |
                      v                v
                +-------------------------+
                |     回答质量评估          |  置信度过滤
                +------------+------------+
                             v
                +-------------------------+
                |     降级兜底逻辑          |  LLM失败 -> 回退规则层
                +------------+------------+
                             v
                      +--------------+
                      |   返回回答     |
                      +--------------+
```

### 数据流

```
用户问题
  |
  v
预处理（分词/归一化）
  |
  v
意图路由器
  |- 高置信度命中规则层 -> 直接返回规则层答案
  |- 低置信度或未命中 -> 调用 LLM 层
  |     |- LLM 成功 -> 返回 LLM 答案
  |     +- LLM 失败（超时/限流/异常）-> 返回规则层最佳匹配 + 降级提示
  +- 两者均无结果 -> 返回默认兜底回答
```

---

## 三、意图路由逻辑

路由决策基于三层信号，按优先级依次判断：

### 3.1 路由决策表

| 优先级 | 条件 | 路由目标 | 理由 |
|--------|------|----------|------|
| 1 | 精确匹配（去除标点后完全一致） | 规则层直接返回 | 确定性 100%，无需 LLM |
| 2 | 规则层相似度 >= 0.7（高阈值） | 规则层直接返回 | 高置信度命中，LLM 是浪费 |
| 3 | 规则层相似度 0.3 ~ 0.7（模糊区间） | 规则层 + LLM 并行，取置信度高者 | 不确定区域，双路验证 |
| 4 | 规则层相似度 < 0.3 或关键词全未命中 | LLM 层 | 规则层无法处理，交给 LLM |
| 5 | LLM 层调用失败 | 回退规则层最佳匹配 | 降级兜底 |

### 3.2 路由阈值配置（config.py 新增）

```python
# Agent 路由配置
AGENT_RULE_HIGH_THRESHOLD = 0.7    # 规则层高置信度阈值
AGENT_RULE_LOW_THRESHOLD = 0.3     # 规则层低置信度阈值
AGENT_LLM_ENABLED = True           # 是否启用 LLM 层
AGENT_LLM_TIMEOUT = 10             # LLM 调用超时（秒）
AGENT_LLM_MAX_RETRIES = 2          # LLM 最大重试次数
AGENT_LLM_MAX_HISTORY = 5          # 最多保留对话轮数
```

### 3.3 路由器伪代码

```python
def route(question):
    rule_score, rule_answer = rule_layer.match(question)

    # 精确匹配或高置信度 -> 直接走规则层
    if rule_score >= RULE_HIGH_THRESHOLD:
        return rule_answer

    # 低置信度 -> 走 LLM
    if rule_score < RULE_LOW_THRESHOLD:
        return llm_layer.ask(question)

    # 模糊区间 -> 双路并行，取高分
    llm_answer = llm_layer.ask(question)
    if llm_answer.confidence > rule_score:
        return llm_answer
    return rule_answer
```

---

## 四、规则层优化方案

### 4.1 优化对比

| 维度 | 当前方案 | 优化后方案 |
|------|----------|------------|
| 匹配方式 | 关键词子串匹配 | TF-IDF 文本相似度 + 关键词匹配加权融合 |
| 同义词 | 无 | 构建同义词表，扩展匹配 |
| 预处理 | 仅 `lower()` | 分词 + 去停用词 + 归一化 |
| 相似度算法 | `count / sqrt(len)` | 余弦相似度（TF-IDF 向量） |
| 知识库结构 | keywords + question | keywords + question + synonyms + expanded_keywords |

### 4.2 TF-IDF 相似度方案

**原理**：将知识库所有问题和用户输入转换为 TF-IDF 向量，计算余弦相似度。

**优势**：
- 零依赖外部 API，纯本地计算
- 对词序不敏感，"怎么提高准确率" 和 "准确率怎么提高" 能匹配
- 自动降低常见词（"的"、"是"）的权重

**依赖**：`scikit-learn`（`TfidfVectorizer` + `cosine_similarity`）

### 4.3 同义词扩展

在 `knowledge_base.json` 中为每个 QA 条目增加 `synonyms` 字段：

```json
{
  "category": "使用技巧",
  "keywords": ["准确", "提高", "技巧"],
  "synonyms": ["精度", "准确率", "效果", "优化", "更好", "改善"],
  "question": "如何提高识别准确率？",
  "answer": "..."
}
```

匹配时将 keywords + synonyms 合并为 expanded_keywords 进行计算。

### 4.4 预处理流水线

```
原始问题 -> 去标点 -> 分词(jieba) -> 去停用词 -> 归一化 -> 清洗后文本
```

新增依赖：`jieba`（中文分词）

### 4.5 匹配算法（融合评分）

```python
final_score = alpha * tfidf_score + beta * keyword_score + gamma * exact_match_bonus
# alpha=0.6, beta=0.3, gamma=0.1 （可调参数）
```

- `tfidf_score`：TF-IDF 余弦相似度（0~1）
- `keyword_score`：当前关键词匹配分数（0~1，归一化后）
- `exact_match_bonus`：精确匹配加分（0 或 1）

---

## 五、LLM 层设计

### 5.1 MiMo API 调用方式

使用 OpenAI 兼容的 Chat Completion 格式（`requests` 直接调用）：

```
POST {MIMO_BASE_URL}/chat/completions
Headers:
  Authorization: Bearer {MIMO_API_KEY}
  Content-Type: application/json

Body:
{
  "model": "mimo-2.5-pro",
  "messages": [
    {"role": "system", "content": "{system_prompt}"},
    {"role": "user", "content": "{user_question}"}
  ],
  "temperature": 0.3,
  "max_tokens": 512,
  "stream": false
}
```

**选择 `requests` 而非 `openai` SDK**：减少一个外部依赖，调用逻辑简单。

### 5.2 System Prompt 设计

```
你是一个专业的手势识别系统问答助手。你的职责是：
1. 回答用户关于本手势识别系统的问题（功能、使用方法、技术原理）
2. 回答与手势识别、YOLO目标检测、CNN卷积神经网络相关的技术问题
3. 保持回答简洁、准确、有条理

约束：
- 不要编造系统不具备的功能
- 不要回答与系统无关的开放性问题（如写代码、翻译、闲聊）
- 如果问题超出你的知识范围，请坦诚说明并建议用户查看官方文档
- 回答使用中文，适当使用 Markdown 格式
```

### 5.3 上下文管理

**方案**：保留最近 N 轮对话历史（默认 5 轮），作为 messages 数组传入。

```
messages = [system_prompt] + 最近N轮对话历史 + [当前问题]
```

**对话历史存储**：内存中维护一个 `collections.deque(maxlen=N*2)`（每轮 = user + assistant 各一条）。

**何时清空**：
- 用户主动点击"清空对话"
- 会话超时（30 分钟无交互）

### 5.4 超时与重试策略

| 参数 | 值 | 说明 |
|------|-----|------|
| 单次超时 | 10 秒 | 超时后触发重试 |
| 最大重试 | 2 次 | 指数退避（1s -> 2s） |
| 总超时 | 25 秒 | 超过后触发降级 |
| 限流（429） | 等待 Retry-After 头指定时间，最长 5 秒 | |
| 服务端错误（5xx） | 立即重试一次 | |

### 5.5 错误处理

```python
try:
    response = call_mimo_api(messages)
    return post_process(response)
except TimeoutError:
    return fallback_to_rule_layer()
except RateLimitError:
    wait_and_retry()
except APIError as e:
    log_error(e)  # 不记录 API Key
    return fallback_to_rule_layer()
```

---

## 六、降级与容灾

### 6.1 降级策略

```
LLM 层调用
  |
  |- 成功 -> 返回 LLM 回答
  |
  |- 超时 / 网络错误
  |     -> 重试（最多 2 次）
  |     -> 仍失败 -> 回退规则层最佳匹配
  |     -> 规则层也无匹配 -> 返回默认回答 + 降级提示
  |
  |- API Key 无效 / 配额耗尽（401/429）
  |     -> 禁用 LLM 层（本 session 内）
  |     -> 后续请求全部走规则层
  |     -> 日志记录警告
  |
  +- 服务端错误（5xx）
        -> 重试一次
        -> 仍失败 -> 回退规则层
```

### 6.2 降级提示

LLM 不可用时，在回答末尾附加提示（不暴露技术细节）：

> [注：当前智能问答服务暂不可用，以上回答来自本地知识库，可能不够完整。]

### 6.3 熔断机制

连续 3 次 LLM 调用失败 -> 触发熔断，后续 60 秒内直接走规则层，不再尝试 LLM。60 秒后恢复尝试。

---

## 七、API Key 安全管理方案

### 7.1 文件结构

```
项目根目录/
+-- .env              # 实际密钥（.gitignore 排除，不提交）
+-- .env.example      # 模板文件（提交到 Git，无真实密钥）
+-- .gitignore        # 包含 .env 排除规则
+-- config.py         # 通过 dotenv 读取环境变量
```

### 7.2 .env.example（提交到 Git）

```
# MiMo API 配置
MIMO_API_KEY=your_api_key_here
MIMO_BASE_URL=https://api.mimo.com/v1
MIMO_MODEL_NAME=mimo-2.5-pro
```

### 7.3 .env（不提交，本地使用）

```
# MiMo API 配置
MIMO_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
MIMO_BASE_URL=https://api.mimo.com/v1
MIMO_MODEL_NAME=mimo-2.5-pro
```

### 7.4 .gitignore 新增规则

```
# 环境变量（密钥）
.env
.env.local
.env.*.local
```

### 7.5 安全红线

| 规则 | 措施 |
|------|------|
| 代码中不硬编码 Key | 通过 `os.getenv()` 读取，无默认真实值 |
| 日志中不打印 Key | 所有日志/异常信息中过滤 Key 内容 |
| 前端不暴露 Key | API 调用在后端完成，前端只接收回答 |
| Git 不提交 .env | `.gitignore` 排除 + CI 检查 |
| 异常信息脱敏 | `except` 捕获时，`str(e)` 中替换 Key 为 `***` |

---

## 八、文件与目录变更规划

### 8.1 新增文件

| 文件路径 | 用途 |
|----------|------|
| `.env` | 存储 API 密钥（本地，不提交） |
| `.env.example` | 密钥模板（提交到 Git） |
| `core/llm_layer.py` | LLM 层封装（MiMo API 调用、重试、上下文管理） |
| `core/intent_router.py` | 意图路由器（决定走规则层还是 LLM 层） |
| `core/text_preprocessor.py` | 文本预处理（分词、去停用词、归一化） |
| `data/synonyms.json` | 同义词表（可选，也可内嵌到知识库） |
| `data/stopwords.txt` | 中文停用词表 |
| `data/custom_dict.txt` | jieba 自定义词典 |

### 8.2 修改文件

| 文件路径 | 变更内容 |
|----------|----------|
| `config.py` | 新增 LLM 相关配置项 + dotenv 加载逻辑 |
| `core/agent.py` | 重构为双层架构入口，集成 Router |
| `app.py` | Agent 初始化适配新架构，传递对话历史 |
| `requirements.txt` | 新增 `python-dotenv`、`jieba`、`scikit-learn` |
| `.gitignore` | 新增 `.env` 排除规则 |
| `data/knowledge_base.json` | 每个 QA 条目增加 `synonyms` 字段 |

---

## 九、实施步骤建议

### Phase 1：安全基础设施 + 配置准备（优先级最高）

1. 创建 `.env.example` 和 `.gitignore`
2. 修改 `config.py`，增加 dotenv 加载逻辑
3. 安装新依赖：`pip install python-dotenv jieba scikit-learn`
4. 更新 `requirements.txt`
5. 验证：启动项目确认配置加载正常，无 Key 时给出 WARNING

### Phase 2：规则层优化

1. 新建 `core/text_preprocessor.py`（分词 + 去停用词 + 归一化）
2. 新建 `data/stopwords.txt`
3. 新建 `data/custom_dict.txt`
4. 修改 `core/agent.py`，替换相似度算法为 TF-IDF + 关键词融合
5. 更新 `data/knowledge_base.json`，为每个条目增加 `synonyms` 字段
6. 测试：用 20 个不同表述的问题验证匹配率提升

### Phase 3：LLM 层集成

1. 新建 `core/llm_layer.py`（封装 MiMo API 调用）
2. 实现上下文管理（deque 维护对话历史）
3. 实现超时重试 + 错误脱敏
4. 测试：单独调用 LLM API 验证连通性

### Phase 4：意图路由 + 降级机制

1. 新建 `core/intent_router.py`（路由决策逻辑）
2. 修改 `core/agent.py`，接入 Router
3. 实现降级兜底 + 熔断机制
4. 更新 `app.py`，适配新 Agent 接口（传递对话历史）
5. 端到端测试：覆盖所有路由分支

### Phase 5：收尾

1. `flake8` 检查，确保零 error 零 warning
2. 补充测试用例
3. 更新 `README.md`（说明 `.env` 配置方式）
4. 验证 `.gitignore` 确实排除了 `.env`

---

## 十、新增依赖清单

| 包名 | 版本 | 用途 |
|------|------|------|
| `python-dotenv` | >= 1.0 | 加载 `.env` 文件 |
| `jieba` | >= 0.42 | 中文分词 |
| `scikit-learn` | >= 1.3 | TF-IDF 向量化 + 余弦相似度 |

---

## 十一、风险与注意事项

| 风险 | 应对 |
|------|------|
| MiMo API 不支持 OpenAI 格式 | 预留 `core/llm_layer.py` 中的接口抽象，切换只需改调用方式 |
| TF-IDF 在小数据集上效果有限 | 知识库仅 10 条，TF-IDF 够用；后续可升级为 sentence-transformers |
| jieba 分词不准 | 维护一份手势识别领域的自定义词典（`data/custom_dict.txt`） |
| API Key 泄露 | 多层防御：.gitignore + dotenv + 日志脱敏 + 前端隔离 |

---

## 实现记录

### Phase 1 实现记录 - 安全基础设施 + 配置准备
- 状态：已完成
- 新增文件：`.env.example`、`.gitignore`
- 修改文件：`config.py`（增加 dotenv 加载 + LLM/路由配置项）、`requirements.txt`（新增 python-dotenv/jieba/scikit-learn/requests）
- 验证结果：config.py 加载正常，无 Key 时输出 WARNING 英文提示（避免 Windows 编码问题）

### Phase 2 实现记录 - 规则层优化
- 状态：已完成
- 新增文件：`core/text_preprocessor.py`（jieba 分词 + 去停用词 + 归一化）、`data/stopwords.txt`、`data/custom_dict.txt`
- 修改文件：`core/agent.py`（TF-IDF + 关键词融合评分）、`data/knowledge_base.json`（每个 QA 增加 synonyms 字段）
- 融合权重：alpha=0.6(TF-IDF) + beta=0.3(关键词) + gamma=0.1(精确匹配)
- 测试结果：精确匹配 1.00，同义表达 0.40~0.81，无关问题 0.00
- 阈值调整：AGENT_SIMILARITY_THRESHOLD 从 0.3 降到 0.2，AGENT_RULE_LOW_THRESHOLD 设为 0.2

### Phase 3 实现记录 - LLM 层集成
- 状态：已完成
- 新增文件：`core/llm_layer.py`（MiMo API 封装）、`test_mimo_api.py`（API 连通性测试脚本）
- 功能：Chat Completion 调用、对话历史管理（deque）、指数退避重试、熔断机制（3次失败→60秒熔断）、错误信息脱敏
- System Prompt：英文撰写，约束只回答手势识别/YOLO/CNN 相关问题
- 无 Key 时 ask() 优雅返回 None，不报错

### Phase 4 实现记录 - 意图路由 + 降级机制
- 状态：已完成
- 新增文件：`core/intent_router.py`（路由决策逻辑）
- 修改文件：`core/agent.py`（新增 HybridAgent 类）、`app.py`（改用 HybridAgent）
- 路由逻辑：精确匹配→规则层直接返回 | 高置信度(>=0.7)→规则层 | 低置信度(<0.2)→LLM | 模糊区间→双路取高分
- 降级：LLM 失败→规则层最佳匹配+降级提示 | 熔断：连续3次失败→60秒内跳过LLM
- app.py 新增 _FallbackAgent 内部类处理初始化失败场景

### Phase 5 实现记录 - 收尾验证
- 状态：已完成
- flake8：全量通过（新增 `.flake8` 配置忽略 app.py 的 E402）
- 端到端测试：9 项测试全部通过（模块导入、配置加载、预处理、规则层匹配、LLM 无 Key 降级、路由器精确匹配/降级、HybridAgent 集成、快捷问题、清空历史）
- 更新文件：`README.md`（新增 .env 配置说明、更新项目结构和开发进度）
- .gitignore 验证：`.env` 被正确排除
