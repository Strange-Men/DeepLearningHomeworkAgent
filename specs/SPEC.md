# SPEC.md - 功能规格书

## 用户故事（Gherkin语法）

### Story 1: 图片识别
```gherkin
Feature: 图片手势识别
  As a 用户
  I want to 上传图片进行手势识别
  So that 我能知道图片中包含哪些手势

  Scenario: 成功识别图片中的手势
    Given 用户打开图片识别页面
    When 用户上传一张包含手势的图片
    Then 系统显示检测结果图片（带边界框和标签）
    And 系统显示检测统计（手势数量、类别、置信度）

  Scenario: 上传无手势的图片
    Given 用户打开图片识别页面
    When 用户上传一张不含手势的图片
    Then 系统提示"未检测到手势"

  Scenario: 上传无效格式文件
    Given 用户打开图片识别页面
    When 用户上传一个.txt文件
    Then 系统提示"请上传图片文件（jpg/png/bmp）"
```

### Story 2: 视频识别
```gherkin
Feature: 视频手势识别
  As a 用户
  I want to 上传视频进行手势识别
  So that 我能分析视频中的手势动作

  Scenario: 成功识别视频中的手势
    Given 用户打开视频识别页面
    When 用户上传一段包含手势的视频
    Then 系统播放带检测框的视频
    And 系统显示实时FPS

  Scenario: 视频播放控制
    Given 视频正在播放
    When 用户点击暂停按钮
    Then 视频暂停在当前帧
    And 检测结果显示在当前帧上
```

### Story 3: 摄像头实时识别
```gherkin
Feature: 摄像头实时手势识别
  As a 用户
  I want to 使用摄像头进行实时手势识别
  So that 我能即时看到手势识别结果

  Scenario: 启动摄像头识别
    Given 用户打开摄像头识别页面
    When 用户点击开始按钮
    Then 系统打开摄像头并显示实时画面
    And 画面上显示检测框和标签
    And 显示实时FPS和延迟

  Scenario: 摄像头不可用
    Given 用户打开摄像头识别页面
    When 系统无法访问摄像头
    Then 系统提示"无法打开摄像头，请检查设备连接"
```

### Story 4: Agent问答
```gherkin
Feature: 智能问答助手
  As a 用户
  I want to 向Agent提问关于手势识别的问题
  So that 我能了解系统功能和使用方法

  Scenario: 询问支持的手势类别
    Given 用户打开Agent问答页面
    When 用户输入"支持哪些手势？"
    Then Agent回复10类手势的完整列表

  Scenario: 询问使用技巧
    Given 用户打开Agent问答页面
    When 用户输入"如何提高识别准确率？"
    Then Agent给出光线、角度、背景等建议

  Scenario: 询问技术原理
    Given 用户打开Agent问答页面
    When 用户输入"YOLO是什么？"
    Then Agent解释YOLO算法的基本原理

  Scenario: 查询历史记录（RAG）
    Given 用户打开Agent问答页面
    When 用户输入"最近有什么检测记录？"
    Then Agent调用RAG工具查询历史记录并返回摘要

  Scenario: 查询模型指标（RAG）
    Given 用户打开Agent问答页面
    When 用户输入"模型的mAP是多少？"
    Then Agent调用RAG工具读取训练指标并返回结果
```

### Story 5: 历史记录
```gherkin
Feature: 识别历史记录
  As a 用户
  I want to 查看历史识别记录
  So that 我能回顾之前的识别结果

  Scenario: 查看历史记录列表
    Given 用户打开历史记录页面
    When 页面加载完成
    Then 显示历史记录列表（时间、手势数量）

  Scenario: 查看记录详情
    Given 历史记录列表中有数据
    When 用户点击某条记录的"查看"按钮
    Then 弹窗显示原图、检测结果图、检测详情

  Scenario: 删除历史记录
    Given 历史记录列表中有数据
    When 用户点击某条记录的"删除"按钮
    Then 该记录从列表中移除
```

### Story 6: 多语言切换
```gherkin
Feature: 多语言支持
  As a 用户
  I want to 切换界面语言
  So that 我能使用母语操作系统

  Scenario: 切换到英文
    Given 用户打开任意页面
    When 用户在语言下拉框选择"English"
    Then 界面文本全部切换为英文
    And Agent知识库切换为英文知识库

  Scenario: 切换到繁体中文
    Given 用户当前使用简体中文
    When 用户在语言下拉框选择"繁體中文"
    Then 界面文本全部切换为繁体中文
    And Agent知识库切换为繁体中文知识库

  Scenario: 语言回退
    Given 用户切换到法文
    When 某个翻译键缺失法文翻译
    Then 系统回退到简体中文翻译
    And 若简体中文也缺失则显示原始键名
```

---

## 技术架构草案

### 数据库Schema（SQLite）

```sql
-- 识别历史记录表
CREATE TABLE detection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,           -- 'image' / 'video' / 'camera'
    source_path TEXT,                    -- 原始文件路径
    result_path TEXT,                    -- 结果文件路径
    detections TEXT NOT NULL,            -- JSON格式检测结果
    gesture_count INTEGER DEFAULT 0,     -- 检测到的手势数量
    fps REAL,                            -- 处理帧率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 检测结果JSON格式
{
  "gestures": [
    {
      "class": "A",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "summary": {
    "total": 2,
    "by_class": {"A": 1, "V": 1}
  }
}
```

### 核心API列表

| 模块 | 函数 | 输入 | 输出 |
|------|------|------|------|
| detector | `detect_image(image)` | numpy数组 | DetectionResult |
| detector | `detect_frame(frame)` | numpy数组 | (annotated_frame, fps) |
| agent | `answer(question, language)` | 字符串, 语言代码 | 字符串 |
| history | `add_record(...)` | 检测结果 | record_id |
| history | `get_records(limit)` | 数量 | List[Record] |
| history | `delete_record(id)` | record_id | bool |
| i18n | `set_language(lang)` | 语言代码 | None |
| i18n | `t(key, **kwargs)` | 翻译键, 参数 | 字符串 |
| tools | `query_history(db_path)` | 数据库路径 | 历史记录摘要 |
| tools | `query_model_metrics()` | 无 | 模型训练指标 |
| tools | `search_knowledge_base(query)` | 查询文本 | 匹配结果 |

### API示例

```python
# 图片检测
from core.detector import GestureDetector
detector = GestureDetector()  # 从 config.py 读取模型路径
result = detector.detect_image(cv2.imread("test.jpg"))
# result = {
#   "image": <标注后的图片>,
#   "gestures": [{"class": "A", "confidence": 0.95, "bbox": [...]}],
#   "count": 1
# }

# Agent问答（双层混合架构）
from core.agent import HybridAgent
agent = HybridAgent()  # 自动加载规则层 + LLM 层 + RAG 工具
answer = agent.answer("支持哪些手势？", language="zh-CN")

# 国际化
from core.i18n import i18n
i18n.set_language("en")
label = i18n.t("tab_image")  # 返回英文翻译

# 历史记录
from core.history import HistoryManager
history = HistoryManager()
record_id = history.add_record("image", "result.jpg", result)
records = history.get_records(limit=10)
```
