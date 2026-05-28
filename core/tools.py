"""检索工具模块 - 为 Agent 提供项目真实数据查询能力。"""

import csv
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta

import config
from core.i18n import t

logger = logging.getLogger(__name__)


def _format_detection_detail(row, type_names):
    """格式化单条记录的检测详情。

    Args:
        row: 数据库记录行
        type_names: 类型名称映射

    Returns:
        str: 格式化的详情文本
    """
    created = row["created_at"][:16] if row["created_at"] else t("tools.unknown_time")
    gc = row["gesture_count"]
    st = type_names.get(row["source_type"], row["source_type"])
    line = (
        f"  - [{created}] {st}，"
        f"{t('tools.gesture_count_short', count=gc)}"
    )

    detections_raw = row["detections"] if row["detections"] else "[]"
    try:
        detections = json.loads(detections_raw)
    except (json.JSONDecodeError, TypeError):
        detections = []

    if not detections:
        return line

    source_type = row["source_type"]
    if source_type in ("image", "camera"):
        details = []
        for d in detections:
            cls_name = d.get("class", "N/A")
            conf = d.get("confidence", 0)
            details.append(
                f"{cls_name} ({t('tools.confidence_short', value=f'{conf:.2%}')})"
            )
        line += (
            "\n    " + t("tools.detection_detail_label") + ": "
            + "、".join(details)
        )
    elif source_type == "video":
        details = []
        for d in detections:
            cls_name = d.get("class", "N/A")
            fc = d.get("frame_count", 0)
            details.append(f"{cls_name}({t('msg.frame_count', count=fc)})")
        line += (
            "\n    " + t("tools.detection_detail_label") + ": "
            + "、".join(details)
        )

    return line


def query_history(query):
    """查询检测历史记录。

    根据 query 中的时间关键词构建 SQL 条件，返回格式化的统计摘要。
    支持中/英/法/繁中多语言时间关键词匹配。

    Args:
        query: 用户问题，用于提取时间范围等关键词

    Returns:
        str: 格式化的历史记录摘要
    """
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row

        # 解析时间关键词（多语言正则匹配）
        now = datetime.now()
        date_filter = None
        label = ""
        fallback_used = False

        today_patterns = [
            r"今天", r"today", r"aujourd",
            r"今日", r"aujourd'hui",
        ]
        yesterday_patterns = [
            r"昨天", r"yesterday", r"hier", r"昨日",
        ]
        week_patterns = [
            r"本周", r"this\s*week", r"cette\s*semaine",
            r"本週",
        ]
        recent_patterns = [
            r"最近|近", r"recent|last|past",
            r"récents?|derni[eè]re?s?",
            r"最近",
        ]

        query_lower = query.lower()

        if any(re.search(p, query_lower) for p in today_patterns):
            date_filter = now.strftime("%Y-%m-%d")
            label = t("tools.today")
        elif any(re.search(p, query_lower) for p in yesterday_patterns):
            yesterday = now - timedelta(days=1)
            date_filter = yesterday.strftime("%Y-%m-%d")
            label = t("tools.yesterday")
        elif any(re.search(p, query_lower) for p in week_patterns):
            monday = now - timedelta(days=now.weekday())
            date_filter = monday.strftime("%Y-%m-%d")
            label = t("tools.this_week")
        elif any(re.search(p, query_lower) for p in recent_patterns):
            match = re.search(r"(\d+)\s*(天|day|jour)", query_lower)
            days = int(match.group(1)) if match else 7
            since = now - timedelta(days=days)
            date_filter = since.strftime("%Y-%m-%d")
            label = t("tools.recent_days", days=days)

        # 构建查询
        if date_filter and label in (t("tools.today"), t("tools.yesterday")):
            sql = (
                "SELECT * FROM detection_history "
                "WHERE date(created_at) = ? "
                "ORDER BY created_at DESC"
            )
            rows = conn.execute(sql, (date_filter,)).fetchall()
        elif date_filter:
            sql = (
                "SELECT * FROM detection_history "
                "WHERE created_at >= ? "
                "ORDER BY created_at DESC"
            )
            rows = conn.execute(sql, (date_filter,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detection_history ORDER BY created_at DESC"
            ).fetchall()

        if not rows:
            if label:
                # 日期过滤无结果时，回退到最近记录
                logger.debug(
                    "query_history: no records for filter '%s', "
                    "falling back to recent records", label
                )
                fallback_rows = conn.execute(
                    "SELECT * FROM detection_history "
                    "ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
                if fallback_rows:
                    rows = fallback_rows
                    fallback_used = True
                else:
                    logger.debug(
                        "query_history: no records in database at all"
                    )
                    return t("tools.no_records")
            else:
                logger.debug("query_history: no records in database")
                return t("tools.no_records")

        # 统计汇总
        total = len(rows)
        type_counts = {}
        for row in rows:
            st = row["source_type"]
            type_counts[st] = type_counts.get(st, 0) + 1

        type_names = {
            "image": t("tools.type_image"),
            "video": t("tools.type_video"),
            "camera": t("tools.type_camera"),
        }
        parts = [
            t("tools.type_count", type=type_names.get(k, k), count=v)
            for k, v in type_counts.items()
        ]

        detail = "、".join(parts)
        if fallback_used:
            summary = (
                t("tools.no_records_period", period=label) + "\n"
                + t("tools.total_prefix", total=total, detail=detail)
            )
        elif label:
            summary = t("tools.total_with_label", label=label, total=total, detail=detail)
        else:
            summary = t("tools.total_prefix", total=total, detail=detail)

        # 最近几条记录详情（含检测结果解析）
        recent = rows[:5]
        detail_lines = []
        for row in recent:
            detail_lines.append(
                _format_detection_detail(row, type_names)
            )

        return summary + "\n\n" + t("tools.recent_records") + "\n" + "\n".join(detail_lines)

    except Exception as e:
        logger.error("query_history failed: %s", e)
        return t("tools.query_error", error=e)
    finally:
        if conn is not None:
            conn.close()


def query_model_metrics(query):
    """查询模型训练指标和配置。

    读取 results.csv 最后一行和 args.yaml，返回格式化摘要。

    Args:
        query: 用户问题（保留扩展用）

    Returns:
        str: 格式化的模型指标摘要
    """
    try:
        # 读取 results.csv 最后一行
        results_path = os.path.join(
            config.BASE_DIR, "runs", "detect", "train4", "results.csv"
        )
        metrics = {}
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = None
                last_row = None
                for row in reader:
                    if row and row[0].strip() == "epoch":
                        headers = [h.strip() for h in row]
                    elif row and headers:
                        last_row = [v.strip() for v in row]
                if headers and last_row:
                    for h, v in zip(headers, last_row):
                        metrics[h] = v

        # 读取 args.yaml
        args_path = os.path.join(
            config.BASE_DIR, "runs", "detect", "train4", "args.yaml"
        )
        args = {}
        if os.path.exists(args_path):
            with open(args_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        key, _, val = line.partition(":")
                        args[key.strip()] = val.strip()

        # 格式化输出
        parts = []

        # 基本信息
        model_name = args.get("model", "YOLOv8n")
        parts.append(f"**Model**: {model_name}")

        # 训练配置
        epochs = args.get("epochs", "N/A")
        batch = args.get("batch", "N/A")
        imgsz = args.get("imgsz", "N/A")
        optimizer = args.get("optimizer", "N/A")
        parts.append(f"**{t('tools.epochs')}**: {epochs}")
        parts.append(f"**Batch Size**: {batch}")
        parts.append(f"**{t('tools.input_size')}**: {imgsz}x{imgsz}")
        parts.append(f"**{t('tools.optimizer')}**: {optimizer}")

        # 最终指标
        mAP50 = metrics.get("metrics/mAP50(B)")
        mAP50_95 = metrics.get("metrics/mAP50-95(B)")
        precision = metrics.get("metrics/precision(B)")
        recall = metrics.get("metrics/recall(B)")

        if mAP50:
            parts.append(f"**mAP50**: {float(mAP50) * 100:.1f}%")
        if mAP50_95:
            parts.append(f"**mAP50-95**: {float(mAP50_95) * 100:.1f}%")
        if precision:
            parts.append(f"**Precision**: {float(precision) * 100:.1f}%")
        if recall:
            parts.append(f"**Recall**: {float(recall) * 100:.1f}%")

        # loss
        box_loss = metrics.get("train/box_loss")
        cls_loss = metrics.get("train/cls_loss")
        if box_loss and cls_loss:
            parts.append(
                f"**{t('tools.train_loss')}**: box={box_loss}, cls={cls_loss}"
            )

        if parts:
            return "\n".join(parts)
        return t("tools.model_data_not_found")

    except Exception as e:
        logger.error("query_model_metrics failed: %s", e)
        return t("tools.query_model_error", error=e)


def query_code(function_name):
    """在项目核心源码中搜索函数/类定义。

    Args:
        function_name: 函数或类名

    Returns:
        str: 匹配的源码片段和说明
    """
    if not function_name or not function_name.strip():
        return t("tools.provide_func_name")

    target = function_name.strip()
    search_files = [
        os.path.join(config.BASE_DIR, "core", "detector.py"),
        os.path.join(config.BASE_DIR, "core", "history.py"),
        os.path.join(config.BASE_DIR, "core", "agent.py"),
        os.path.join(config.BASE_DIR, "core", "llm_layer.py"),
        os.path.join(config.BASE_DIR, "core", "intent_router.py"),
        os.path.join(config.BASE_DIR, "core", "tools.py"),
        os.path.join(config.BASE_DIR, "app.py"),
        os.path.join(config.BASE_DIR, "config.py"),
    ]

    results = []
    for fpath in search_files:
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue

        fname = os.path.relpath(fpath, config.BASE_DIR)

        # 搜索 def target 或 class target
        pattern = re.compile(
            rf"^\s*(def|class)\s+{re.escape(target)}\b"
        )
        for i, line in enumerate(lines):
            if pattern.search(line):
                # 提取函数/类体
                indent = len(line) - len(line.lstrip())
                block = [line.rstrip()]
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() == "":
                        block.append("")
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= indent and next_line.strip():
                        break
                    block.append(next_line.rstrip())
                block = block[:30]
                code = "\n".join(block)
                results.append(
                    f"**{fname}** (line {i + 1}):\n"
                    f"```python\n{code}\n```"
                )

    if not results:
        return t("tools.func_not_found", name=target)

    return "\n\n".join(results)


def search_knowledge_base(query):
    """在知识库中检索最相关的 QA 对。

    基于关键词匹配，返回最相关的问答对。
    匹配分数低于阈值时返回空字符串，让 LLM 用通用知识回答。

    Args:
        query: 用户问题

    Returns:
        str: 最相关的 QA 对内容，无匹配时返回空字符串
    """
    try:
        if not os.path.exists(config.KNOWLEDGE_BASE_PATH):
            logger.warning("Knowledge base file not found: %s",
                           config.KNOWLEDGE_BASE_PATH)
            return ""

        with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = data.get("qa_pairs", [])
        if not qa_pairs:
            logger.warning("Knowledge base is empty")
            return ""

        import jieba

        query_lower = query.lower()
        best_score = 0
        best_qa = None

        for qa in qa_pairs:
            score = 0
            # 关键词匹配
            keywords = qa.get("keywords", []) + qa.get("synonyms", [])
            for kw in keywords:
                if kw.lower() in query_lower:
                    score += 2
            # 基于 jieba 分词的词级匹配
            question = qa.get("question", "")
            query_words = set(jieba.cut(query))
            question_words = set(jieba.cut(question))
            overlap = query_words & question_words
            score += len(overlap) * 1.5

            if score > best_score:
                best_score = score
                best_qa = qa

        # 匹配分数低于阈值，视为无相关内容
        min_score_threshold = 2.0
        if best_qa and best_score >= min_score_threshold:
            logger.debug("KB match: score=%.2f, question='%s'",
                         best_score, best_qa.get("question", ""))
            return f"**{best_qa.get('question', '')}**\n\n{best_qa['answer']}"

        logger.debug("KB no match above threshold (best=%.2f) for: %s",
                     best_score, query[:50])
        return ""

    except Exception as e:
        logger.error("search_knowledge_base failed: %s", e)
        return ""


def build_tool_registry():
    """根据当前语言构建工具注册表。"""
    return [
        {
            "name": "query_history",
            "func": query_history,
            "keywords": [kw.strip() for kw in t("keywords.history").split(",")],
            "description": "查询检测历史记录和统计数据",
        },
        {
            "name": "query_model_metrics",
            "func": query_model_metrics,
            "keywords": [kw.strip() for kw in t("keywords.model").split(",")],
            "description": "查询模型训练指标和配置",
        },
        {
            "name": "query_code",
            "func": query_code,
            "keywords": [kw.strip() for kw in t("keywords.code").split(",")],
            "description": "查询项目源码中的函数和类",
        },
        {
            "name": "search_knowledge_base",
            "func": search_knowledge_base,
            "keywords": [],
            "description": "搜索知识库",
        },
    ]


TOOL_REGISTRY = build_tool_registry()
