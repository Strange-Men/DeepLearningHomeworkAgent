"""工具注册表模块 - 为 LangChain Agent 提供可调用的工具集。"""

import csv
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta

import config
from core.i18n import t

logger = logging.getLogger(__name__)


# ── 工具函数 ────────────────────────────────────────────


def _format_detection_detail(row, type_names):
    """格式化单条记录的检测详情。"""
    created = (
        row["created_at"][:16]
        if row["created_at"]
        else t("tools.unknown_time")
    )
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
                f"{cls_name} "
                f"({t('tools.confidence_short', value=f'{conf:.2%}')})"
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
            details.append(
                f"{cls_name}({t('msg.frame_count', count=fc)})"
            )
        line += (
            "\n    " + t("tools.detection_detail_label") + ": "
            + "、".join(details)
        )

    return line


def query_history(query: str) -> str:
    """查询检测历史记录。"""
    t0 = time.time()
    logger.debug("tool query_history: START | query=%r", query)
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row

        now = datetime.now()
        date_filter = None
        label = ""
        fallback_used = False

        today_patterns = [
            r"今天", r"today", r"aujourd", r"今日", r"aujourd'hui",
        ]
        yesterday_patterns = [
            r"昨天", r"yesterday", r"hier", r"昨日",
        ]
        week_patterns = [
            r"本周", r"this\s*week", r"cette\s*semaine", r"本週",
        ]
        recent_patterns = [
            r"最近|近", r"recent|last|past",
            r"r\xc3\xa9cents?|derni[\xc3\xa8\xc3\xa9]e?s?",
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
            match = re.search(r"(\d+)\s*(\xe5\xa4\xa9|day|jour)", query_lower)
            days = int(match.group(1)) if match else 7
            since = now - timedelta(days=days)
            date_filter = since.strftime("%Y-%m-%d")
            label = t("tools.recent_days", days=days)

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
                "SELECT * FROM detection_history "
                "ORDER BY created_at DESC"
            ).fetchall()

        if not rows:
            if label:
                fallback_rows = conn.execute(
                    "SELECT * FROM detection_history "
                    "ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
                if fallback_rows:
                    rows = fallback_rows
                    fallback_used = True
                else:
                    return t("tools.no_records")
            else:
                return t("tools.no_records")

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
            summary = t(
                "tools.total_with_label", label=label,
                total=total, detail=detail,
            )
        else:
            summary = t(
                "tools.total_prefix", total=total, detail=detail
            )

        recent = rows[:5]
        detail_lines = [_format_detection_detail(row, type_names)
                        for row in recent]

        logger.debug("tool query_history: OK after %.3fs | rows=%d", time.time()-t0, len(rows))
        return (
            summary + "\n\n"
            + t("tools.recent_records") + "\n"
            + "\n".join(detail_lines)
        )

    except Exception as e:
        logger.error("query_history failed: %s", e)
        logger.debug("tool query_history: ERROR after %.3fs | %s", time.time()-t0, e)
        return t("tools.query_error", error=e)
    finally:
        if conn is not None:
            conn.close()


def query_model_metrics(query: str) -> str:
    """查询模型训练指标和配置。"""
    t0 = time.time()
    logger.debug("tool query_model_metrics: START | query=%r", query)
    try:
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

        parts = []
        model_name = args.get("model", "YOLOv8n")
        parts.append(f"Model: {model_name}")
        parts.append(f"Epochs: {args.get('epochs', 'N/A')}")
        parts.append(f"Batch Size: {args.get('batch', 'N/A')}")
        parts.append(
            f"Input Size: "
            f"{args.get('imgsz', 'N/A')}x{args.get('imgsz', 'N/A')}"
        )
        parts.append(
            f"Optimizer: {args.get('optimizer', 'N/A')}"
        )

        mAP50 = metrics.get("metrics/mAP50(B)")
        mAP50_95 = metrics.get("metrics/mAP50-95(B)")
        precision = metrics.get("metrics/precision(B)")
        recall = metrics.get("metrics/recall(B)")

        if mAP50:
            parts.append(f"mAP50: {float(mAP50) * 100:.1f}%")
        if mAP50_95:
            parts.append(f"mAP50-95: {float(mAP50_95) * 100:.1f}%")
        if precision:
            parts.append(f"Precision: {float(precision) * 100:.1f}%")
        if recall:
            parts.append(f"Recall: {float(recall) * 100:.1f}%")

        box_loss = metrics.get("train/box_loss")
        cls_loss = metrics.get("train/cls_loss")
        if box_loss and cls_loss:
            parts.append(
                f"Train Loss: box={box_loss}, cls={cls_loss}"
            )

        logger.debug("tool query_model_metrics: OK after %.3fs", time.time()-t0)
        return "\n".join(parts) if parts else t("tools.model_data_not_found")

    except Exception as e:
        logger.error("query_model_metrics failed: %s", e)
        logger.debug("tool query_model_metrics: ERROR after %.3fs | %s", time.time()-t0, e)
        return t("tools.query_model_error", error=e)


def query_code(function_name: str) -> str:
    """在项目核心源码中搜索函数/类定义。"""
    t0 = time.time()
    logger.debug("tool query_code: START | function_name=%r", function_name)
    if not function_name or not function_name.strip():
        logger.debug("tool query_code: empty input, %.3fs", time.time()-t0)
        return t("tools.provide_func_name")

    target = function_name.strip()
    search_files = [
        os.path.join(config.BASE_DIR, "core", "detector.py"),
        os.path.join(config.BASE_DIR, "core", "history.py"),
        os.path.join(config.BASE_DIR, "core", "agent.py"),
        os.path.join(config.BASE_DIR, "core", "llm.py"),
        os.path.join(config.BASE_DIR, "core", "tools.py"),
        os.path.join(config.BASE_DIR, "core", "retrieval.py"),
        os.path.join(config.BASE_DIR, "core", "rag_retriever.py"),
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
        pattern = re.compile(
            rf"^\s*(def|class)\s+{re.escape(target)}\b"
        )
        for i, line in enumerate(lines):
            if pattern.search(line):
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
        logger.debug("tool query_code: not found after %.3fs", time.time()-t0)
        return t("tools.func_not_found", name=target)

    logger.debug("tool query_code: OK after %.3fs | results=%d", time.time()-t0, len(results))
    return "\n\n".join(results)


def search_knowledge_base(query: str, retriever=None) -> str:
    """通过向量检索器搜索知识库。"""
    t0 = time.time()
    logger.debug("tool search_knowledge_base: START | query=%r", query)
    if retriever is None:
        logger.debug("tool search_knowledge_base: no retriever, %.3fs", time.time()-t0)
        return ""
    results = retriever.search(query, top_k=3)
    logger.debug("tool search_knowledge_base: search done | %.3fs | results=%d", time.time()-t0, len(results))
    if not results:
        return ""
    # 返回最相关的结果
    parts = []
    for r in results:
        if r["score"] >= 0.3:
            parts.append(r["text"])
    if parts:
        logger.debug("tool search_knowledge_base: OK after %.3fs | matched=%d", time.time()-t0, len(parts))
        return "\n\n---\n\n".join(parts[:2])
    logger.debug("tool search_knowledge_base: no match after %.3fs", time.time()-t0)
    return ""


# ── LangChain Tool 构建 ────────────────────────────────


def build_langchain_tools(retriever=None):
    """构建 LangChain Tool 列表。

    Args:
        retriever: ChromaRetriever 实例

    Returns:
        list[Tool]: LangChain Tool 对象列表
    """
    from langchain_core.tools import Tool

    def _search_kb(query: str) -> str:
        return search_knowledge_base(query, retriever=retriever)

    tools = [
        Tool(
            name="query_history",
            func=query_history,
            description=(
                "查询手势检测历史记录。支持时间范围：今天、昨天、"
                "本周、最近N天。返回检测次数统计和详情。"
                "输入应为包含时间关键词的查询字符串。"
            ),
        ),
        Tool(
            name="query_model_metrics",
            func=query_model_metrics,
            description=(
                "查询模型训练指标和配置信息。返回 mAP50、mAP50-95、"
                "Precision、Recall、Loss 等指标。"
                "输入应为查询内容，如'模型精度'、'训练指标'。"
            ),
        ),
        Tool(
            name="query_code",
            func=query_code,
            description=(
                "在项目源码中搜索函数或类的定义，返回源码片段。"
                "输入应为函数名或类名，如'detect_image'、'GestureDetector'。"
            ),
        ),
        Tool(
            name="search_knowledge_base",
            func=_search_kb,
            description=(
                "在手势识别知识库中搜索相关信息。涵盖手势教程、"
                "YOLO原理、使用技巧、故障排除等内容。"
                "输入应为搜索关键词或问题。"
            ),
        ),
    ]

    return tools


# 旧版兼容：TOOL_SCHEMAS（供其他模块引用）
TOOL_SCHEMAS = [
    {
        "name": "query_history",
        "description": (
            "查询手势检测历史记录。支持时间范围：今天、昨天、"
            "本周、最近N天。返回检测次数统计和详情。"
        ),
        "parameters": {
            "query": "查询条件，包含时间关键词"
        },
    },
    {
        "name": "query_model_metrics",
        "description": (
            "查询模型训练指标和配置信息。返回 mAP50、mAP50-95、"
            "Precision、Recall、Loss 等指标。"
        ),
        "parameters": {
            "query": "查询内容，如'模型精度'、'训练指标'"
        },
    },
    {
        "name": "query_code",
        "description": "在项目源码中搜索函数或类的定义，返回源码片段。",
        "parameters": {
            "function_name": "要搜索的函数名或类名"
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "在手势识别知识库中搜索相关信息。涵盖手势教程、"
            "YOLO原理、使用技巧、故障排除等内容。"
        ),
        "parameters": {
            "query": "搜索关键词或问题"
        },
    },
]


def build_tool_registry(retriever=None):
    """构建旧版工具注册表（兼容接口）。"""
    def _search_kb(query: str) -> str:
        if retriever is None:
            return ""
        results = retriever.search(query, top_k=1)
        if not results:
            return ""
        r = results[0]
        # 兼容 ChromaRetriever (dict) 和 TFIDFRetriever (SearchResult)
        if isinstance(r, dict):
            if r.get("score", 0) >= 0.2:
                return r.get("text", "")
        else:
            if r.score >= 0.2:
                return f"**{r.question}**\n\n{r.answer}"
        return ""

    registry = [
        {"name": "query_history", "func": query_history,
         "schema": TOOL_SCHEMAS[0]},
        {"name": "query_model_metrics", "func": query_model_metrics,
         "schema": TOOL_SCHEMAS[1]},
        {"name": "query_code", "func": query_code,
         "schema": TOOL_SCHEMAS[2]},
        {"name": "search_knowledge_base", "func": _search_kb,
         "schema": TOOL_SCHEMAS[3]},
    ]

    return registry


TOOL_REGISTRY = build_tool_registry(retriever=None)
