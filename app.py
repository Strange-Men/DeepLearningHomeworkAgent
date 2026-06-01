"""主入口 - Gradio手势识别界面。"""

import os

# HuggingFace 国内镜像（必须在所有 HF 相关 import 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 添加 libs 目录到 DLL 搜索路径（解决 OpenH264 加载问题）
os.add_dll_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

import json
import logging
import time
import shutil
import subprocess
import cv2
import gradio as gr

import config

config.setup_logging()
logger = logging.getLogger(__name__)
from core.i18n import init as i18n_init, t, set_language, get_language
from core.detector import GestureDetector
from core.agent import LangChainAgent
from core.history import HistoryManager
from utils.image_utils import pil_to_cv2, cv2_to_pil

import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

i18n_init()

detector = GestureDetector()
try:
    agent = LangChainAgent()
except Exception as e:
    logger.warning(t("app.agent_init_warning", error=e), exc_info=True)
    from core.agent import _FallbackAgent
    agent = _FallbackAgent()
history = HistoryManager()

_LANG_LABELS = {
    "zh-CN": "简体中文",
    "zh-TW": "繁體中文",
    "en": "English",
    "fr": "Français",
}


def detect_image_fn(image):
    """图片识别处理函数。"""
    if image is None:
        return None, t("msg.please_upload_image"), ""
    cv_image = pil_to_cv2(image)
    result = detector.detect_image(cv_image)
    result_pil = cv2_to_pil(result["image"])
    gestures = result["gestures"]
    if not gestures:
        stats = t("msg.no_gesture_detected")
        detail = ""
    else:
        stats = t("msg.detected_count", count=result['count'])
        detail_lines = []
        for i, g in enumerate(gestures, 1):
            conf_str = f"{g['confidence']:.2%}"
            detail_lines.append(
                f"{i}. {g['class']} "
                f"({t('msg.confidence', value=conf_str)})"
            )
        detail = "\n".join(detail_lines)
    result_path = os.path.join(
        config.RESULTS_DIR, f"img_{int(time.time())}.jpg"
    )
    cv2.imwrite(result_path, result["image"])
    history.add_record(
        "image", "", result_path, gestures, result["count"]
    )
    return result_pil, stats, detail


def _get_ffmpeg_exe():
    """获取 ffmpeg 路径：优先系统安装，其次 imageio-ffmpeg 捆绑版。"""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        return None


def _convert_to_h264_mp4(src_path):
    """将视频转码为浏览器可播放的 H.264 MP4，返回新路径或 None。"""
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        return None
    dst_path = os.path.splitext(src_path)[0] + ".mp4"
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y", "-i", src_path,
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                dst_path,
            ],
            capture_output=True, timeout=300,
        )
        if (result.returncode == 0
                and os.path.exists(dst_path)
                and os.path.getsize(dst_path) > 0):
            return dst_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def process_video_fn(video):
    """视频逐帧识别处理函数。"""
    if video is None:
        return None, t("msg.please_upload_video")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, t("msg.cannot_open_video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 用 XVID 写入 AVI（可靠，不依赖外部编码库）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    avi_path = os.path.join(
        config.HISTORY_VIDEOS_DIR, f"output_{timestamp}.avi"
    )
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None, t("msg.encoder_init_failed")
    # 每个类别出现的帧数（同一帧内同一类别只计1次）
    class_frame_counts = {}
    frame_count = 0
    conf_threshold = 0.5
    names = detector.model.names
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.model.predict(
            source=frame,
            imgsz=detector.imgsz,
            conf=detector.conf,
            verbose=False
        )
        result = results[0]
        annotated = result.plot()
        # 当前帧中已记录的类别（去重）
        frame_classes = set()
        for box in result.boxes:
            conf_val = float(box.conf[0])
            if conf_val < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            if cls_name not in frame_classes:
                frame_classes.add(cls_name)
                class_frame_counts[cls_name] = (
                    class_frame_counts.get(cls_name, 0) + 1
                )
        writer.write(annotated)
        frame_count += 1
    total_gestures = len(class_frame_counts)
    cap.release()
    writer.release()
    if frame_count == 0:
        return None, t("msg.video_empty")
    if not os.path.exists(avi_path) or os.path.getsize(avi_path) == 0:
        return None, t("msg.video_write_failed")
    # 转码为 H.264 MP4，确保浏览器可播放
    mp4_path = _convert_to_h264_mp4(avi_path)
    if mp4_path:
        os.remove(avi_path)
        result_path = mp4_path
    else:
        result_path = avi_path
    detections_summary = [
        {"class": name, "frame_count": count}
        for name, count in class_frame_counts.items()
    ]
    history.add_record(
        "video", video, result_path, detections_summary, total_gestures, fps
    )
    return result_path, t("msg.process_complete", frames=frame_count, fps=f"{fps:.1f}")


camera_running = False
camera_cap = None
camera_frame_count = 0
CAMERA_SAVE_INTERVAL = 30


def start_camera():
    """启动摄像头。"""
    global camera_running, camera_cap, camera_frame_count
    camera_running = True
    camera_cap = cv2.VideoCapture(0)
    camera_frame_count = 0
    if not camera_cap.isOpened():
        camera_running = False
        return None, t("msg.cannot_open_camera")
    return None, t("msg.camera_started")


def stop_camera():
    """停止摄像头。"""
    global camera_running, camera_cap
    camera_running = False
    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None
    return None, t("msg.camera_stopped")


def capture_frame():
    """获取摄像头帧并检测。"""
    global camera_frame_count
    if not camera_running or camera_cap is None:
        return None, t("msg.camera_not_started")
    ret, frame = camera_cap.read()
    if not ret:
        return None, t("msg.cannot_read_frame")

    # 直接运行模型预测以获取检测详情
    start = time.time()
    results = detector.model.predict(
        source=frame,
        imgsz=detector.imgsz,
        conf=detector.conf,
        verbose=False
    )
    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0
    result = results[0]
    annotated = result.plot()

    gestures = []
    names = detector.model.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        gestures.append({
            "class": names[cls_id],
            "confidence": float(box.conf[0]),
        })

    camera_frame_count += 1

    # 每隔 N 帧且检测到手势时保存一次历史记录
    if (camera_frame_count % CAMERA_SAVE_INTERVAL == 0
            and gestures):
        result_path = os.path.join(
            config.RESULTS_DIR,
            f"cam_{int(time.time())}.jpg"
        )
        cv2.imwrite(result_path, annotated)
        history.add_record(
            "camera", "", result_path, gestures, len(gestures)
        )

    result_pil = cv2_to_pil(annotated)
    info = f"FPS: {fps:.1f}"
    return result_pil, info


def agent_answer(message, chat_history):
    """Agent问答处理函数。"""
    try:
        if not message.strip():
            return chat_history, ""
        lang = get_language()
        response = agent.answer(message, lang=lang)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, ""
    except Exception:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({
            "role": "assistant",
            "content": t("msg.answer_failed"),
        })
        return chat_history, ""


def quick_question_fn(question):
    """快捷问题处理。"""
    return question


def load_history_list():
    """加载历史记录列表。"""
    records = history.get_records()
    if not records:
        return gr.update(choices=[], value=None), t("msg.no_history")
    choices = []
    for r in records:
        label = (
            f"[{r['id']}] {r['source_type']} | "
            f"{r['created_at'][:19]} | "
            f"{t('tools.gesture_count_short', count=r['gesture_count'])}"
        )
        choices.append(label)
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        t("msg.record_count", count=len(records)),
    )


def view_record_detail(selected):
    """查看记录详情。"""
    if not selected:
        return None, None, t("msg.please_select_record")
    record_id = int(selected.split("]")[0].replace("[", ""))
    record = history.get_record_by_id(record_id)
    if not record:
        return None, None, t("msg.record_not_found")
    result_path = record.get("result_path", "")
    image = None
    video = None
    if result_path and os.path.exists(result_path):
        if record["source_type"] == "video":
            video = result_path
        else:
            img = cv2.imread(result_path)
            if img is not None:
                image = cv2_to_pil(img)
    detections = json.loads(record.get("detections", "{}"))
    gestures = detections if isinstance(detections, list) else []
    detail = f"{t('msg.record_id', id=record['id'])}\n"
    detail += f"{t('msg.record_type', type=record['source_type'])}\n"
    detail += f"{t('msg.record_time', time=record['created_at'])}\n"
    detail += (
        f"{t('msg.gesture_count_label', count=record['gesture_count'])}\n"
    )
    if record.get('fps'):
        detail += f"FPS: {record['fps']:.1f}\n"
    if gestures:
        detail += f"\n{t('msg.detection_results')}\n"
        for g in gestures:
            if 'frame_count' in g:
                detail += (
                    f"  {g.get('class', 'N/A')}: "
                    f"{t('msg.frame_count', count=g['frame_count'])}\n"
                )
            elif 'total_count' in g:
                detail += (
                    f"  {g.get('class', 'N/A')}: "
                    f"{t('msg.count_times', count=g['total_count'])}\n"
                )
            else:
                detail += (
                    f"  {g.get('class', 'N/A')}"
                    f" ({g.get('confidence', 0):.2%})\n"
                )
    return image, video, detail


def delete_record_fn(selected):
    """删除选中记录。"""
    if not selected:
        return t("msg.please_select_to_delete")
    record_id = int(selected.split("]")[0].replace("[", ""))
    file_deleted = history.delete_record(record_id)
    if file_deleted:
        return t("msg.record_deleted_with_file")
    return t("msg.record_deleted")


def clear_all_records():
    """清空所有记录。"""
    deleted_count = history.clear_all()
    return t("msg.all_records_cleared", count=deleted_count)


def change_language(lang):
    """切换语言并更新所有组件文本。

    Args:
        lang: 语言代码

    Returns:
        list: 所有组件的 gr.update()
    """
    set_language(lang)

    # 重新加载 Agent（切换知识库）
    try:
        agent.reload(lang)
    except AttributeError:
        pass

    # 获取快捷问题
    try:
        quick_qs = agent.get_quick_questions()
    except Exception:
        quick_qs = []
    q_btns = []
    for i in range(6):
        if i < len(quick_qs):
            q_btns.append(gr.update(value=quick_qs[i], visible=True))
        else:
            q_btns.append(gr.update(visible=False))

    return [
        gr.update(label="📷 " + t("tab.image")),   # tab_image
        gr.update(label="🎬 " + t("tab.video")),   # tab_video
        gr.update(label="📹 " + t("tab.camera")),  # tab_camera
        gr.update(label="🤖 " + t("tab.agent")),   # tab_agent
        gr.update(label="📋 " + t("tab.history")),  # tab_history
        gr.update(value=t("app.title")),          # header_title
        gr.update(value=t("app.subtitle")),        # header_subtitle
        gr.update(label=t("label.upload_image")),   # img_input
        gr.update(value=t("btn.detect")),           # img_btn
        gr.update(label=t("label.detection_result")),  # img_output
        gr.update(label=t("label.statistics")),     # img_stats
        gr.update(label=t("label.detail_info")),    # img_detail
        gr.update(label=t("label.upload_video")),   # vid_input
        gr.update(value=t("btn.process")),          # vid_btn
        gr.update(label=t("label.detection_result")),  # vid_output
        gr.update(label=t("label.process_status")),  # vid_status
        gr.update(value=t("btn.detect")),           # cam_start_btn
        gr.update(value=t("btn.stop")),             # cam_stop_btn
        gr.update(label=t("label.realtime_frame")),  # cam_output
        gr.update(label=t("label.status_info")),    # cam_info
        gr.update(label=t("label.chat_history")),   # chatbot
        gr.update(label=t("label.input_question")),  # msg_input
        gr.update(placeholder=t("msg.placeholder_question")),  # msg_input placeholder
        gr.update(value=t("btn.send")),             # send_btn
        *q_btns,                                     # quick question buttons (6)
        gr.update(value=t("btn.refresh_list")),     # refresh_btn
        gr.update(value=t("btn.clear_all")),        # clear_btn
        gr.update(label=t("label.record_stats")),   # history_count
        gr.update(label=t("label.history_list")),   # history_list
        gr.update(value=t("btn.view_detail")),      # view_btn
        gr.update(value=t("btn.delete_selected")),  # del_btn
        gr.update(label=t("label.result_image")),   # detail_image
        gr.update(label=t("label.result_video")),   # detail_video
        gr.update(label=t("label.record_detail")),  # detail_text
    ]


CUSTOM_CSS = """
/* ========== 全局基础 ========== */
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 50%, #e8ecf4 100%) !important;
    min-height: 100vh;
    padding: 16px 24px 32px !important;
}

/* ========== 顶部标题区域 ========== */
.gradio-container > .header,
.gradio-container > div:first-child {
    text-align: center;
}

.gradio-container h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    margin-bottom: 4px !important;
    letter-spacing: 0.5px;
}

.gradio-container h1 + p,
.gradio-container h1 + div p {
    font-size: 1rem !important;
    color: #6b7280 !important;
    margin-bottom: 16px !important;
}

/* ========== Tab 栏 ========== */
.tabs > .tab-nav {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
}

.tabs > .tab-nav > button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.3s ease !important;
    position: relative;
}

.tabs > .tab-nav > button:hover {
    color: #1e293b !important;
    background: rgba(255, 255, 255, 0.5) !important;
}

.tabs > .tab-nav > button.selected {
    background: #ffffff !important;
    color: #1e293b !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    font-weight: 600 !important;
}

/* ========== Tab 内容面板 ========== */
.tabitem {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
}

/* ========== 按钮 ========== */
button.primary,
button.lg.primary {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
    padding: 10px 24px !important;
}

button.primary:hover,
button.lg.primary:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
    transform: translateY(-1px) !important;
}

button.secondary,
button.lg.secondary {
    background: rgba(255, 255, 255, 0.85) !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.3s ease !important;
    padding: 10px 24px !important;
}

button.secondary:hover,
button.lg.secondary:hover {
    background: #ffffff !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

button.stop,
button.lg.stop {
    background: linear-gradient(135deg, #ef4444 0%, #f87171 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.3) !important;
    transition: all 0.3s ease !important;
}

button.stop:hover,
button.lg.stop:hover {
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* ========== 输入框 / Textbox ========== */
textarea,
input[type="text"] {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.3s ease !important;
    padding: 10px 14px !important;
}

textarea:focus,
input[type="text"]:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15), 0 2px 8px rgba(0, 0, 0, 0.06) !important;
}

/* ========== 下拉框 Dropdown ========== */
.gradio-dropdown {
    border-radius: 10px !important;
}

.gradio-dropdown > .wrap > .secondary-wrap > .single-select {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: rgba(255, 255, 255, 0.9) !important;
}

/* ========== 图片 / 视频 上传 & 预览区域 ========== */
.image-upload-area,
.video-upload-area,
.upload-container {
    border-radius: 12px !important;
    overflow: hidden !important;
}

.image-container,
.video-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
}

/* 上传区域虚线框 */
.upload-box {
    border-radius: 12px !important;
    border: 2px dashed #cbd5e1 !important;
    background: rgba(248, 250, 252, 0.8) !important;
    transition: all 0.3s ease !important;
}

.upload-box:hover {
    border-color: #818cf8 !important;
    background: rgba(248, 250, 252, 1) !important;
}

/* ========== Chatbot 消息气泡 ========== */
.chatbot .message {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
    line-height: 1.6 !important;
    max-width: 80% !important;
    word-wrap: break-word !important;
}

.chatbot .user {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: #fff !important;
    margin-left: auto !important;
    border-bottom-right-radius: 4px !important;
}

.chatbot .bot,
.chatbot .assistant {
    background: rgba(255, 255, 255, 0.92) !important;
    color: #1e293b !important;
    margin-right: auto !important;
    border-bottom-left-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
}

.chatbot .message-wrap {
    padding: 8px 12px !important;
}

/* Chatbot 容器 */
.chatbot {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    background: rgba(248, 250, 252, 0.6) !important;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}

/* 快捷问题小按钮 */
.quick-question-btn,
button.sm {
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    padding: 6px 16px !important;
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid #e2e8f0 !important;
    color: #6366f1 !important;
    transition: all 0.3s ease !important;
}

button.sm:hover {
    background: #eef2ff !important;
    border-color: #818cf8 !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12) !important;
}

/* ========== 摄像头 Tab 启动/停止按钮行 ========== */
#component-0 .row {
    gap: 12px !important;
}

/* ========== 历史记录 Tab ========== */
/* Radio 列表 */
.gradio-radio .wrap,
.gradio-radio label {
    border-radius: 10px !important;
}

.gradio-radio input[type="radio"] + span,
.gradio-radio .radio-item {
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    background: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid #e8ecf0 !important;
    transition: all 0.3s ease !important;
}

.gradio-radio input[type="radio"]:checked + span,
.gradio-radio .radio-item.selected {
    background: #eef2ff !important;
    border-color: #818cf8 !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12) !important;
}

/* ========== 通用面板卡片化 ========== */
.block,
.gradio-group,
.gradio-box {
    border-radius: 12px !important;
}

/* ========== Label 标签美化 ========== */
label > .label-wrap > span,
.label-wrap > .label-text {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 0.88rem !important;
}

/* ========== 滚动条美化 ========== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* ========== 图片/视频结果区域圆角 ========== */
.preview-container,
.image-preview,
video {
    border-radius: 12px !important;
}

/* ========== 响应式微调 ========== */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px 12px 16px !important;
    }
    .tabitem {
        padding: 16px !important;
    }
    .tabs > .tab-nav > button {
        padding: 8px 12px !important;
        font-size: 0.82rem !important;
    }
}
"""

with gr.Blocks(title=t("app.title")) as demo:
    header_title = gr.Markdown(f"# {t('app.title')}")
    header_subtitle = gr.Markdown(t("app.subtitle"))

    # 语言切换下拉框
    lang_dropdown = gr.Dropdown(
        choices=list(_LANG_LABELS.values()),
        value=_LANG_LABELS[get_language()],
        label=t("label.language"),
        interactive=True,
    )

    with gr.Tab("📷 " + t("tab.image")) as tab_image:
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label=t("label.upload_image"))
                img_btn = gr.Button(t("btn.detect"), variant="primary")
            with gr.Column():
                img_output = gr.Image(type="pil", label=t("label.detection_result"))
                img_stats = gr.Textbox(label=t("label.statistics"), interactive=False)
                img_detail = gr.Textbox(
                    label=t("label.detail_info"), interactive=False, lines=6
                )
        img_btn.click(
            fn=detect_image_fn,
            inputs=[img_input],
            outputs=[img_output, img_stats, img_detail]
        )

    with gr.Tab("🎬 " + t("tab.video")) as tab_video:
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label=t("label.upload_video"))
                vid_btn = gr.Button(t("btn.process"), variant="primary")
            with gr.Column():
                vid_output = gr.Video(label=t("label.detection_result"))
                vid_status = gr.Textbox(
                    label=t("label.process_status"), interactive=False
                )
        vid_btn.click(
            fn=process_video_fn,
            inputs=[vid_input],
            outputs=[vid_output, vid_status]
        )

    with gr.Tab("📹 " + t("tab.camera")) as tab_camera:
        with gr.Row():
            cam_start_btn = gr.Button(t("btn.detect"), variant="primary")
            cam_stop_btn = gr.Button(t("btn.stop"), variant="stop")
        cam_output = gr.Image(type="pil", label=t("label.realtime_frame"))
        cam_info = gr.Textbox(label=t("label.status_info"), interactive=False)
        cam_timer = gr.Timer(0.033)
        cam_timer.tick(
            fn=capture_frame, inputs=[], outputs=[cam_output, cam_info]
        )
        cam_start_btn.click(
            fn=start_camera, inputs=[], outputs=[cam_output, cam_info]
        )
        cam_stop_btn.click(
            fn=stop_camera, inputs=[], outputs=[cam_output, cam_info]
        )

    with gr.Tab("🤖 " + t("tab.agent")) as tab_agent:
        chatbot = gr.Chatbot(label=t("label.chat_history"), height=400)
        with gr.Row():
            msg_input = gr.Textbox(
                label=t("label.input_question"),
                placeholder=t("msg.placeholder_question"),
                scale=4,
            )
            send_btn = gr.Button(t("btn.send"), variant="primary", scale=1)
        try:
            quick_qs = agent.get_quick_questions()
        except Exception:
            quick_qs = []
        quick_btns_row1 = []
        quick_btns_row2 = []
        with gr.Row():
            for q in quick_qs[:3]:
                btn = gr.Button(q, size="sm")
                btn.click(
                    fn=quick_question_fn,
                    inputs=[gr.State(q)],
                    outputs=[msg_input],
                )
                quick_btns_row1.append(btn)
        with gr.Row():
            for q in quick_qs[3:6]:
                btn = gr.Button(q, size="sm")
                btn.click(
                    fn=quick_question_fn,
                    inputs=[gr.State(q)],
                    outputs=[msg_input],
                )
                quick_btns_row2.append(btn)
        # Pad to 6 buttons total for change_language callback
        all_quick_btns = quick_btns_row1 + quick_btns_row2
        while len(all_quick_btns) < 6:
            dummy = gr.Button(visible=False)
            all_quick_btns.append(dummy)
        send_btn.click(
            fn=agent_answer,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        msg_input.submit(
            fn=agent_answer,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )

    with gr.Tab("📋 " + t("tab.history")) as tab_history:
        with gr.Row():
            refresh_btn = gr.Button(t("btn.refresh_list"), variant="primary")
            clear_btn = gr.Button(t("btn.clear_all"), variant="stop")
        history_count = gr.Textbox(label=t("label.record_stats"), interactive=False)
        history_list = gr.Radio(
            label=t("label.history_list"), choices=[], interactive=True
        )
        with gr.Row():
            view_btn = gr.Button(t("btn.view_detail"))
            del_btn = gr.Button(t("btn.delete_selected"))
        with gr.Row():
            detail_image = gr.Image(type="pil", label=t("label.result_image"))
            detail_video = gr.Video(label=t("label.result_video"))
            detail_text = gr.Textbox(
                label=t("label.record_detail"), interactive=False, lines=10
            )
        refresh_btn.click(
            fn=load_history_list, inputs=[], outputs=[history_list, history_count]
        )
        view_btn.click(
            fn=view_record_detail,
            inputs=[history_list],
            outputs=[detail_image, detail_video, detail_text],
        )
        del_btn.click(
            fn=delete_record_fn, inputs=[history_list], outputs=[history_count]
        )
        clear_btn.click(
            fn=clear_all_records, inputs=[], outputs=[history_count]
        )

    # 语言切换回调
    _all_outputs = [
        tab_image, tab_video, tab_camera, tab_agent, tab_history,
        header_title, header_subtitle,
        img_input, img_btn, img_output, img_stats, img_detail,
        vid_input, vid_btn, vid_output, vid_status,
        cam_start_btn, cam_stop_btn, cam_output, cam_info,
        chatbot, msg_input, msg_input, send_btn,
        *all_quick_btns,
        refresh_btn, clear_btn, history_count, history_list,
        view_btn, del_btn, detail_image, detail_video, detail_text,
    ]

    def _on_language_change(label):
        """语言下拉框回调：label -> lang code。"""
        label_to_code = {v: k for k, v in _LANG_LABELS.items()}
        lang = label_to_code.get(label, "zh-CN")
        return change_language(lang)

    lang_dropdown.change(
        fn=_on_language_change,
        inputs=[lang_dropdown],
        outputs=_all_outputs,
    )

if __name__ == "__main__":
    demo.launch(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        share=config.APP_SHARE,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
