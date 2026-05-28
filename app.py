"""主入口 - Gradio手势识别界面。"""

import os

# 添加 libs 目录到 DLL 搜索路径（解决 OpenH264 加载问题）
os.add_dll_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

import json
import time
import shutil
import subprocess
import cv2
import gradio as gr

import config
from core.detector import GestureDetector
from core.agent import HybridAgent
from core.history import HistoryManager
from utils.image_utils import pil_to_cv2, cv2_to_pil

detector = GestureDetector()
try:
    agent = HybridAgent()
except Exception as e:
    print(f"[WARNING] Agent初始化失败: {e}，将使用空知识库")

    class _FallbackAgent:
        def answer(self, q):
            return "Agent initialization failed, please check config."

        def get_quick_questions(self):
            return []

        def clear_history(self):
            pass

    agent = _FallbackAgent()
history = HistoryManager()


def detect_image_fn(image):
    """图片识别处理函数。"""
    if image is None:
        return None, "请上传图片", ""
    cv_image = pil_to_cv2(image)
    result = detector.detect_image(cv_image)
    result_pil = cv2_to_pil(result["image"])
    gestures = result["gestures"]
    if not gestures:
        stats = "未检测到手势"
        detail = ""
    else:
        stats = f"检测到 {result['count']} 个手势"
        detail_lines = []
        for i, g in enumerate(gestures, 1):
            detail_lines.append(
                f"{i}. {g['class']} (置信度: {g['confidence']:.2%})"
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
        return None, "请上传视频"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "无法打开视频文件"
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
        return None, "视频编码器初始化失败"
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
        return None, "视频为空"
    if not os.path.exists(avi_path) or os.path.getsize(avi_path) == 0:
        return None, "视频输出文件写入失败"
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
    return result_path, f"处理完成: {frame_count} 帧, FPS: {fps:.1f}"


camera_running = False
camera_cap = None


def start_camera():
    """启动摄像头。"""
    global camera_running, camera_cap
    camera_running = True
    camera_cap = cv2.VideoCapture(0)
    if not camera_cap.isOpened():
        camera_running = False
        return None, "无法打开摄像头，请检查设备连接"
    return None, "摄像头已启动"


def stop_camera():
    """停止摄像头。"""
    global camera_running, camera_cap
    camera_running = False
    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None
    return None, "摄像头已停止"


def capture_frame():
    """获取摄像头帧并检测。"""
    if not camera_running or camera_cap is None:
        return None, "摄像头未启动"
    ret, frame = camera_cap.read()
    if not ret:
        return None, "无法读取摄像头画面"
    annotated, fps = detector.detect_frame(frame)
    result_pil = cv2_to_pil(annotated)
    info = f"FPS: {fps:.1f}"
    return result_pil, info


def agent_answer(message, chat_history):
    """Agent问答处理函数。"""
    try:
        if not message.strip():
            return chat_history, ""
        response = agent.answer(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, ""
    except Exception:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "回答生成失败，请稍后重试。"})
        return chat_history, ""


def quick_question_fn(question):
    """快捷问题处理。"""
    return question


def load_history_list():
    """加载历史记录列表。"""
    records = history.get_records()
    if not records:
        return gr.update(choices=[], value=None), "暂无历史记录"
    choices = []
    for r in records:
        label = f"[{r['id']}] {r['source_type']} | {r['created_at'][:19]} | 手势数: {r['gesture_count']}"
        choices.append(label)
    return gr.update(choices=choices, value=choices[0] if choices else None), f"共 {len(records)} 条记录"


def view_record_detail(selected):
    """查看记录详情。"""
    if not selected:
        return None, None, "请选择记录"
    record_id = int(selected.split("]")[0].replace("[", ""))
    record = history.get_record_by_id(record_id)
    if not record:
        return None, None, "记录不存在"
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
    detail = f"ID: {record['id']}\n"
    detail += f"类型: {record['source_type']}\n"
    detail += f"时间: {record['created_at']}\n"
    detail += f"手势数量: {record['gesture_count']}\n"
    if record.get('fps'):
        detail += f"FPS: {record['fps']:.1f}\n"
    if gestures:
        detail += "\n检测结果:\n"
        for g in gestures:
            if 'frame_count' in g:
                detail += f"  {g.get('class', 'N/A')}: {g['frame_count']}帧\n"
            elif 'total_count' in g:
                detail += f"  {g.get('class', 'N/A')}: {g['total_count']}次\n"
            else:
                detail += (
                    f"  {g.get('class', 'N/A')}"
                    f" ({g.get('confidence', 0):.2%})\n"
                )
    return image, video, detail


def delete_record_fn(selected):
    """删除选中记录。"""
    if not selected:
        return "请选择要删除的记录"
    record_id = int(selected.split("]")[0].replace("[", ""))
    history.delete_record(record_id)
    choices, count_text = load_history_list()
    return "记录已删除"


def clear_all_records():
    """清空所有记录。"""
    history.clear_all()
    return "所有记录已清空"


with gr.Blocks(title="手势识别系统") as demo:
    gr.Markdown("# YOLOv8 手势识别系统")
    gr.Markdown("基于YOLOv8n的实时手势识别系统，支持图片/视频/摄像头识别")

    with gr.Tab("图片识别"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="上传图片")
                img_btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                img_output = gr.Image(type="pil", label="检测结果")
                img_stats = gr.Textbox(label="统计", interactive=False)
                img_detail = gr.Textbox(label="详细信息", interactive=False, lines=6)
        img_btn.click(
            fn=detect_image_fn,
            inputs=[img_input],
            outputs=[img_output, img_stats, img_detail]
        )

    with gr.Tab("视频识别"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="上传视频")
                vid_btn = gr.Button("开始处理", variant="primary")
            with gr.Column():
                vid_output = gr.Video(label="检测结果")
                vid_status = gr.Textbox(label="处理状态", interactive=False)
        vid_btn.click(
            fn=process_video_fn,
            inputs=[vid_input],
            outputs=[vid_output, vid_status]
        )

    with gr.Tab("摄像头识别"):
        with gr.Row():
            cam_start_btn = gr.Button("开始识别", variant="primary")
            cam_stop_btn = gr.Button("停止识别", variant="stop")
        cam_output = gr.Image(type="pil", label="实时画面")
        cam_info = gr.Textbox(label="状态信息", interactive=False)
        cam_timer = gr.Timer(0.033)
        cam_timer.tick(fn=capture_frame, inputs=[], outputs=[cam_output, cam_info])
        cam_start_btn.click(fn=start_camera, inputs=[], outputs=[cam_output, cam_info])
        cam_stop_btn.click(fn=stop_camera, inputs=[], outputs=[cam_output, cam_info])

    with gr.Tab("Agent问答"):
        chatbot = gr.Chatbot(label="对话历史", height=400)
        with gr.Row():
            msg_input = gr.Textbox(label="输入问题", placeholder="请输入您的问题...", scale=4)
            send_btn = gr.Button("发送", variant="primary", scale=1)
        with gr.Row():
            try:
                quick_qs = agent.get_quick_questions()
            except Exception:
                quick_qs = []
            for q in quick_qs[:3]:
                gr.Button(q, size="sm").click(
                    fn=quick_question_fn, inputs=[gr.State(q)], outputs=[msg_input]
                )
        with gr.Row():
            for q in quick_qs[3:6]:
                gr.Button(q, size="sm").click(
                    fn=quick_question_fn, inputs=[gr.State(q)], outputs=[msg_input]
                )
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

    with gr.Tab("历史记录"):
        with gr.Row():
            refresh_btn = gr.Button("刷新列表", variant="primary")
            clear_btn = gr.Button("清空所有", variant="stop")
        history_count = gr.Textbox(label="记录统计", interactive=False)
        history_list = gr.Radio(label="历史记录", choices=[], interactive=True)
        with gr.Row():
            view_btn = gr.Button("查看详情")
            del_btn = gr.Button("删除选中")
        with gr.Row():
            detail_image = gr.Image(type="pil", label="结果图片")
            detail_video = gr.Video(label="结果视频")
            detail_text = gr.Textbox(label="记录详情", interactive=False, lines=10)
        refresh_btn.click(fn=load_history_list, inputs=[], outputs=[history_list, history_count])
        view_btn.click(fn=view_record_detail, inputs=[history_list], outputs=[detail_image, detail_video, detail_text])
        del_btn.click(fn=delete_record_fn, inputs=[history_list], outputs=[history_count])
        clear_btn.click(fn=clear_all_records, inputs=[], outputs=[history_count])

if __name__ == "__main__":
    demo.launch(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        share=config.APP_SHARE,
        theme=gr.themes.Soft()
    )
