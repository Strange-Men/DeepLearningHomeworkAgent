import cv2
from ultralytics import YOLO
import torch
import warnings

# --- 配置 ---
# 1. 模型路径 (请根据你的实际情况修改！)
model_path = 'E:/YOLOv8/ultralytics-main/runs/detect/train4/weights/best.pt'

# 2. 类别名称 (请根据你的 custom_data.yaml 文件修改！)
class_names = ['A', 'number 7', 'D', 'I', 'L', 'V', 'W', 'Y', 'I love you', 'number 5'] # 示例：['cat', 'dog']

# --- 加载模型 (兼容旧版本 PyTorch 的方式) ---
print(f"正在加载模型: {model_path}")

# 忽略 PyTorch 2.6.0 以下版本加载模型时的安全警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Weights only load failed.*')

# 尝试加载模型
try:
    # 对于旧版本 PyTorch，Ultralytics 库内部会处理加载逻辑
    model = YOLO(model_path)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# --- 打开摄像头 ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

print("摄像头已启动。按 'q' 键退出。")

# --- 实时检测循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取视频流。")
        break

    # --- 使用模型进行检测 ---
    results = model(frame, conf=0.25)

    # --- 在图像上绘制检测结果 ---
    annotated_frame = results[0].plot()

    # --- 显示结果 ---
    cv2.imshow("YOLOv8 Camera Detection", annotated_frame)

    # --- 等待按键，如果按下 'q' 则退出循环 ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 释放资源 ---
cap.release()
cv2.destroyAllWindows()
print("程序已退出，资源已释放。")