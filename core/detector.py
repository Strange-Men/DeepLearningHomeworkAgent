"""检测引擎模块 - 基于YOLOv8的手势检测。"""

import time
from ultralytics import YOLO

import config


class GestureDetector:
    """手势检测器。"""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_PATH
        self.model = YOLO(self.model_path)
        self.imgsz = config.MODEL_IMAGE_SIZE
        self.conf = config.MODEL_CONFIDENCE_THRESHOLD

    def detect_image(self, image):
        """检测单张图片。

        Args:
            image: numpy数组(BGR格式)

        Returns:
            dict: {"image": 标注图, "gestures": 检测列表, "count": 数量}
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        result = results[0]
        annotated = result.plot()
        gestures = []
        names = self.model.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            gestures.append({
                "class": names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
        return {
            "image": annotated,
            "gestures": gestures,
            "count": len(gestures)
        }

    def detect_frame(self, frame):
        """检测单帧并返回标注帧和FPS。

        Args:
            frame: numpy数组(BGR格式)

        Returns:
            tuple: (annotated_frame, fps)
        """
        start = time.time()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        annotated = results[0].plot()
        return annotated, fps
