"""图像工具模块 - PIL与OpenCV格式转换。"""

import numpy as np
from PIL import Image


def pil_to_cv2(pil_image):
    """PIL Image转OpenCV numpy数组(BGR)。"""
    rgb_array = np.array(pil_image)
    if len(rgb_array.shape) == 2:
        return rgb_array
    return rgb_array[:, :, ::-1].copy()


def cv2_to_pil(cv2_image):
    """OpenCV numpy数组(BGR)转PIL Image。"""
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb_array = cv2_image[:, :, ::-1]
    return Image.fromarray(rgb_array)
