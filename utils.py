import cv2 # type: ignore
import os
import numpy as np  # type: ignore
from datetime import datetime

def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = max(x2_1, x2_2)
    y2_i = max(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def expand_bbox(bbox, frame_shape, padding_ratio=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(frame_shape[1], x2 + padding_x)
    y2 = min(frame_shape[0], y2 + padding_y)

    return (x1, y1, x2, y2)

def get_timestamp():
    current_time = datetime.now()
    return {
        'datetime': current_time,
        'date_str': current_time.strftime('%Y-%m-%d'),
        'time_str': current_time.strftime('%H:%M:%S'),
        'filename_timestamp': current_time.strftime('%Y%m%d_%H%M%S')
    }