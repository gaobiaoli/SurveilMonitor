import numpy as np
import os
import cv2
import torch
def _plot_one_point(point, img, color, radius):
    point = (int(point[0]), int(point[1]))
    cv2.circle(img, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)

def plot_points(points, labels, img, colors, radius=5):
    for label, point in zip(labels,points):
        color = colors[int(label)]
        _plot_one_point(point, img, color, radius)
    return img
def bbox2point(bboxes,x_offset=0.5,y_offset=0.5):
    x = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) * x_offset
    y = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) * y_offset
    return torch.stack([x, y], dim=1).cpu().numpy()
class SurRecord():
    def __init__(self) -> None:
        pass
    def register(self):
        pass