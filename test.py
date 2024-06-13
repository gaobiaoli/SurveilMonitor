import os
import cv2
import time
import clip
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmcv.transforms import Compose
from mmdet.visualization import DetLocalVisualizer
from PIL import Image

from vUtils.player import Player
from vUtils.capture import BaseVideoCapture,FasterVideoCapture,VibrationCalibrator,DeVibVideoCapture


from ClipAdapter.adapter import ClipAdapter
from utils.clip_utils import (
    batch_crop,
    batch_preprocess,
    assign_clip_label,
    generate_colors
)
from utils.SurRecord import *
if __name__ == "__main__":
    # videoId="D06_20210318134012"
    videoIds = ["D06_20210115183104"]
    videoPath = [f"/CV/WD8T1/06/{videoId}.mp4" for videoId in videoIds]  # 视频路径
    videoId = videoIds[0]
    # videoId="D06_20210318083729"
    # videoPath = f"/CV/WD16T/2021.02-2021.04/D06/{videoId}.mp4"  # 视频路径
    mtx, dist = np.load(
        "/home/gaobiaoli/dataset/undistort/D03.npy", allow_pickle=True
    )  # 畸变参数
    baseImg = cv2.imread(
        f"/CV/3T/dataset-public/videoBaseImg/D06/baseImg/{videoId}.png"
    )  # videobase
    baseH = np.load(
        f"/home/gaobiaoli/dataset/D06/baseH/{videoId}.npy"
    )  # Htobase
    BimH = np.load("/home/gaobiaoli/dataset/base/8F_D06.npy")  # HtoBim
    bimFloor = cv2.imread("/home/gaobiaoli/dataset/base/target8F_L.png")  # Bim图
    
    calibrator = VibrationCalibrator(baseImg=baseImg, baseHomography=baseH)
    cap = DeVibVideoCapture(
        videoPath=videoPath, initStep=0,interval=1, mtx=mtx, dist=dist, calibrator=calibrator
    )

    player = Player()
    ## model

    while True:
        t1=time.time()
        ret, frame = cap.read()
        if ret:
                # player.show(frame)
                print("Frame:{}---FPS:{}".format(cap.count(),1/(time.time()-t1)))
        else:
            break
