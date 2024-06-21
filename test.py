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
    videoIds = [
         "D05_20210318092518",
                
        "D05_20210318100143"
                ]
    videoPath = [f"G:\\surveillance\\{videoId}.mp4" for videoId in videoIds]
    videoId = videoIds[0]
    # videoId="D06_20210318083729"
    # videoPath = f"/CV/WD16T/2021.02-2021.04/D06/{videoId}.mp4"  # 视频路径
    mtx, dist = np.load(
        "undistort/D03.npy", allow_pickle=True
    )  # 畸变参数
    baseImg = cv2.imread(
        "base\\8F_D05.png"
    )  # videobase
    baseH = None # Htobase
    baseImg = cv2.undistort(baseImg, mtx, dist)
    
    calibrator = VibrationCalibrator(baseImg=baseImg, baseHomography=baseH)
    cap = DeVibVideoCapture(
        videoPath=videoPath, initStep=60000,interval=100, mtx=mtx, dist=dist, calibrator=calibrator
    )

    player = Player()
    ## model

    while True:
        t1=time.time()
        ret, frame = cap.read()
        if cap.count()>65000:
            cv2.imwrite(f"result/gt/{videoId}/{cap.count()}.jpg",frame)
        if ret:
                player.show(frame)
                print("Frame:{}---FPS:{}".format(cap.count(),1/(time.time()-t1)))
        else:
            break
