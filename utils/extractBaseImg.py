import cv2
import os
import numpy as np
from multiprocessing import Pool
from utils.videoutils import BaseVideoCapture

if __name__ == "__main__":
    # ImgSavePath="/CV/3T/dataset-public/videoBaseImg/D06/baseImg"
    # # videoPath="/CV/WD16T/2020.11-2021.01/D6"
    # videoPath="/CV/WD8T1/06"
    # mtx, dist = np.load("/home/gaobiaoli/dataset/undistort/D03.npy", allow_pickle=True)
    # videoList=os.listdir(videoPath)
    # def process(video):
    #     capture = BaseVideoCapture(os.path.join(videoPath, video),mtx=mtx,dist=dist)
    #     ret, frame = capture.read(interval=1500)
    #     if ret:
    #         savePath = os.path.join(ImgSavePath,video.split(".")[0] + ".png")
    #         cv2.imwrite(savePath,frame)
    #     print(video)
    # pool=Pool(10)
    # for video in videoList:
    #     pool.apply_async(func=process,args=(video,))
    #     # process(video=video)
    # pool.close()
    # pool.join()

    if True:
        # videoPath="/CV/WD16T/2020.11-2021.01/D6/D06_20201218090732.mp4"
        videoPath="/CV/WD8T1/06/D06_20210115153242.mp4"
        capture = BaseVideoCapture(videoPath)
        ret, frame = capture.read()
        cv2.imwrite("/home/gaobiaoli/dataset/base/7F_D06.png",frame)