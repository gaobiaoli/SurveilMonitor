import os
import cv2
import time
import numpy as np
from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot
from utils.videoutils import (
    Player,
    MaskedVideoCapture,
    VibrationCalibrator,
    OrthoMaskCropper,
)

if __name__ == "__main__":
    videoId="D06_20210310122801"
    videoPath = f"/CV/WD16T/2021.02-2021.04/D06/{videoId}.mp4"  # 视频路径
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
    trans_bim = cv2.warpPerspective(
        bimFloor, np.linalg.inv(BimH), baseImg.shape[:2][::-1], borderValue=0
    )
    maskPath = "/home/gaobiaoli/dataset/base/mask"  # mask图像
    cropSavePath="/CV/3T/dataset-public/videoBaseImg/D06/cropImg"
    maskList = os.listdir(maskPath)
    maskFileList = [str(i + 1) + ".png" for i in range(10)]
    maskList = []
    for j in range(len(maskFileList)):
        mask = cv2.imread(os.path.join(maskPath, maskFileList[j]), 0)
        maskList.append(mask)
    croper = OrthoMaskCropper(maskList)

    calibrator = VibrationCalibrator(baseImg=baseImg, baseHomography=baseH)
    cap = MaskedVideoCapture(
        videoPath=videoPath, initStep=1000,interval=500, mtx=mtx, dist=dist, calibrator=calibrator
    )
    player = Player()
    ## model
    checkpoint="/home/gaobiaoli/dataset/iter_2000.pth"
    config="/home/gaobiaoli/dataset/config.py"
    device="cuda:0"
    model = init_model(config, checkpoint, device=device)
    id=3
    while True:
        t1=time.time()
        ret, frame = cap.read()
        if ret:
            slabList = croper.orthogonalize(frame)
            # for i in range(len(slabList)):
            #     filename=videoId+"_"+str(cap.count())+"_"+maskFileList[i]
            #     cv2.imwrite(os.path.join(cropSavePath,filename),slabList[i])
            #     print(filename)
            #
            result=[]
            for i in range(len(slabList)):
                result.append(inference_model(model, slabList[i]))
            draw_img = show_result_pyplot(model, slabList[id], result[id],show=False)

            # result=inference_model(model, frame)
            # draw_img = show_result_pyplot(model, frame, result,show=False)

            player.show(draw_img)
            print("Frame:{}---FPS:{}".format(cap.count(),1/(time.time()-t1)))
        else:
            break
