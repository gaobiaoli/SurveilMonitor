import os 
import numpy as np
import cv2
import hashlib
import pickle
from utils.videoutils import VibrationCalibrator
    
if __name__=="__main__":
    baseImgPath='/CV/3T/dataset-public/videoBaseImg/D06/baseImg'                        #待配准图像文件夹(无畸变)
    baseImg=cv2.imread("/home/gaobiaoli/dataset/base/7F_D06_1.png")                       #配准基准图像
    baseH=np.load("/home/gaobiaoli/dataset/base/7F_D06_1.npy")                            #基准图 To Bim
    bimFloor = cv2.imread("/home/gaobiaoli/dataset/base/target7F_L.png")                #Bim图
    mtx, dist = np.load("/home/gaobiaoli/dataset/undistort/D03.npy", allow_pickle=True) #畸变矩阵
    baseHPath="/home/gaobiaoli/dataset/D06/baseH"                                       #H保存位置
    initImg="D06_20210115175538.png"                                                    #中间配准起始点
    border=["D06_20210106092841","D06_20210122140925"]                                  #图像边界，仅处理中间的图像

    save=True
    valid=True
    validPath="/home/gaobiaoli/dataset/D06/test"                                        #测试图像的保存路径

    baseImg=cv2.undistort(baseImg, mtx, dist)
    trans_bim = cv2.warpPerspective(
        bimFloor, np.linalg.inv(baseH), baseImg.shape[:2][::-1], borderValue=0
    )
    trans_bim = cv2.cvtColor(trans_bim, cv2.COLOR_BGR2GRAY)

    
    baseImgList=os.listdir(baseImgPath)


    def get_timestamp(file_name):
    # 假设文件名格式为 "D06-yyyymmddhhmmss"
        return file_name.split('_')[1].split('.')[0]

    # 按时间戳对文件列表进行排序
    baseImgList = sorted(baseImgList, key=get_timestamp)

    initIndex=baseImgList.index(initImg)
    

    # print("########逆序开始########")
    # imgList=baseImgList[0:initIndex+1][::-1]
    # Calibrator=VibrationCalibrator(baseImg=baseImg)
    # for imgfile in imgList:
    #     if imgfile.split(".")[0] in border:
    #         break
    #     img=cv2.imread(os.path.join(baseImgPath,imgfile))
    #     Calibrator.calibrate(img)
    #     H=Calibrator.getHomography()
    #     if save:
    #         np.save(os.path.join(baseHPath,imgfile.split(".")[0]+".npy"),H)
    #     if valid:
    #         img = cv2.warpPerspective(img, H, img.shape[:2][::-1], borderValue=0)
    #         img[trans_bim!=197]=0
    #         cv2.imwrite(os.path.join(validPath,imgfile.split(".")[0]+".jpg"),img)
    #     print(os.path.join(baseHPath,imgfile.split(".")[0]+".npy"))
    
    print("########正序开始########")
    imgList=baseImgList[initIndex:]
    Calibrator=VibrationCalibrator(baseImg=baseImg)
    for imgfile in imgList:
        if imgfile.split(".")[0] in border:
            break
        img=cv2.imread(os.path.join(baseImgPath,imgfile))
        Calibrator.calibrate(img)
        H=Calibrator.getHomography()
        if save:
            np.save(os.path.join(baseHPath,imgfile.split(".")[0]+".npy"),H)
        if valid:
            img = cv2.warpPerspective(img, H, img.shape[:2][::-1], borderValue=0)
            img[trans_bim!=197]=0
            cv2.imwrite(os.path.join(validPath,imgfile.split(".")[0]+".jpg"),img)
        
        print(os.path.join(baseHPath,imgfile.split(".")[0]+".npy"))