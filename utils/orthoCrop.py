import os
import numpy as np
import cv2
from utils.videoutils import OrthoMaskCropper

if __name__ == "__main__":
    baseImgPath = "/CV/3T/dataset-public/videoBaseImg/D06/baseImg"  # base图像
    baseHPath = "/home/gaobiaoli/dataset/D06/baseH"  # 去抖动矩阵
    maskPath = "/home/gaobiaoli/dataset/base/mask"  # mask图像
    cropSavePath = "/home/gaobiaoli/dataset/D06/crop"  # 保存路线
    baseImgList = os.listdir(baseImgPath)
    maskList = os.listdir(maskPath)

    maskFileList = [str(i + 1) + ".png" for i in range(10)]
    maskList = []
    for j in range(len(maskFileList)):
        mask = cv2.imread(os.path.join(maskPath, maskFileList[j]), 0)
        maskList.append(mask)

    croper = OrthoMaskCropper(maskList)

    for i in range(len(baseImgList)):
        imgName = baseImgList[i]
        imgH = np.load(os.path.join(baseHPath, imgName.split(".")[0] + ".npy"))
        img = cv2.imread(os.path.join(baseImgPath, imgName))
        img = cv2.warpPerspective(img, imgH, img.shape[:2][::-1], borderValue=0)

        orthoImgList = croper.orthogonalize(img)
        for j in range(len(orthoImgList)):
            filename = (
                imgName.split(".")[0] + "_" + maskFileList[j].split(".")[0] + ".png"
            )
            path = os.path.join(cropSavePath, filename)
            cv2.imwrite(path, orthoImgList[j])
            print(path)
