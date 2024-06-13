import numpy as np
import cv2
import hashlib
import pickle


from vUtils.capture import FasterVideoCapture,BaseVideoCapture,DeVibVideoCapture


from typing import Union


class Player(object):
    def __init__(self, window_name="Player", width=900, height=600):
        self.window_name = window_name
        self.player_inited = False
        self.writer_inited = False
        self.width = width
        self.height = height

    def init_player(self,window_name,width,height):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def init_writer(self,filename,FPS,width,height):
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.output = cv2.VideoWriter(filename, fourcc, FPS, (width, height))
        self.writer_inited=True
        return self.output
    
    def show(self, img):
        if not self.player_inited:
            self.init_player(self.window_name,self.width,self.height)
            self.player_inited=True
        
        cv2.imshow(self.window_name, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow(self.window_name)
            exit()

    def write(self,img):
        if not self.writer_inited:
            self.init_writer(filename="output.avi",FPS=25,width=img.shape[1], height=img.shape[0])
        self.output.write(img)

    def showImg(self, img):
        if not self.player_inited:
            self.init_player(self.window_name,self.width,self.height)
            self.player_inited=True
        cv2.imshow(self.window_name, img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyWindow(self.window_name)
            exit()
        
    def release(self):
        if self.writer_inited:
            self.output.release()
        cv2.destroyAllWindows()

class VibrationCalibrator:
    """实现newFrame向baseImg的单应性变换"""

    def __init__(self, baseImg=None, baseHomography=None):
        self.H_old2base = baseHomography  # 如果第一张图片无法配准，需要提供初始的单应性矩阵
        self.baseImg = baseImg  # 基准图片
        self.oldImg = baseImg

        self.detector = cv2.ORB_create(nfeatures=30000)
        self.threshold = 20000  # 特征点小于阈值则跳过
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def getHomography(self):
        return self.H_old2base

    def getFeaturePoint(self, img):
        
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def calHomography(self, old_img, new_img):
        """Find a Homography Matrix
        transfer new Image to old Image"""
        kp1, des1 = self.getFeaturePoint(old_img)
        kp2, des2 = self.getFeaturePoint(new_img)

        if len(kp2) < self.threshold:
            print("图像错误：跳过")
            return False, self.getHomography()

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        old_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        new_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(new_pts, old_pts, cv2.RANSAC, 5.0)
        return True, H

    def calibrate(self, newFrame):
        if self.H_old2base is None:
            ret, self.H_old2base = self.calHomography(
                old_img=self.baseImg, new_img=newFrame
            )
        else:
            ret, H_new2old = self.calHomography(old_img=self.oldImg, new_img=newFrame)
            if ret:
                self.H_old2base = np.dot(H_new2old, self.H_old2base)
        if ret:
            self.oldImg = newFrame
        return self.getHomography()


class OrthoMaskCropper:
    def __init__(self, maskList) -> None:
        self.maskList = maskList  # mask的灰度图
        self.maskBox = []  # mask的拟合四边形
        self.orthoMatrix = []  # mask区域的正交变换矩阵
        self.orthoSize = []
        self.maskInit()

    def __len__(self):
        return len(self.maskList)

    def maskInit(self):
        """处理mask,获得四边形bbox、正交化矩阵、正交后的尺寸"""
        for mask in self.maskList:
            ret, box = self.getBoxFromMask(mask)
            if not ret:
                self.maskBox.append(None)
                self.orthoMatrix.append(None)
                self.orthoSize.append(None)
            else:
                x, y, w, h = cv2.boundingRect(box)
                box = self.reorder_box(box.reshape(-1, 2).astype(np.float32))
                dst = np.array(
                    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
                ).reshape(-1, 2)
                matrix = cv2.getPerspectiveTransform(box, dst)
                self.orthoMatrix.append(matrix)
                self.maskBox.append(box)
                self.orthoSize.append((w, h))

    def getBoxFromMask(self, mask) -> (bool, np.array):
        """从灰度图中拟合四边形,其他多边形时输出False"""
        # 膨胀与腐蚀预处理
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 获得轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        # 多边形拟合
        box = cv2.approxPolyDP(contours[0], epsilon, True)
        if len(box) != 4:
            return False, None
        return True, box

    def reorder_box(self, box):
        """将bbox四个点重排序为顺时针,左上角为第一个点"""
        # 找到最左上角的点的索引
        top_left_index = np.argmin(np.sum(box, axis=1))

        # 确保最左上角的点成为第一个点
        box = np.roll(box, -top_left_index, axis=0)

        # 计算质心，以便确定顺时针顺序
        centroid = np.mean(box, axis=0)

        # 按顺时针方向对点进行排序
        angles = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        box = box[sorted_indices]
        return box

    def orthogonalize(self, img: np.array):
        """将img裁切并正交化"""
        orthoImg = []
        for i in range(len(self.orthoMatrix)):
            if self.orthoMatrix[i] is None:
                orthoImg.append(None)
            else:
                orthoImg.append(
                    cv2.warpPerspective(img, self.orthoMatrix[i], self.orthoSize[i])
                )
        return orthoImg


class DeVibVideoCapture(FasterVideoCapture):
    def __init__(
        self,
        videoPath: str,
        initStep: int = 0,
        interval: int = 0,
        mtx: Union[None, np.array] = None,
        dist: Union[None, np.array] = None,
        calibrator: Union[None, VibrationCalibrator] = None,
    ) -> None:
        super().__init__(videoPath=videoPath, interval=interval,initStep=initStep, mtx=mtx, dist=dist)
        self.calibrator = calibrator

    def read(self):
        """间隔interval帧读取"""
        ret,frame = super().read()
        if not ret:
            return False, None
        if self.calibrator is not None:
            Homography = self.calibrator.calibrate(frame)
            frame = cv2.warpPerspective(
                frame, Homography, frame.shape[:2][::-1], borderValue=0
            )
        return ret, frame
