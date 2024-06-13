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
    videoIds = ["D06_20210318083729","D06_20210318091309"]
    videoPath = [f"/CV/WD16T/2021.02-2021.04/D06/{videoId}.mp4" for videoId in videoIds]  # 视频路径
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
    trans_bim = cv2.warpPerspective(
        bimFloor, np.linalg.inv(BimH), baseImg.shape[:2][::-1], borderValue=0
    )
    trans_bim = cv2.cvtColor(trans_bim, cv2.COLOR_BGR2GRAY)
    cropSavePath="/CV/3T/dataset-public/videoBaseImg/D06/cropImg"
    
    

    calibrator = VibrationCalibrator(baseImg=baseImg, baseHomography=baseH)
    cap = DeVibVideoCapture(
        videoPath=videoPath, initStep=0,interval=100, mtx=mtx, dist=dist, calibrator=calibrator
    )

    player = Player()
    ## model
    checkpoint="/home/gaobiaoli/dataset/epoch_100.pth"
    config="/home/gaobiaoli/dataset/det-config.py"
    device="cuda:3"
    model = init_detector(config, checkpoint, device=device)
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    ## clip
    clip_token=["a photo of a worker squatting","a photo of a worker bending", "a photo of a worker standing"]
    clip_meta = dict(
        classes=tuple(clip_token),
        palette=generate_colors(len(clip_token)))
    visualizer.dataset_meta=clip_meta
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_adapter=ClipAdapter(clip_model,classnames=clip_token,device=device)
    clip_adapter.load("/home/gaobiaoli/dataset/adapter1.pth")
    status_list=[]
    img_path=f"/home/gaobiaoli/dataset/result/{videoId}"
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    # dir="/home/gaobiaoli/dataset/result/worker"
    # id=0
    while True:
        t1=time.time()
        ret, frame = cap.read()
        if ret:
            result = inference_detector(model, frame, test_pipeline=test_pipeline)
            
            result.pred_instances = result.pred_instances[
                    result.pred_instances.scores > 0.4]
            # clip
            img_batch = batch_crop(img=frame, bboxes=result.pred_instances.bboxes,expansion_ratio=0.2,square=True)
            img_batch = [
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for img in img_batch
            ]
            if len(img_batch)>0:
                imagebatch = batch_preprocess(imgs=img_batch, preprocessFunc=preprocess)
                probs=clip_adapter(imagebatch).cpu()
                result = assign_clip_label(result.pred_instances, probs)
                # 保存检测结果(坐标点)
                result.pred_instances = result.pred_instances[
                    result.pred_instances.labels==0]
                points=bbox2point(result.pred_instances.bboxes,y_offset=0.75)
                if len(points) > 0:
                    points=cv2.perspectiveTransform(np.array(points).reshape(-1,1,2),BimH).reshape(-1,2).tolist()
                else:
                    points=[]
                # frame = plot_points(points,result.pred_instances.labels,bimFloor,generate_colors(len(clip_token)))
                # status_list.append(points)
                # torch.save(status_list,f"/home/gaobiaoli/dataset/result/{videoId}_i10_4.pth")
                #
                # cv2.imwrite(os.path.join(img_path,f"{cap.count()}.jpg"),frame)
                
                # 保存worker
                # for i,img in enumerate(img_batch):
                #     if result.pred_instances.scores[i]>0.5:
                #         img.save(os.path.join(dir,f"{id}.jpg"))
                #         id+=1

                #
                visualizer.add_datasample(
                    name='video',
                    image=frame,
                    data_sample=result,
                    draw_gt=False,
                    show=False,
                    pred_score_thr=0.4)
                frame = visualizer.get_image()
                # frame = plot_points(points,result.pred_instances.labels,frame,generate_colors(len(clip_token)))
                
                # player.show(frame)
                # player.write(frame)
            print("Frame:{}---FPS:{}".format(cap.count(),1/(time.time()-t1)))
        else:
            break
