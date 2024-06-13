import numpy as np
import cv2
import torch
from density import PDM
from utils.videoutils import Player
from density import Delta
if __name__=="__main__":
    results_path="dataset/result/D06_20210318083729_i100_2.pth"
    results=torch.load(results_path)
    bim_floor = cv2.imread("/home/gaobiaoli/dataset/base/target8F_L.png")
    player=Player(width=bim_floor.shape[0], height=bim_floor.shape[1])
    density=np.zeros(bim_floor.shape[:2],dtype=np.float32)
    pdm=PDM(density,alpha=1.2,frame_interval=100,T_need=30)
    pdm.density_map[-1,-1]=pdm.threshold
    heatmapshow = None
    history=[]
    delta = Delta(bim_floor)
    results_list=results[0:100]
    delta.add_points_list(results_list)
    
    
    pdm.add_sample(delta.get_raw(masked=True))
    map=pdm.get_PDM(clip=True)
    # map=pdm.get_prob()
    # history.append(map.copy())
    heatmapshow = cv2.normalize(map,heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    fusion=cv2.addWeighted(heatmapshow,0.5,bim_floor,0.5,1)
    # # #370 530
    player.showImg(fusion)
    # player.write(fusion)
    player.release()