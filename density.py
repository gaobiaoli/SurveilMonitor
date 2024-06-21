import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.ndimage
import scipy.io
import pickle
from vUtils.player import Player
class PDM():
    def __init__(self,density_map,alpha=1.5,T_need=40,FPS=25,frame_interval=500,sigma=25) -> None:
        self.density_map=density_map
        self.flag_map=np.zeros_like(density_map)
        self.delta_map=np.zeros_like(density_map)
        self.sigma = sigma
        self.threshold=self.cal_threshold(alpha=alpha,T_need=T_need,FPS=FPS,frame_interval=frame_interval,sigma=sigma)
        self.alpha=alpha
        self.T_need=T_need
        self.FPS=FPS
        self.frame_interval=frame_interval
        self.sigma=sigma

    def set_alpha(self,alpha):
        self.alpha=alpha
        self.threshold=self.cal_threshold(alpha=alpha,T_need=self.T_need,FPS=self.FPS,frame_interval=self.frame_interval,sigma=self.sigma)
        
        return self.threshold

    def cal_threshold(self,alpha,T_need,FPS,frame_interval,sigma,inplace=False):
        threshold=FPS*T_need/(2*frame_interval*np.pi*sigma**2)*alpha
        if inplace: 
            self.threshold=threshold
        return threshold
    
    def update_map(self,sample,sigma=None):
        if sigma is None:
            sigma=self.sigma
        self.density_map=scipy.ndimage.filters.gaussian_filter(sample, sigma, mode='constant')
        self.flag_map[self.density_map>self.threshold]=1

    def add_sample(self,sample,sigma=None):
        if sigma is None:
            sigma=self.sigma
        self.density_map+=scipy.ndimage.filters.gaussian_filter(sample, sigma, mode='constant')
        self.flag_map[self.density_map>self.threshold]=1
        # self.density_map=scipy.ndimage.uniform_filter(self.density_map,size=10,mode='constant')
        # self.flag_map[self.density_map>self.threshold]=1
        # self.flag_map[self.density_map>self.threshold]=self.density_map[self.density_map>self.threshold]


    def get_PDM(self,clip=False):
        if clip:
            return self.clip(self.density_map,self.threshold)
        return self.density_map
    
    def get_prob(self):
        return self.get_PDM(clip=True)/self.threshold

    def get_flag(self):
        self.flag_map=np.zeros_like(self.density_map)
        self.flag_map[self.density_map>self.threshold]=1
        return self.flag_map
    
    def clip(self,density_map,threshold):
        return np.minimum(density_map,threshold)
    
class Delta():
    def __init__(self,img,shape=None,sigma=30) -> None:
        self.shape=shape if shape is not None else img.shape[:2]
        self.sigma=sigma
        self.mask=self.cal_mask(img)
        self.raw_map=np.zeros(self.shape,dtype=np.float32)

    def cal_mask(self,img):
        # cal bim mask
        mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY_INV)[1]==255
        return mask.astype(np.float32)
    
    def process_mask(self,map):
        mask=(map!=0).astype(np.uint8)
        kernel = np.ones((self.sigma,self.sigma),np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return opening
    
    def process(self,map):
        # post processing
        mask=(map!=0).astype(np.uint8)
        kernel = np.ones((self.sigma,self.sigma),np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return map*opening
    
    def get_raw(self,masked=False,process=False):
        
        output=self.raw_map
        if masked:
            output=output*self.mask
        if process:
            output=self.process(output)
        return output
    
    def add_points(self,points):
        self.raw_map+=self.points2map(points)

    def add_points_list(self,points_list):
        self.raw_map+=self.points_list2map(points_list)

    def points2map(self,points,shape=None):
        # single sample to map 
        if shape==None:
            shape=self.shape
        map=np.zeros(shape[:2],dtype=np.float32)
        for point in points:
            if int(point[1])<0 or int(point[0])<0:
                continue
            if int(point[1])>map.shape[0]-1 or int(point[0])>map.shape[1]-1:
                continue
            map[int(point[1]),int(point[0])]+=1
        return map
    
    def points_list2map(self,points_list,shape=None):
        # a list of samples to map
        if shape==None:
            shape=self.shape
        map=np.zeros(shape[:2],dtype=np.float32)
        for points in points_list:
            map+=self.points2map(points)
        return map
    
    def plot(self,delta=None,masked=False,process=False,color = (0,0,0)):
        if delta is None:
            delta=self.get_raw(masked=masked,process=process)   
        image = np.expand_dims(np.ones_like(delta, dtype=np.uint8), axis=-1).repeat(3, axis=-1)*255
        for y in range(delta.shape[0]):
            for x in range(delta.shape[1]):
                if delta[y, x] != 0:
                    cv2.circle(image, (x, y), 1, color, -1)
        return image
if __name__=="__main__":
    results_path="dataset/result/D06_20210318083729_i100_3.pth"
    results=torch.load(results_path)
    bim_floor = cv2.imread("/home/gaobiaoli/dataset/base/target8F_L.png")
    player=Player(width=bim_floor.shape[0], height=bim_floor.shape[1])
    density=np.zeros(bim_floor.shape[:2],dtype=np.float32)
    pdm=PDM(density,alpha=1.2,frame_interval=100,T_need=30)
    pdm.density_map[-1,-1]=pdm.threshold
    heatmapshow = None
    history=[]
    delta = Delta(img=bim_floor,shape=bim_floor.shape[:2])
    for i in range(len(results)):
        delta.add_points(results[i])
        pdm.update_map(delta.get_raw(masked=True,process=True))
        map=pdm.get_PDM(clip=True)
        heatmapshow = cv2.normalize(map,heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        fusion=cv2.addWeighted(heatmapshow,0.5,bim_floor,0.5,1)
        # # #370 530
        # player.show(fusion)
        player.write(fusion)
        print(i)
    player.release()
    # with open(results_path.split('.')[0]+'.pkl', "wb") as file:
    #     pickle.dump(history, file)
