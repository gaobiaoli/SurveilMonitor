import numpy as np
import sys
sys.path.append('..')
import cv2
import os
import glob
def cal_single_gt(gt_path,mask,H):
    gt_list=glob.glob(gt_path+"/*.png")
    gt_list=sorted(gt_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    gt_dict={}
    A_mask=np.sum(mask>0)
    gt_sum=np.zeros_like(mask)
    for i in range(len(gt_list)):
        gt_i_raw=cv2.imread(gt_list[i],0)
        gt_i = cv2.warpPerspective(
                gt_i_raw, H, mask.shape[::-1], borderValue=0
            )
        gt_sum+=gt_i
        A_gt=np.sum((gt_sum*mask)>0)
        gt_dict[int(os.path.basename(gt_list[i]).split(".")[0])]=A_gt/A_mask
    return gt_dict
def cal_single_pred(mask_list,flag_map):
    percentage_list=[]
    for mask in mask_list:
        A_mask=np.sum(mask>0)
        A_pred=np.sum(flag_map*(mask>0)>0)
        percentage_list.append(A_pred/A_mask)
    return percentage_list