import numpy as np
import torch
import colorsys
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
def generate_colors(num_colors):
    # Generate equally spaced HSV colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    
    # Convert HSV to RGB
    rgb_colors = [colorsys.hsv_to_rgb(*color) for color in hsv_colors]
    
    # Convert to 8-bit integers
    rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]
    
    return rgb_colors

def crop(img, bbox, expansion_ratio,square):
    # Ensure bounding box coordinates are integers
    bbox = tuple(map(int, bbox))
    expand_x_factor = 1
    expand_y_factor = 1
    long_edge_factor=0.5
    # Calculate expansion distances
    if square:
        higher= (bbox[3] - bbox[1]) > (bbox[2] - bbox[0])
        expand_x_factor = 1 if higher else long_edge_factor
        expand_y_factor = 1 if not higher else long_edge_factor   

    expand_x = int((bbox[2] - bbox[0]) * expansion_ratio * expand_x_factor) 
    expand_y = int((bbox[3] - bbox[1]) * expansion_ratio * expand_y_factor) 
    # Expand the bounding box
    expanded_bbox = (
        max(0, bbox[0] - expand_x),
        max(0, bbox[1] - expand_y),
        min(img.shape[1], bbox[2] + expand_x),
        min(img.shape[0], bbox[3] + expand_y)
    )

    # Crop the image
    cropped_img = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]

    return cropped_img

def batch_crop(img,bboxes,expansion_ratio=0.1,square=True):
    croped_list=[]
    for bbox in bboxes:
        croped_list.append(crop(img=img,bbox=bbox,expansion_ratio=expansion_ratio,square=square))
    return croped_list

def batch_preprocess(imgs,preprocessFunc):
    imgs=[preprocessFunc(img) for img in imgs]
    return torch.stack(imgs,dim=0)

def assign_clip_label(pred_instances,clip_probs):
    assert len(pred_instances)==len(clip_probs)
    new_instances = InstanceData()
    new_det_data_sample = DetDataSample()
    
    new_instances.bboxes = pred_instances.bboxes
    new_instances.labels = clip_probs.argmax(axis=1)
    new_instances.scores = pred_instances.scores
    new_det_data_sample.pred_instances = new_instances
    return new_det_data_sample