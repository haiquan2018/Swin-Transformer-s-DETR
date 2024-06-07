import json
import os
import cv2
from tqdm import tqdm
from mmengine.config import Config, DictAction
import pdb
from mmdet.registry import DATASETS
import numpy as np
from copy import deepcopy
cfg = Config.fromfile('/mnt/afs/shuhaoyu/mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_bdd.py')

dataset = DATASETS.build(cfg.train_dataloader.dataset.dataset)
i = 0
for data in dataset:
    img = cv2.imread(data['img_path'])
    img_name = data['img_path'].split('/')[-1]
    json_file = 'data/bdd/det_annotations/train/' + img_name.replace('jpg','json')
    with open(json_file ,'r') as f:
        info = json.load(f)
    old_bbox = []
    img_copy = deepcopy(img)
    for anno_info in info['frames'][0]['objects']:
        if 'box2d' not in anno_info: continue
        bbox = anno_info['box2d']
        x1,y1,x2,y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        old_bbox.append([x1,y1,x2,y2])
        
        img_copy = cv2.rectangle(img_copy, (x1,y1), (x2,y2), (255, 0, 255), 1)
    cv2.imwrite(f'./vis/{i}_old.jpg', img_copy)
    for gt in data['gt_bboxes']:
        x1,y1,x2,y2 = gt.astype(np.int32)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 255), 1)
    cv2.imwrite(f'./vis/{i}.jpg', img)
    i += 1
    pdb.set_trace()