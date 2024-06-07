import json
import os
import cv2
from tqdm import tqdm
def json_read(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def box2xywh(bbox):
    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x2']
    y2 = bbox['y2']

    w = x2 - x1 
    h = y2 - y1

    return [x1,y1,w,h], w*h

root = '/mnt/afs/shuhaoyu/mmdetection/data/bdd/det_annotations'
mode = ['train','val']

category_list = []

for m in mode:
    image_id = 0
    anno_id = 0
    img_list = []
    anno_list = []
    lists = os.listdir(os.path.join(root, m))
    for l in tqdm(lists):
        img_info = dict(file_name='', height=None, width=None, id=None)
        infos = json_read(os.path.join(root, m, l))
        name = infos['name']
        img_info['file_name'] = os.path.join('images',m, name+'.jpg')
        img = cv2.imread(os.path.join('/mnt/afs/shuhaoyu/mmdetection/data/bdd',img_info['file_name']))
        h, w, _ = img.shape
        img_info['height'] = h
        img_info['width'] = w
        img_info['id'] = image_id
        img_list.append(img_info)
        for anno in infos['frames'][0]['objects']:
            category = anno['category']
            if category != 'car':continue
            if 'box2d' not in anno: continue
            annotation_info = {'segmentation':[],'bbox':[],'category_id':0,'area':0,'image_id':0,'id':0,
                                'ignore':False, 'iscrowd': 0,'uncertain': True, 'logo': False, 'in_dense_image': False}
            
            if category not in category_list:
                category_list.append(category)
            category_id = category_list.index(category)
            annotation_info['category_id'] = category_id+1
            annotation_info['image_id'] = image_id
            annotation_info['id'] = anno_id
            
                
            box2d = anno['box2d']
            bbox , area = box2xywh(box2d)
            annotation_info['bbox'] = bbox
            annotation_info['area'] = area
            anno_list.append(annotation_info)
            anno_id += 1
        image_id += 1
    categories_list = []
    for idx , c in enumerate(category_list):
        categories_dict = {'id':idx+1, 'name':c , 'supercategory':c}
        categories_list.append(categories_dict)
    data = {'type':'bdd', 'annotations':anno_list,'images':img_list,'categories':categories_list}
    with open('/mnt/afs/shuhaoyu/mmdetection/data/bdd/det_annotations/'+m+'.json','w') as f:
        f.write(json.dumps(data))

    print(category_list, '\n', len(category_list))