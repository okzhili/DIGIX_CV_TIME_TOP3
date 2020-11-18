import argparse
import os
import sys
from os import mkdir
import cv2
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import functional as F
import json

from tqdm import tqdm

sys.path.append('.')
from utils.bbox_utils import localize_from_map

def vis_actmap(model,img_size, loader, out_dir):
    model.cuda()
    model.eval()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    results = []
    with torch.no_grad():
        a =tqdm(total=(55168+29758)/200)
        for i, batch in enumerate(loader):
            a.update(1)
            data, pid,_,_,img_path = batch
            data = data.cuda()
            featmap = model(data)
            featmap = (featmap**2).sum(1)
            for j in range(featmap.size(0)):
                fm = featmap[j].detach().cpu().numpy()
                fm[0:1, 0:1] = 0
                fm[0:1, 19:20] = 0
                fm[19:20, 19:20] = 0
                fm[19:20, 0:1] = 0
                fm = cv2.resize(fm,  (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                bbox = localize_from_map(fm, threshold_ratio=1.5)

                img = cv2.imread(img_path[j])
                height, width, _ = img.shape
                bbox = np.array(bbox, dtype=np.float32)
                bbox[0::2] *= width / img_size[1]
                bbox[1::2] *= height / img_size[0]

                bbox = np.array(bbox, dtype=np.int)

                results.append({'img_path': '/'.join(img_path[j].split('/')[-2:]), 'bbox': bbox.tolist()})

                crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                path = out_dir+'/'+img_path[j].split('/')[-2]
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(path+'/'+img_path[j].split('/')[-1], crop)
