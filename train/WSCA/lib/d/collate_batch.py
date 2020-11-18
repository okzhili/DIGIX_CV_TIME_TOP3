# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

#默认的collate_fn函数是要求一个batch中的图片都具有相同size（因为要做stack操作），
# 当一个batch中的图片大小都不同时，可以使用自定义的collate_fn函数，则一个batch中的图片不再被stack操作

def train_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths
