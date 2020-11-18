from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import random
# import cv2
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


import torchvision.transforms as T

# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
# _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]






def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # #
            # s = random.randint(0,1)
            # # print(s)
            # if s:
            #     # print(s)
            #     r, g, b = img.split()
            #     channel_list = [r, g, b]
            #     random.shuffle(channel_list)
            #     img =  Image.merge("RGB",channel_list)


            #
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # b, g, r = cv2.split(img)
            # # 随机打乱顺序
            # channel_list = [r, g, b]
            # random.shuffle(channel_list)
            # img = cv2.merge(channel_list)


            # img = np.array(img)
            # img = img.astype(np.float32)

            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    ##### revised by luo
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        # for _, pid, camid, trackid in data:
        # print(data)
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
            # tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        # num_tracks = len(tracks)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid  = self.dataset[index]
        # print(img_path)
        img = read_image(img_path)

        # if(pid>=3094):
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # if(random.random()>0.5):
        #     pass
            # pid = pid+ 3094
            # print(pid)

        if self.transform is not None:
            img = self.transform(img)
            # img1 = val_transforms_scr(img)
            # img2 = val_transforms_v(img)

            # img = (img1,img2)


            # img = img['image']
            # img = img.astype(np.float32)
            # img = img.transpose(2,0,1)



            # print(img.shape)
            # img = torch.tensor(img).float()

        return img, pid, camid,img_path.split('/')[-1]
        # return img, pid, camid,img_path.split('/')[-1]