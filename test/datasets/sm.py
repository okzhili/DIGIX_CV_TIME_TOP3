import glob
import pickle
import re
import os.path as osp

from .bases import BaseImageDataset


class SM(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root='../data', verbose=True, **kwargs):
        super(SM, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()
        query = self._process_dir_test(self.query_dir)
        gallery = self._process_dir_test(self.gallery_dir)

        train,balance_train=[],[]



        if verbose:
            print("=> sm loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.balance_train = balance_train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_test(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        for img_path in img_paths:
            dataset.append((img_path, -1, 1,1))
        return dataset

