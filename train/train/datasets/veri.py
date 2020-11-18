import glob
import re
import os.path as osp

from .bases import BaseImageDataset


# class VeRi(BaseImageDataset):
#     """
#        VeRi-776
#        Reference:
#        Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.
#
#        URL:https://vehiclereid.github.io/VeRi/
#
#        Dataset statistics:
#        # identities: 776
#        # images: 37778 (train) + 1678 (query) + 11579 (gallery)
#        # cameras: 20
#        """
#
#     dataset_dir = 'VeRi'
#
#     def __init__(self, root='../data', verbose=True, **kwargs):
#         super(VeRi, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.dataset_dir, 'image_train')
#         self.query_dir = osp.join(self.dataset_dir, 'image_query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
#
#         self._check_before_run()
#
#         train = self._process_dir(self.train_dir, relabel=True)
#         query = self._process_dir(self.query_dir, relabel=False)
#         gallery = self._process_dir(self.gallery_dir, relabel=False)
#
#         if verbose:
#             print("=> VeRi-776 loaded")
#             self.print_dataset_statistics(train, query, gallery)
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))
#
#     def _process_dir(self, dir_path, relabel=False):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#         pattern = re.compile(r'([-\d]+)_c(\d+)')
#
#         pid_container = set()
#         for img_path in img_paths:
#             pid, _ = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             pid_container.add(pid)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#
#         dataset = []
#         for img_path in img_paths:
#             pid, camid = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             assert 0 <= pid <= 776  # pid == 0 means background
#             assert 1 <= camid <= 20
#             camid -= 1  # index starts from 0
#             if relabel: pid = pid2label[pid]
#             dataset.append((img_path, pid, camid,1))
#
#         return dataset
import glob
import re

import os.path as osp
import os
from .bases import BaseImageDataset
import pickle
"""
Veri数据集命名方式
0001_c015_00013225_0.jpg
0001 车辆ID
c015 相机ID
00XX 文件名
"""


test_flag = False


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VeRi'

    def __init__(self, root='./', verbose=True, **kwargs):
        super(VeRi, self).__init__()

        print(root)
        # self.dataset_dir = osp.join(root, self.dataset_dir)/media/bi/文档1/19/lab/python/Vehicle_Re-identification/doc/dataset/
        self.dataset_dir = '/home/lab3/bi/0716/Veri/VeRi'
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()
        #因为train里面的文件夹类别不是连续的，所以需要重新lable
        #如果是别的数据集可以改成其他方式，读取txt方式
        # train = self._process_dir(self.train_dir, relabel=True)
        # query = self._process_dir(self.query_dir, relabel=False)
        # gallery = self._process_dir(self.gallery_dir, relabel=False)
        train = self.get_lable_data(root,state="train")
        query = []
        gallery = []
        # train = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/all_train_scr.pkl',state="train")
        # train = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/mix_train_balance.pkl',state="train")
        # query = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/huafen_all_train.pkl',state="query")
        # query = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/huafen_all_train.pkl',state="query")
        # gallery = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/huafen_all_train.pkl',state="gallery")
        if test_flag:


            #scr
            train = self.train = self.get_lable_data(root,state="train")
            # train = self.get_lable_data2('/home/lab3/bi/0716/Veri/ai_city/label_train.txt')
            query = self.get_lable_data1('./datasets/crop_query_B.txt',gallery_flag=False)
            gallery = self.get_lable_data1('./datasets/crop_gallery_B.txt',gallery_flag=True)

            # train = self.train = self.get_lable_data(root, state="train")
            # # train = self.get_lable_data2('/home/lab3/bi/0716/Veri/ai_city/label_train.txt')
            # query =  self.train = self.get_lable_data(root, state="train")
            # gallery = self.get_lable_data('/home/lab3/bi/0716/Veri/ai_city/tools/huafen_all_train.pkl', state="gallery")

        #地址 类别 camera_id
        #('/media/bi/文档1/19/lab/python/Vehicle_Re-identification/doc/dataset/VeRi/image_query/0002_c002_00030600_0.jpg', 2, 0)
        # print(train)
        # print(gallery)
        # print(query)
        print("data dir---------------",self.dataset_dir)
        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        #test track 应该是gallery
        #全部测试track  其中一行代表相同车辆 相同摄像头id
        #这里貌似不能这么用，代码里有int(img_id)
        # self.test_tracks = self._read_tracks(osp.join(self.dataset_dir, 'test_track.txt'))


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        pass
        # if not osp.exists(self.dataset_dir):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        # if not osp.exists(self.query_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_dir))
        # if not osp.exists(self.gallery_dir):
        #     raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    #这里需要参考arcity2020修改一下，如果是测试情况，将类别和相机id置0
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            # dataset.append((img_path, pid, 0))
        return dataset
    def get_lable_data(self,label_path,state):

        dataset = []
        if state=="train":
            train= pickle.load(open(label_path, 'rb'))
            for data in train:
                # dataset.append((data[0], int(data[1]), 0,int(data[2])))
                # img_path = '/data-tmp/shangpin/crop3/'+ data[0][-20:]
                img_path = data[0]
                # print(img_path)
                # print(data[0])
                dataset.append((img_path, int(data[1]), 0))
        if state=="query":
            # train1, query, gallery = pickle.load(open('/home/lab3/bi/0716/Veri/ai_city/tools/huafen.pkl', 'rb'))
            train, query, gallery = pickle.load(open(label_path, 'rb'))
            for data in query:
                img_path = os.path.join('/home/lab3/bi/0716/shuma/train_data', data[0])
                dataset.append((img_path,int(data[1]),0))
        if state=="gallery":
            # train1, query, gallery = pickle.load(open('/home/lab3/bi/0716/Veri/ai_city/tools/huafen.pkl', 'rb'))
            train, query, gallery = pickle.load(open(label_path, 'rb'))
            for data in gallery:
                img_path = os.path.join('/home/lab3/bi/0716/shuma/train_data', data[0])
                dataset.append((img_path,int(data[1]),0))
        # print(dataset)
        return dataset
    def get_lable_data2(self,label_path):
        #/media/bi/Data/huawei/code/tools/label_gallery.txt
        dataset = []
        with open(label_path, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            data_list = data.split('\n')
            data_list.pop()

            for num, d in enumerate(data_list):
                # print(d)
                add, label = d.split(',')
                img_path = os.path.join('/home/lab3/bi/0716/shuma/train_data', add)
                dataset.append((img_path, int(label), 0))
        return dataset
    def get_lable_data1(self,label_path,gallery_flag):
        #/media/bi/Data/huawei/code/tools/label_gallery.txt
        dataset = []
        with open(label_path, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            data_list = data.split('\n')
            # data_list.pop()

            for num, d in enumerate(data_list):
                # print(d)
                add, label = d.split(',')
                if gallery_flag:
                    # img_path = os.path.join('/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_gallery_new', add)
                    img_path = add
                    # img_path = os.path.join('/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_gallery', add)
                    # img_path = os.path.join('/home/lab3/bi/0716/shuma/test_data_A/gallery', add)
                else:
                    # img_path = os.path.join('/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_query_new', add)
                    # img_path = os.path.join('/home/lab3/bi/0716/Veri/ai_city/tools/output/crop/crop_query', add)
                    img_path = add
                    # img_path = os.path.join('/home/lab3/bi/0716/shuma/test_data_A/query', add)
                dataset.append((img_path, int(label), 0))
        return dataset
