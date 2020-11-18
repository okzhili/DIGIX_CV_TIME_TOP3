import os
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model_fpn,make_model_layer34,make_model_layer4,make_model_resnet34

from solver import make_optimizer, WarmupMultiStepLR ,WarmupMultiStepLR_resnest
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
import pickle
from config import cfg
import os
import re
import time

#自动监视gpu当有空闲时自动训练模型
import torch


def make_label(path):
    train = pickle.load(open(os.path.join(path,'train_crop_img_add.pkl'), 'rb'))
    # '/home/lab3/bi/0716/Veri/ai_city/tools/mix_train.pkl'
    scr_train = train
    pickle.dump(scr_train, open('./datasets/mix_train.pkl', 'wb'))

    new_train = []
    temp = -1
    cnt = 0
    for i in train:
        if i[1] != temp:
            # temp = i[1]
            cnt = 1
            new_train.append(i)
        elif cnt < 20:

            cnt += 1
            new_train.append(i)
        temp = i[1]

    pickle.dump(new_train, open('./datasets/mix_train_balance.pkl', 'wb'))





if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default=" ", help="", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--data_path", default=" ", help="path to config file", type=str
    )
    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--pretrain_model_path", default=" ", help="path to config file", type=str
    )
    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--mode", default=" ", help="path to config file", type=str
    )
    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--out_dir", default=" ", help="path to config file", type=str
    )
    # parser.add_argument(
    #     # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
    #     "--size", default=" ", help="path to config file", type=str
    # )
    # parser.add_argument(
    #     # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
    #     "--model", default=" ", help="path to config file", type=str
    # )





    args = parser.parse_args()


    #relabel
    #暂时不用
    make_label(args.data_path)


    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.MODEL.PRETRAIN_PATH = args.pretrain_model_path

    cfg.OUTPUT_DIR = args.out_dir

    # cfg.INPUT.SIZE_TRAIN = [int(args.size),int(args.size)]

    # cfg.MODEL.NAME = args.model

    cfg.merge_from_list(args.opts)

    cfg.DATASETS.ROOT_DIR = 'datasets/mix_train.pkl'



    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)




    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    print("num class",num_classes)


    def get_gpu(min_memory, wait_time, min_num):
        """
        :param smallest_memory: 需求最小显存
        :param wait_time: 每次扫描等待间隔
        :param min_num: 需求最小的GPU数量
        :return: 返回满足需求的GPU列表
        """
        num = 1
        while True:
            print('正在第' + str(num) + '次扫描')
            cmd = "nvidia-smi"
            tmp = os.popen(cmd).read()
            result = re.findall('([0-9]+)MiB / ([0-9]+)MiB', tmp, re.M | re.I)
            i = 0
            can_use_gpu = []
            for k in result:
                a = int(k[1]) - int(k[0])
                print(a)
                if a > min_memory:
                    can_use_gpu.append(i)
                i += 1
            if len(can_use_gpu) > min_num - 1:
                gpus = ""
                for a in can_use_gpu:
                    gpus += str(a) + ','
                return gpus[:-1]
            num += 1
            time.sleep(wait_time)


    # a = get_gpu(1000, 10, 1)
    # print(a)
    if args.mode == "fpn":
        print("using fpn mode")
        model = make_model_fpn(cfg, num_class=num_classes)
    elif args.mode =="layer34":
        print("using layer3,4")
        model = make_model_layer34(cfg, num_class=num_classes)
    elif args.mode =="resnet34_fpn":
        print("using resnet34 mode")
        model = make_model_resnet34(cfg, num_class=num_classes)
    elif args.mode == "layer4":
        print("using layer4 mode")
        model = make_model_layer4(cfg, num_class=num_classes)
    else:
        raise Exception("mode不存在")




    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)


    if cfg.MODEL.NAME == 'resnest50' or cfg.MODEL.NAME == 'resnest50_ibn'  or \
            cfg.MODEL.NAME == 'resnest101' or cfg.MODEL.NAME == 'resnest101_ibn':
        print("using resnest50 or resnest101 warm up learning")
        scheduler = WarmupMultiStepLR_resnest(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    else:
        print("using norm model warm up learning")
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetuning......')

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query
    )
