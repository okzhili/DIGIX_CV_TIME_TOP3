# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import time
from torch.backends import cudnn

sys.path.append('../')
sys.path.append('./')
from lib.config import cfg
from lib.engine.train_net import do_train
from lib.modeling import build_model
from lib.layers import make_loss
from lib.solver import make_optimizer, build_lr_scheduler
from lib.utils.logger import setup_logger
from lib.d import make_data_loader
import numpy as np
import random
import pickle
from tqdm import tqdm



def make_train_label(cfg):
    train = []
    query = []
    gallery = []
    temp = -1
    cnt = 0

    path = os.path.join(cfg.train_data_path,'train_data/label.txt')

    with open(path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        data_list = data.split('\n')
        relabel = 0
        for num, d in enumerate(tqdm(data_list)):
            add, label = d.split(',')
            if label!=temp:
                cnt=0
                relabel+=1
                # flag = True
            elif label==temp:
                cnt+=1
            temp = label
            # if cnt==0:
            #     gallery.append((add,int(label)))
            # elif cnt==1:
            #     query.append((add, int(label)))
            # else:
            add = os.path.join(cfg.train_data_path,'train_data',add)
            train.append((add, int(relabel-1)))


    test_gallery_path = os.path.join(cfg.test_data_path,'gallery')
    test_query_path = os.path.join(cfg.test_data_path,'query')

    for num,i in enumerate(tqdm(os.listdir(test_gallery_path))):
        add = os.path.join(test_gallery_path,i)
        gallery.append((add,0))

    for num,i in enumerate(tqdm(os.listdir(test_query_path))):
        add = os.path.join(test_query_path,i)
        query.append((add,0))






    # path =os.path.join(cfg.data_path,'huafen.pkl')

    pickle.dump((train, query, gallery), open('./dianshang/huafen.pkl', 'wb'))


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)
    #test
    # model.load_param("/media/bi/Data/Mycode/car_demo/ai_city/tools/output/best.pth")
    # model = torch.load_state_dict("/media/bi/Data/Mycode/car_demo/ai_city/tools/output/best.pth")
    # model.cuda()
    optimizer = make_optimizer(cfg, model)



    loss_func = make_loss(cfg, num_classes)  # modified by gu

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        last_epoch = -1
    elif cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        start_epoch = 0
        last_epoch = -1
        model.load_param(cfg.MODEL.PRETRAIN_PATH, skip_fc=False)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'resume':
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cuda')
        start_epoch = checkpoint['epoch']
        last_epoch = start_epoch
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        #optimizer = make_optimizer(cfg, model)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('resume from {}'.format(cfg.MODEL.PRETRAIN_PATH))
    else:
        start_epoch = 0
        last_epoch = -1
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    scheduler = build_lr_scheduler(optimizer, cfg.SOLVER.LR_SCHEDULER, cfg, last_epoch)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    do_train(
        cfg,
        model,
        dataset,
        train_loader,
        val_loader,
        optimizer,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query,
        start_epoch  # add for using self trained model
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--config_file", default="configs/veri.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--train_data_path", default=" ", help="path to config file", type=str
    )

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--test_data_path", default=" ", help="path to config file", type=str
    )

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--pretrain_model_path", default=" ", help="path to config file", type=str
    )

    parser.add_argument(
        # "--config_file", default="/media/bi/Data/Mycode/car_demo/AICity2020-VOC-ReID-7c453723e6e9179d175772921f93441cfa621dc1/configs/aicity20.yml", help="path to config file", type=str
        "--out_dir", default="./dianshang", help="path to config file", type=str
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # if args.data_path != "":
    #     cfg.merge_from_file(args.data_path)
    # if args.pretrain_model_path != "":
    #     cfg.merge_from_file(args.pretrain_model_path)




    cfg.merge_from_list(args.opts)

    cfg.MODEL.PRETRAIN_PATH = args.pretrain_model_path

    cfg.OUTPUT_DIR = args.out_dir

    cfg.freeze()


    output_dir = cfg.OUTPUT_DIR



    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if True:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open("configs/veri.yml", 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)


    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)


    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    make_train_label(args)
    train(cfg)





if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    main()
