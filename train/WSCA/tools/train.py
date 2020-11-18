# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from flyai.train_helper import upload_data, download, sava_train_model
download("data/veri.yml")
import argparse
import os
import sys
import torch

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
        "--config_file", default="data/veri.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg)





if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    print("begin train................")
    # download("pre_model/resnet50-19c8e357.pth")
    # download("dianshang/train_data.zip")
    # main()
